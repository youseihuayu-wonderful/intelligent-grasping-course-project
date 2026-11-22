"""Spawn-safe, read-only evidence collection around the unchanged pilot physics."""

from contextlib import ExitStack
import gzip
import importlib
import json
import math
from pathlib import Path
import sys
import traceback
import warnings


_PHYSICS_STEP_S = 0.002


def run_trial(spec: list, source_repo: str, directory: str) -> dict:
    """Run one original trial, retaining dense observations and explicit faults."""
    trial_id, settings, metadata = spec
    root = Path(directory)
    instrumentation = {
        "trial_id": trial_id,
        "status": "internal_error",
        "physics_step_s": _PHYSICS_STEP_S,
        "dense_samples": 0,
        "read_only_checks_passed": True,
        "python_warnings": [],
        "python_numeric_warning_count": 0,
    }
    raw = None
    original_path = sys.path[:]
    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        try:
            source = Path(source_repo).resolve()
            sys.path.insert(0, str(source))
            # Native libraries are deliberately imported only in the spawned worker.
            import numpy as np
            import mujoco

            physics = importlib.import_module("grasp_study_physics")
            if Path(physics.__file__).resolve() != source / "grasp_study_physics.py":
                raise RuntimeError("Imported physics module is not from source_repo")
            original_simulation = physics.ExplorationSimulation
            with ExitStack() as streams:
                class ObservedSimulation(original_simulation):
                    def __init__(self, original_settings):
                        # Do not observe or intercept construction/reset itself.
                        super().__init__(original_settings)
                        self._evidence_dt = float(self.model.opt.timestep)
                        if self._evidence_dt != _PHYSICS_STEP_S:
                            raise RuntimeError("Original physics timestep is not 0.002 s")
                        if float(self.data.time) != 0.0:
                            raise RuntimeError("Original constructor did not reset simulation time")
                        self._evidence_snapshots = tuple(
                            (getattr(self.data, name),
                             np.empty_like(getattr(self.data, name)),
                             np.empty(getattr(self.data, name).shape, dtype=np.bool_))
                            for name in ("qpos", "qvel", "ctrl")
                        )
                        self._evidence_force = np.empty(6)
                        path = root / "physics_steps" / f"{trial_id}.jsonl.gz"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        self._evidence_stream = streams.enter_context(
                            gzip.open(path, "xt", encoding="utf-8")
                        )
                        header = {
                            "record_type": "header",
                            "schema_version": 1,
                            "trial_id": trial_id,
                            "physics_step_s": self._evidence_dt,
                            "target_body_id": int(self.skill.target_body),
                            "finger_body_ids": sorted(int(body) for body in self.skill.fingers),
                            "table_body_id": int(self.skill.table),
                            "contact_row_columns": [
                                "target_geom_id", "other_geom_id", "other_body_id",
                                "efc_address", "normal_force_n",
                            ],
                            "finger_normal_force_threshold_n": 0.01,
                            "minimum_lift_m": 0.10,
                            "relative_position_tolerance_m": 0.01,
                        }
                        self._evidence_stream.write(json.dumps(header, allow_nan=False) + "\n")

                    def _assert_evidence_unchanged(self, time_before):
                        unchanged = float(self.data.time) == time_before
                        for current, snapshot, equal in self._evidence_snapshots:
                            np.equal(current, snapshot, out=equal)
                            unchanged = bool(equal.all()) and unchanged
                        if not unchanged:
                            instrumentation["read_only_checks_passed"] = False
                            raise RuntimeError("Dense observer changed qpos, qvel, ctrl, or time")

                    def _record_evidence(self):
                        time_before = float(self.data.time)
                        for current, snapshot, _ in self._evidence_snapshots:
                            np.copyto(snapshot, current)
                        try:
                            skill = self.skill
                            contacts = []
                            for index, contact in enumerate(self.data.contact):
                                geom_a, geom_b = int(contact.geom1), int(contact.geom2)
                                body_a = int(self.model.geom_bodyid[geom_a])
                                body_b = int(self.model.geom_bodyid[geom_b])
                                if skill.target_body == body_a:
                                    target_geom, other_geom, other_body = geom_a, geom_b, body_b
                                elif skill.target_body == body_b:
                                    target_geom, other_geom, other_body = geom_b, geom_a, body_a
                                else:
                                    continue
                                address = int(contact.efc_address)
                                normal_force = None
                                if address >= 0:
                                    mujoco.mj_contactForce(
                                        self.model, self.data, index, self._evidence_force
                                    )
                                    normal_force = float(self._evidence_force[0])
                                contacts.append([
                                    target_geom, other_geom, other_body, address, normal_force,
                                ])
                            target_position = self.data.xpos[skill.target_body]
                            tcp_position = self.data.site_xpos[skill.tcp]
                            # Query the original predicates, rather than substituting
                            # the recorder's own implementation for controller evidence.
                            predicates = skill.grasp_postconditions()
                            record = {
                                "record_type": "step",
                                "step_index": instrumentation["dense_samples"] + 1,
                                "time_s": time_before,
                                "phase": skill.phase,
                                "status": skill.status,
                                "target_position_m": target_position.tolist(),
                                "target_quaternion_wxyz": self.data.xquat[skill.target_body].tolist(),
                                "tcp_position_m": tcp_position.tolist(),
                                "reference_z_m": (None if skill.reference_z is None
                                                  else float(skill.reference_z)),
                                "hold_relative_m": (None if skill.hold_relative is None
                                                    else skill.hold_relative.tolist()),
                                "target_contacts": contacts,
                                "controller_predicates": predicates,
                                "read_only_state_unchanged": True,
                            }
                            encoded = json.dumps(record, allow_nan=False) + "\n"
                        finally:
                            # Guard failures must remain execution errors, not failed grasps.
                            self._assert_evidence_unchanged(time_before)
                        self._evidence_stream.write(encoded)
                        instrumentation["dense_samples"] += 1

                    def step(self, duration_s=0.02):
                        if duration_s != self._evidence_dt:
                            raise ValueError("Dense observer requires exactly one physics tick per step")
                        before = float(self.data.time)
                        super().step(duration_s)
                        elapsed = float(self.data.time) - before
                        if not math.isclose(elapsed, self._evidence_dt, rel_tol=0.0, abs_tol=1e-12):
                            raise RuntimeError("Original step did not advance exactly one physics tick")
                        self._record_evidence()

                physics.ExplorationSimulation = ObservedSimulation
                try:
                    raw = physics.run_one(spec, root)
                finally:
                    physics.ExplorationSimulation = original_simulation
            instrumentation["status"] = "completed"
        except Exception as exc:
            error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            instrumentation["error"] = error
            raw = {
                "trial_id": trial_id,
                "settings": settings,
                "metadata": metadata,
                "trial_status": "internal_error",
                "success": None,
                "failure_kind": "internal_error",
                "failure_reason": str(exc),
                "error": error,
            }
            # Do not overwrite an original raw row if source evidence was saved
            # before a later trace/close error. The returned attempt is still a nonlabel.
            path = root / "trials" / f"{trial_id}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                with path.open("x", encoding="utf-8") as stream:
                    json.dump(raw, stream, allow_nan=False)
                    stream.write("\n")
        finally:
            sys.path[:] = original_path
            instrumentation["python_warnings"] = [
                {
                    "category": warning.category.__name__,
                    "message": str(warning.message),
                    "filename": warning.filename,
                    "lineno": warning.lineno,
                }
                for warning in captured_warnings
            ]
            instrumentation["python_numeric_warning_count"] = sum(
                issubclass(warning.category, RuntimeWarning) for warning in captured_warnings
            )
            path = root / "execution_meta" / f"{trial_id}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as stream:
                json.dump(instrumentation, stream, allow_nan=False)
                stream.write("\n")
    return {"raw": raw, "instrumentation": instrumentation}
