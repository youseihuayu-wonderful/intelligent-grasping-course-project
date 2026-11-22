"""Audit all pilot attempts, export development-only data, and losslessly archive evidence."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
import math
from pathlib import Path
import tarfile
import time
import traceback

from audit_data_quality import decode_json, require
from audit_trace import audit_trace
from baseline import FEATURES
from design_independent_test import ROOT, payload
from pilot_evidence_audit import audit_physics_steps
from run_development_pilot import load_pilot, protected_hashes, sha_file

REPOSITORY = "https://github.com/youseihuayu-wonderful/intelligent-grasping-course-project"
RELEASE_TAG = "development-pilot-v1"


def write_json(path, value):
    with Path(path).open("xb") as stream:
        stream.write(payload(value))


def feature_values(row):
    settings = row["settings"]
    result = {name: settings[name] for name in FEATURES}
    pose = row["initial_pose"]
    xyz, quat = pose["position_m"], pose["quaternion_wxyz"]
    require(len(xyz) == 3 and len(quat) == 4, "Invalid settled pose shape")
    require(all(type(value) in (int, float) and math.isfinite(value) for value in (*xyz, *quat)),
            "Nonfinite or nonnumeric settled pose")
    norm = math.hypot(*quat)
    require(math.isclose(norm, 1, rel_tol=0, abs_tol=1e-5), "Initial quaternion must be unit length")
    w, x, y, z = (value/norm for value in quat)
    result.update(x_m=xyz[0], y_m=xyz[1], yaw_rad=math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z)))
    require(all(type(result[name]) in (int, float) and math.isfinite(result[name]) for name in FEATURES[1:]),
            "Invalid numeric predictor")
    return result


def check_variant(row, historical):
    settings, actual = row["settings"], row["variant_features"]
    require(math.isclose(actual["actual_mass_kg"], settings["mass_kg"], rel_tol=1e-12, abs_tol=1e-12),
            "Compiled mass differs from planned mass")
    for field in ("actual_principal_inertia_kg_m2", "collision_world_aabb_size_m"):
        require(len(actual[field]) == 3 and all(math.isfinite(value) and value > 0 for value in actual[field]),
                f"Invalid compiled diagnostic: {field}")
    friction = settings["sliding_friction"]
    for field in ("object_collision_sliding_friction", "finger_collision_sliding_friction"):
        require(bool(actual[field]) and all(math.isclose(value, friction, rel_tol=0, abs_tol=1e-12)
                                           for value in actual[field]), "Configured friction mismatch")
    require(bool(actual["object_collision_friction"]) and all(len(vector) == 3 and all(
        math.isclose(a, b, rel_tol=0, abs_tol=1e-12)
        for a, b in zip(vector, (friction, .005, .0001))) for vector in actual["object_collision_friction"]),
        "Full friction-vector mismatch")
    dimensions = historical["contact_model"]["object_collision_condim"][settings["target"]]
    require(bool(actual["object_collision_contact_dimensions"]) and
            all(value == dimensions for value in actual["object_collision_contact_dimensions"]),
            "Object contact model differs from original family profile")
    require(actual["mujoco_enableflags"] & 16, "MULTICCD is disabled")
    require(all(math.isclose(value, friction, rel_tol=0, abs_tol=1e-6)
                for value in row["actual_finger_target_sliding_frictions"]), "Observed friction mismatch")
    require(all(value == dimensions for value in row["actual_finger_target_contact_dimensions"]),
            "Observed contact dimensions mismatch")
    if row["success"]:
        require(bool(row["actual_finger_target_sliding_frictions"])
                and bool(row["actual_finger_target_contact_dimensions"]), "Positive row lacks native contact evidence")
    offset = historical["study_objects"][settings["target"]]["grasp_offset_body_m_at_scale_1"]
    require(len(actual["grasp_offset_body_m"]) == 3 and all(math.isclose(
        value, base*settings["scale"]+settings[f"bias_{axis}_m"], rel_tol=0, abs_tol=1e-12)
        for axis, base, value in zip("xyz", offset, actual["grasp_offset_body_m"])), "Grasp profile/bias mismatch")


def json_records(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            require(bool(line.strip()), "Blank dense evidence line")
            yield decode_json(line)


def write_csv(path, rows, fields):
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({name: json.dumps(row.get(name), sort_keys=True, allow_nan=False)
                         if isinstance(row.get(name), (list, dict)) else row.get(name)
                         for name in fields} for row in rows)


def make_archive(evidence, archive, inventory):
    require(not archive.exists() and not archive.is_symlink(), "Evidence archive must be a new path")
    # Stable headers; preserve every logical file byte, including original gzip headers.
    with archive.open("xb") as output:
        with gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|") as bundle:
                for relative in sorted(inventory):
                    path = evidence/relative
                    info = tarfile.TarInfo(f"development_pilot_v1/{relative}")
                    info.size, info.mode, info.mtime = path.stat().st_size, 0o644, 0
                    with path.open("rb") as stream:
                        bundle.addfile(info, stream)
    expected = {f"development_pilot_v1/{relative}" for relative in inventory}
    observed = set()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            require(member.isfile() and member.name in expected and member.name not in observed,
                    "Invalid or duplicate evidence archive member")
            observed.add(member.name)
            relative = member.name.removeprefix("development_pilot_v1/")
            stream = bundle.extractfile(member)
            require(stream is not None, "Missing archive member stream")
            import hashlib
            hasher = hashlib.sha256()
            size = 0
            for block in iter(lambda: stream.read(1024*1024), b""):
                hasher.update(block)
                size += len(block)
            require(hasher.hexdigest() == inventory[relative]["sha256"]
                    and size == inventory[relative]["bytes"], "Archived evidence bytes differ from source")
    require(observed == expected, "Evidence archive is incomplete")
    return {"filename": archive.name, "bytes": archive.stat().st_size, "sha256": sha_file(archive),
            "logical_files": len(inventory), "all_member_bytes_verified": True,
            "download_url": f"{REPOSITORY}/releases/download/{RELEASE_TAG}/{archive.name}",
            "release_url": f"{REPOSITORY}/releases/tag/{RELEASE_TAG}"}


def audit(evidence_dir, source_repo, plan_dir, output, archive):
    evidence, output, archive = Path(evidence_dir).resolve(), Path(output), Path(archive)
    require(not output.exists() and not output.is_symlink(), "Audit export output must be new")
    require(not archive.exists() and not archive.is_symlink(), "Evidence archive output must be new")
    require(not archive.resolve().is_relative_to(evidence), "Archive cannot be inside its own evidence tree")
    plan, policy, historical, _, _ = load_pilot(source_repo, plan_dir)
    require(decode_json((evidence/"plan.json").read_bytes()) == plan
            and decode_json((evidence/"policy.json").read_bytes()) == policy, "Collected plan/policy differs")
    manifest = decode_json((evidence/"run_manifest.json").read_bytes())
    require(manifest["protected_input_sha256"] == protected_hashes(plan_dir), "Protected inputs changed")
    before = decode_json((evidence/"runtime_before.json").read_bytes())
    after = decode_json((evidence/"runtime_after.json").read_bytes())
    require(before["input_sha256"] == after["input_sha256"] and not before["input_mismatches"]
            and not after["input_mismatches"], "Physical runtime changed during collection")
    rows = decode_json((evidence/"collection/trials.json").read_bytes())["rows"]
    metas = decode_json((evidence/"collection/instrumentation.json").read_bytes())["rows"]
    require(len(rows) == len(metas) == len(plan["trials"]) == 320, "Missing/extra aggregate attempts")
    require([row["trial_id"] for row in rows] == [spec[0] for spec in plan["trials"]], "Trial ordering/identity mismatch")
    groups = {group["condition_id"]: group["group_id"] for group in plan["condition_groups"]}
    attempts, labelled, audited, issues, cases = [], [], [], [], []
    counts, phases = defaultdict(Counter), Counter()
    selected = set()
    started = time.monotonic()
    for index, (spec, row, meta) in enumerate(zip(plan["trials"], rows, metas)):
        trial_id, settings, metadata = spec
        record = {"row_index": index, "trial_id": trial_id, "group_id": groups[metadata["condition_id"]],
                  "split": "train", "target": settings["target"], "raw_success": row.get("success"),
                  "trial_status": row["trial_status"], "audit_passed": False}
        try:
            require(row["settings"] == settings and row["metadata"] == metadata, "Raw inputs differ from frozen spec")
            require(meta["trial_id"] == trial_id and meta["status"] == "completed"
                    and meta["read_only_checks_passed"] and meta["python_numeric_warning_count"] == 0,
                    "Instrumentation error or Python numeric warning")
            require(row["trial_status"] == "completed" and type(row.get("success")) is bool
                    and row["warnings"] == [], "Unusable physical attempt")
            raw_path = evidence/"collection/trials"/f"{trial_id}.json"
            meta_path = evidence/"collection/execution_meta"/f"{trial_id}.json"
            require(decode_json(raw_path.read_bytes()) == row and decode_json(meta_path.read_bytes()) == meta,
                    "Individual and aggregate evidence disagree")
            check_variant(row, historical)
            sparse_path = evidence/"collection/traces"/f"{trial_id}.json.gz"
            sparse = audit_trace(row, decode_json(gzip.decompress(sparse_path.read_bytes())))
            dense_path = evidence/"collection/physics_steps"/f"{trial_id}.jsonl.gz"
            dense = audit_physics_steps(row, json_records(dense_path))
            require(dense["step_count"] == meta["dense_samples"], "Dense sample count differs from worker instrumentation")
            features = feature_values(row)
            dataset_row = {"row_index": index, "trial_id": trial_id, "group_id": record["group_id"],
                           "split": "train", **features, "success": int(row["success"]),
                           "execution_sim_s": row["execution_sim_s"],
                           "evidence_source": f"collection/trials/{trial_id}.json", "role": "development_pilot_only"}
            labelled.append(dataset_row)
            record.update(audit_passed=True, dense_samples=dense["step_count"],
                          sparse_samples=sparse["sample_count"], raw_record_sha256=sha_file(raw_path),
                          dense_evidence_sha256=sha_file(dense_path), sparse_evidence_sha256=sha_file(sparse_path),
                          verified_hold_s=row["verified_hold_s"], dense_audit=dense)
            counts[settings["target"]]["successes" if row["success"] else "failures"] += 1
            if not row["success"]:
                phases[row["last_active_phase"]] += 1
            case_key = (settings["target"], row["success"])
            if case_key not in selected:
                selected.add(case_key)
                cases.append({"trial_id": trial_id, "target": settings["target"], "success": row["success"],
                              "features": features, "last_active_phase": row["last_active_phase"],
                              "recorded_failure_kind": row["failure_kind"], "postconditions": row["postconditions"],
                              "dense_audit": dense, "raw_record_path": dataset_row["evidence_source"]})
        except (ValueError, KeyError, TypeError, OSError, EOFError) as error:
            issue = {"trial_id": trial_id, "error_type": type(error).__name__, "message": str(error),
                     "traceback": traceback.format_exc()}
            issues.append(issue)
            record["audit_error"] = {key: value for key, value in issue.items() if key != "traceback"}
        attempts.append({**record, "dense_audit": record.get("dense_audit")})
        audited.append(record)
    output.mkdir(parents=True, exist_ok=False)
    fields = ["row_index", "trial_id", "group_id", "split", *FEATURES, "success", "execution_sim_s", "evidence_source", "role"]
    write_csv(output/"dataset.csv", labelled, fields)
    write_csv(output/"trial_audit.csv", audited, ["row_index", "trial_id", "group_id", "split", "target", "raw_success",
              "trial_status", "audit_passed", "dense_samples", "sparse_samples", "verified_hold_s",
              "raw_record_sha256", "dense_evidence_sha256", "sparse_evidence_sha256", "dense_audit", "audit_error"])
    write_json(output/"attempts.json", attempts)
    write_json(output/"audit_issues.json", issues)
    write_json(output/"case_studies.json", cases)
    write_json(output/"collection_summary.json", decode_json((evidence/"collection/summary.json").read_bytes()))
    write_json(output/"runtime_before.json", before)
    write_json(output/"runtime_after.json", after)
    inventory = {}
    for path in sorted(evidence.rglob("*")):
        if path.is_file():
            require(not path.is_symlink(), "Symlink in raw evidence tree")
            inventory[path.relative_to(evidence).as_posix()] = {"bytes": path.stat().st_size, "sha256": sha_file(path)}
    write_json(output/"evidence_manifest.json", {"archive_prefix": "development_pilot_v1", "files": inventory})
    archive_record = make_archive(evidence, archive, inventory)
    write_json(output/"archive.json", archive_record)
    total_successes = sum(row["success"] for row in labelled)
    report = {"status": "passed_pilot_evidence_quality_checks" if not issues and len(labelled) == 320 else "failed_pilot_evidence_quality_checks",
              "role": "development_pilot_only_not_final_test", "planned_attempts": 320,
              "attempt_records": len(rows), "verified_labelled_rows": len(labelled),
              "successes": total_successes, "failures": len(labelled)-total_successes,
              "success_rate_on_usable_pilot_trials": total_successes/len(labelled) if labelled else None,
              "families": {target: {"attempts": 64, "successes": counts[target]["successes"],
                          "failures": counts[target]["failures"]} for target in policy["family_mass_levels_kg"]},
              "failure_last_active_phase_counts": dict(phases), "audit_issues": len(issues),
              "dense_samples_verified": sum(record.get("dense_samples", 0) for record in audited),
              "sparse_samples_verified": sum(record.get("sparse_samples", 0) for record in audited),
              "physics_step_s": .002, "runtime_input_files_matched": before["runtime_input_files_checked"],
              "protected_inputs_unchanged": True, "final_test_plan_opened": False,
              "final_test_trials_executed": 0, "models_trained_or_scored": 0,
              "dataset_sha256": sha_file(output/"dataset.csv"), "archive": archive_record,
              "audit_and_archive_wall_s": time.monotonic()-started,
              "implementation_sha256": {name: sha_file(ROOT/name) for name in
                  ("audit_development_pilot.py", "pilot_evidence_audit.py", "audit_trace.py")},
              "limits": ["Nonrepresentative four-corner development pilot, not final-test or classifier performance.",
                         "20 condition groups with paired commands/errors; 320 trials are not 320 independent conditions.",
                         "Every saved physics tick is checked, but contacts come from the same simulator solver and reference values from the controller, not independent real-world sensors.",
                         "Per-step measurements reconstruct the declared predicates, not the full dynamics or sim-to-real validity.",
                         "Raw original labels are preserved; unusable/unverifiable attempts remain explicit issues and never become convenient failed grasps."]}
    write_json(output/"report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), flush=True)
    require(not issues and len(labelled) == 320, "Evidence retained and issues published; pilot quality gate did not pass")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--plan-dir", type=Path, default=ROOT/"protocols/independent_test_v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    audit(args.evidence_dir, args.source_repo, args.plan_dir, args.output, args.archive)


if __name__ == "__main__":
    main()
