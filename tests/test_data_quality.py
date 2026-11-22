"""Consumer-visible corruption checks for the independent input/evidence audit."""
import copy
import math
from pathlib import Path
import tempfile
import unittest

from audit_data_quality import (FEATURES, FrozenEvidence, audit_inputs,
                                decode_json, git_blob_hash)


def inputs():
    settings = dict(target="bowl", scale=1., mass_kg=.18, sliding_friction=.8,
                    x_m=.51, y_m=-.09, yaw_rad=1., opening_m=.08,
                    closing_force_n=4., close_trigger_m=0., descent_speed_m_s=.1,
                    bias_x_m=.001, bias_y_m=-.001, bias_z_m=.002)
    row = {
        "settings": settings, "trial_status": "completed", "success": True,
        "warnings": [], "execution_sim_s": 5., "failure_kind": None,
        "failure_reason": None, "last_active_phase": "hold", "collisions": {"episodes": 0},
        "initial_pose": {"position_m": [.5, -.08, .75],
                         "quaternion_wxyz": [math.cos(.1), 0., 0., math.sin(.1)]},
        "actual_finger_target_sliding_frictions": [.8],
        "actual_finger_target_contact_dimensions": [3],
        "variant_features": {
            "actual_mass_kg": .18, "actual_principal_inertia_kg_m2": [.001, .001, .001],
            "collision_world_aabb_size_m": [.1, .1, .05],
            "object_collision_sliding_friction": [.8], "finger_collision_sliding_friction": [.8],
            "object_collision_friction": [[.8, .005, .0001]],
            "object_collision_contact_dimensions": [3], "mujoco_enableflags": 16,
            "grasp_offset_body_m": [.001, -.048, .044],
        },
    }
    csv = {key: str(settings[key]) for key in FEATURES}
    csv.update(x_m="0.5", y_m="-0.08", yaw_rad="0.2", success="1", execution_sim_s="5.0")
    protocol = {"contact_model": {"object_collision_condim": {"bowl": 3}},
                "study_objects": {"bowl": {"grasp_offset_body_m_at_scale_1": [0., -.047, .042]}}}
    return csv, row, protocol


class InputEvidenceAuditTests(unittest.TestCase):
    def test_settled_pose_not_nominal_pose_is_the_observable_input(self):
        csv, row, protocol = inputs()
        result = audit_inputs(csv, row, protocol)
        self.assertLess(result["max_feature_absolute_error"], 1e-12)
        for axis in ("x_m", "y_m", "yaw_rad"):
            with self.subTest(axis=axis):
                wrong = {**csv, axis: str(row["settings"][axis])}
                with self.assertRaisesRegex(ValueError, "pre-execution evidence"):
                    audit_inputs(wrong, row, protocol)

    def test_csv_label_and_postexecution_duration_must_match_raw_evidence(self):
        for key, value in (("success", "0"), ("execution_sim_s", "4.0")):
            with self.subTest(key=key):
                csv, row, protocol = inputs()
                csv[key] = value
                with self.assertRaisesRegex(ValueError, "label|duration"):
                    audit_inputs(csv, row, protocol)

    def test_compiled_mass_and_friction_cannot_be_faked_by_settings(self):
        changes = (("actual_mass_kg", .19), ("finger_collision_sliding_friction", [.4]),
                   ("object_collision_friction", [[.8, .01, .0001]]))
        for key, value in changes:
            with self.subTest(key=key):
                csv, row, protocol = inputs()
                row["variant_features"][key] = value
                with self.assertRaisesRegex(ValueError, "mass|friction"):
                    audit_inputs(csv, row, protocol)

    def test_positive_label_requires_recorded_contact_diagnostics(self):
        for key in ("actual_finger_target_sliding_frictions", "actual_finger_target_contact_dimensions"):
            with self.subTest(key=key):
                csv, row, protocol = inputs()
                row[key] = []
                with self.assertRaisesRegex(ValueError, "no recorded finger-target contact"):
                    audit_inputs(csv, row, protocol)

    def test_geometry_profile_and_latent_offset_are_not_interchangeable(self):
        csv, row, protocol = inputs()
        row["settings"]["bias_x_m"] = .003
        with self.assertRaisesRegex(ValueError, "scaled profile plus latent execution error"):
            audit_inputs(csv, row, protocol)

    def test_contact_model_family_and_numerical_warnings_are_enforced(self):
        csv, row, protocol = inputs()
        wrong = copy.deepcopy(row)
        wrong["variant_features"]["object_collision_contact_dimensions"] = [6]
        with self.assertRaisesRegex(ValueError, "contact dimensions"):
            audit_inputs(csv, wrong, protocol)
        row["warnings"] = ["BADQACC"]
        with self.assertRaisesRegex(ValueError, "Numerical warning"):
            audit_inputs(csv, row, protocol)

    def test_collision_precedence_is_not_a_causal_claim(self):
        csv, row, protocol = inputs()
        csv["success"] = "0"
        row.update(success=False, failure_reason="抬升过程中失去双侧夹持接触",
                   failure_kind="unplanned_contact")
        row["collisions"]["episodes"] = 1
        result = audit_inputs(csv, row, protocol)
        self.assertEqual(result["failure_reason_code"], "lost_bilateral_contact_during_lift")
        self.assertEqual(result["failure_kind"], "unplanned_contact")
        row["failure_kind"] = "grasp_contact_or_retention"
        with self.assertRaisesRegex(ValueError, "failure kind"):
            audit_inputs(csv, row, protocol)

    def test_json_nonfinite_literals_and_overflow_are_rejected(self):
        for content in (b'{"value": NaN}', b'{"value": Infinity}', b'{"value": 1e999}'):
            with self.subTest(content=content):
                with self.assertRaisesRegex(ValueError, "Nonfinite JSON"):
                    decode_json(content)

    def test_changed_original_evidence_is_rejected_before_auditing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "trial.json"
            original = b'{"success":true}'
            reader = FrozenEvidence.__new__(FrozenEvidence)
            reader.root, reader.files = root.resolve(), {}
            reader.blobs = {"trial.json": git_blob_hash(original)}
            path.write_bytes(b'{"success":false}')
            with self.assertRaisesRegex(ValueError, "differ from frozen snapshot"):
                reader.read("trial.json")
            self.assertEqual(reader.files, {})


if __name__ == "__main__":
    unittest.main()
