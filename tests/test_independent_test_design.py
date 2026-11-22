"""Planning, blinding and tamper checks; no simulator, fitted model or labels."""
import copy
from pathlib import Path
import tempfile
import unittest

from design_independent_test import (PILOT_SEED, PROTOCOL_ID, ROOT, build_plan,
                                     check_output_paths, digest, make_policy,
                                     payload, validate_plan, verify_bundle)


def source_fixture():
    masses = {
        "bowl": [.06, .18, .3, .45], "bottle": [.06, .18, .3, .45],
        "spoon": [.015, .025, .04, .06], "glass": [.08, .15, .25, .35],
        "pen": [.005, .01, .02, .03],
    }
    historical = {
        "scale_levels": [.8, 1., 1.2], "sliding_friction_levels": [.2, .4, .8, 1.2],
        "study_objects": {name: {"mass_levels_kg": values} for name, values in masses.items()},
        "pose_bounds": {"x_m": [.475, .525], "y_m": [-.1, -.06], "yaw_rad": [-.15, .15]},
        "control_levels": {"opening_m": [.04, .08], "closing_force_n": [2., 4., 8.],
                           "close_trigger_m": [0., .01, .02], "descent_speed_m_s": [.04, .1, .2]},
        "baseline_control": {"opening_m": .08, "closing_force_n": 4.,
                             "close_trigger_m": 0., "descent_speed_m_s": .1},
        "condition_groups": [], "trials": [],
    }
    for index, (name, values) in enumerate(masses.items()):
        nominal = dict(target=name, scale=.8, mass_kg=values[0], sliding_friction=.2,
                       x_m=.5, y_m=-.08, yaw_rad=0.)
        historical["condition_groups"].append({"condition_id": f"old.{name}", "nominal_settings": nominal})
        historical["trials"].append([f"old.{name}.trial", nominal, {"nuisance_seed": 100+index}])
    return historical, {"physical_collector_sha256": "0"*64}


class IndependentTestDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.historical, cls.environment = source_fixture()
        cls.policy = make_policy(cls.historical, cls.environment)
        cls.pilot = build_plan(cls.policy, cls.historical, "pilot", PILOT_SEED)
        cls.final = build_plan(cls.policy, cls.historical, "final", "a"*64, other_plans=(cls.pilot,))

    def audit_final(self, plan=None, historical=None):
        return validate_plan(self.final if plan is None else plan,
                             self.historical if historical is None else historical,
                             self.policy, other_plans=(self.pilot,))

    def test_final_has_complete_grid_paired_controls_and_unseen_conditions(self):
        result = self.audit_final()
        self.assertEqual(result["condition_groups"], 240)
        self.assertEqual(result["planned_attempts"], 3840)
        self.assertEqual(set(result["distinct_controls_per_family"].values()), {54})
        self.assertGreaterEqual(result["minimum_matched_property_xy_separation_m"], .005)
        self.assertFalse(result["physical_initialization_checked"])
        self.assertEqual({spec[2]["split"] for spec in self.final["trials"]}, {"test"})

    def test_development_pilot_cannot_be_presented_as_final_test(self):
        result = validate_plan(self.pilot, self.historical, self.policy)
        self.assertEqual(result["planned_attempts"], 320)
        self.assertEqual({spec[2]["split"] for spec in self.pilot["trials"]}, {"train"})
        wrong = copy.deepcopy(self.pilot)
        wrong["role"] = "final"
        with self.assertRaises(ValueError):
            self.audit_final(wrong)

    def test_missing_trials_and_changed_candidate_commands_are_rejected(self):
        for change in ("missing", "control"):
            with self.subTest(change=change):
                wrong = copy.deepcopy(self.final)
                if change == "missing":
                    wrong["trials"].pop()
                else:
                    wrong["trials"][0][1]["closing_force_n"] = 99.
                with self.assertRaises(ValueError):
                    self.audit_final(wrong)

    def test_unpaired_execution_noise_and_partition_change_are_rejected(self):
        for change in ("bias", "partition", "seed"):
            with self.subTest(change=change):
                wrong = copy.deepcopy(self.final)
                spec = wrong["trials"][0]
                if change == "bias":
                    spec[1]["bias_x_m"] += .001
                elif change == "partition":
                    spec[2]["split"] = "train"
                else:
                    spec[2]["nuisance_seed"] = 100
                with self.assertRaises(ValueError):
                    self.audit_final(wrong)

    def test_newly_discovered_historical_pose_reuse_blocks_approval(self):
        historical = copy.deepcopy(self.historical)
        context = self.final["condition_groups"][0]["nominal_settings"]
        historical["condition_groups"].append({"condition_id": "old.extra", "nominal_settings": context})
        with self.assertRaises(ValueError):
            self.audit_final(historical=historical)

    def test_outcomes_and_model_predictions_cannot_enter_a_plan(self):
        for field in ("success", "final_pose", "p_success"):
            with self.subTest(field=field):
                wrong = copy.deepcopy(self.final)
                wrong["trials"][0][2][field] = 1
                with self.assertRaises(ValueError):
                    self.audit_final(wrong)

    def test_policy_drift_invalidates_a_frozen_plan(self):
        wrong = copy.deepcopy(self.policy)
        wrong["evaluation"]["threshold"] = .7
        with self.assertRaises(ValueError):
            validate_plan(self.final, self.historical, wrong, other_plans=(self.pilot,))

    def test_private_plan_cannot_be_written_inside_git(self):
        with tempfile.TemporaryDirectory(dir=ROOT, prefix=".design_guard_") as public_parent:
            public = Path(public_parent) / "public"
            private = ROOT / "do-not-create-private-plan"
            with self.assertRaises(ValueError):
                check_output_paths(public, private)
            self.assertFalse(private.exists())

    def test_existing_public_artifact_is_not_overwritten(self):
        with tempfile.TemporaryDirectory(dir=ROOT, prefix=".design_guard_") as public_parent:
            with tempfile.TemporaryDirectory() as private_parent:
                public = Path(public_parent) / "existing"
                public.write_bytes(b"keep this")
                with self.assertRaises(ValueError):
                    check_output_paths(public, Path(private_parent) / "private")
                self.assertEqual(public.read_bytes(), b"keep this")

    def bundle_fixture(self, public, private):
        pilot_bytes, final_bytes = payload(self.pilot), payload(self.final)
        validation = {"pilot": validate_plan(self.pilot, self.historical, self.policy),
                      "final": self.audit_final(), "final_plan_sha256": digest(final_bytes),
                      "policy_sha256": digest(payload(self.policy)),
                      "status": "passed_design_checks_only_not_physics", "labels_generated": 0,
                      "models_trained_or_scored": 0}
        commitment = {"schema_version": 1, "protocol_id": PROTOCOL_ID,
                      "policy_sha256": digest(payload(self.policy)), "pilot_plan_sha256": digest(pilot_bytes),
                      "final_plan_sha256": digest(final_bytes), "final_plan_bytes": len(final_bytes),
                      "condition_groups": 240, "planned_attempts": 3840,
                      "status": "planned_not_collected_private_inputs_withheld"}
        for name, value in (("policy.json", self.policy), ("pilot_plan.json", self.pilot),
                            ("final_commitment.json", commitment), ("design_validation.json", validation)):
            (public / name).write_bytes(payload(value))
        (private / "final_plan.json").write_bytes(final_bytes)
        (private / "final_plan.json").chmod(0o600)
        return commitment

    def test_private_plan_tamper_is_detected_without_publishing_inputs(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            public, private = Path(a), Path(b)
            self.bundle_fixture(public, private)
            result = verify_bundle(public, private, self.historical, self.environment)
            self.assertEqual(result["labels_generated"], 0)
            self.assertEqual(result["final"]["planned_attempts"], 3840)
            wrong = copy.deepcopy(self.final)
            wrong["trials"][0][1]["x_m"] += .001
            (private / "final_plan.json").write_bytes(payload(wrong))
            with self.assertRaises(ValueError):
                verify_bundle(public, private, self.historical, self.environment)

    def test_public_commitment_cannot_contain_private_seed(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            public, private = Path(a), Path(b)
            commitment = self.bundle_fixture(public, private)
            commitment["seed_hex"] = self.final["seed_hex"]
            (public / "final_commitment.json").write_bytes(payload(commitment))
            with self.assertRaises(ValueError):
                verify_bundle(public, private, self.historical, self.environment)

    def test_relaxed_private_file_permissions_block_approval(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            public, private = Path(a), Path(b)
            self.bundle_fixture(public, private)
            (private / "final_plan.json").chmod(0o644)
            with self.assertRaises(ValueError):
                verify_bundle(public, private, self.historical, self.environment)


if __name__ == "__main__":
    unittest.main()
