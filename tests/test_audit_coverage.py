"""Independent synthetic protocol faults, without source-audit imports or data edits."""

import copy
import hashlib
from itertools import product
import json
import unittest

from audit_coverage import audit_coverage


COMMANDS = ("opening_m", "closing_force_n", "close_trigger_m", "descent_speed_m_s")
LEVELS = {"opening_m": [.04, .08], "closing_force_n": [2., 4., 8.],
          "close_trigger_m": [0., .01, .02], "descent_speed_m_s": [.04, .1, .2]}


def fixture():
    """One complete 48-condition family, with known paired latent conflicts."""
    pool = [dict(zip(COMMANDS, values)) for values in product(*LEVELS.values())]
    protocol = {
        "protocol_id": "fixture", "condition_groups": [], "trials": [],
        "condition_group_count": 48, "planned_attempts": 768,
        "controls_per_condition": 8, "paired_draws_per_condition": 2,
        "group_split_counts": {"train": 32, "validation": 8, "test": 8},
        "family_group_split_counts": {"bowl": {"train": 32, "validation": 8, "test": 8}},
        "planned_split_attempts": {"train": 512, "validation": 128, "test": 128},
        "scale_levels": [.8, 1., 1.2], "sliding_friction_levels": [.2, .4, .8, 1.2],
        "study_objects": {"bowl": {"mass_levels_kg": [.06, .18, .3, .45]}},
        "control_levels": LEVELS, "control_pool": pool,
        "pose_bounds": {"x_m": [.475, .525], "y_m": [-.1, -.06], "yaw_rad": [-.15, .15]},
    }
    csv_rows, raw_rows = [], []
    for index, (scale, mass, friction) in enumerate(product([.8, 1., 1.2], [.06, .18, .3, .45], [.2, .4, .8, 1.2])):
        split, local, n = (("train", index, 32) if index < 32 else
                           ("validation", index - 32, 8) if index < 40 else
                           ("test", index - 40, 8))
        condition = f"fixture.bowl_{index:02d}"
        nominal = {"target": "bowl", "scale": scale, "mass_kg": mass, "sliding_friction": friction}
        for axis, (low, high) in protocol["pose_bounds"].items():
            nominal[axis] = low + (high - low) * (local + .5) / n
        controls = [pool[(index * 8 + control) % len(pool)].copy() for control in range(8)]
        protocol["condition_groups"].append({"condition_id": condition, "split": split,
                                             "nominal_settings": nominal, "controls": controls})
        tokens = ["condition:" + condition,
                  "nominal:" + json.dumps(nominal, sort_keys=True, separators=(",", ":"))]
        group_id = "group_" + hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()
        for control_id, command in enumerate(controls):
            for draw in (0, 1):
                trial_id = f"{condition}.control_{control_id}.draw_{draw}"
                settings = {**nominal, **command, "bias_x_m": .001 * (draw + 1),
                            "bias_y_m": -.001, "bias_z_m": .0005}
                metadata = {"condition_id": condition, "split": split, "control_id": control_id,
                            "nuisance_seed": index * 2 + draw, "protocol_id": "fixture"}
                protocol["trials"].append([trial_id, settings.copy(), metadata.copy()])
                raw_rows.append({"trial_id": trial_id, "settings": settings.copy(), "metadata": metadata.copy(),
                                 "initial_state_fingerprint": f"state-{index}", "success": bool(draw),
                                 "failure_kind": None if draw else "unplanned_contact",
                                 "failure_reason": None if draw else "recorded contact co-occurrence"})
                csv_rows.append({"trial_id": trial_id, "group_id": group_id, "split": split,
                                 **{key: str(value) for key, value in {**nominal, **command}.items()},
                                 "success": str(draw)})
    return csv_rows, raw_rows, protocol


class CoverageAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = fixture()

    def setUp(self):
        self.csv, self.raw, self.protocol = copy.deepcopy(self.source)

    def audit(self):
        return audit_coverage(self.csv, self.raw, self.protocol)

    def test_unordered_rows_cover_complete_plan_and_latent_conflicts_are_not_corruption(self):
        self.raw.reverse()
        self.csv = self.csv[1::2] + self.csv[::2]
        result = self.audit()
        self.assertEqual(result["trial_count"], 768)
        self.assertEqual(result["group_count"], 48)
        self.assertEqual(result["families"]["bowl"]["property_grid_combinations"], 48)
        self.assertEqual(result["families"]["bowl"]["distinct_control_configurations"], 54)
        self.assertEqual(result["observable_feature_duplicates"], {
            "distinct_vectors": 384, "duplicate_vector_groups": 384,
            "rows_in_duplicate_vectors": 768, "redundant_rows": 384,
            "conflicting_label_groups": 384, "cross_split_groups": 0})
        self.assertEqual(result["exact_physical_input_duplicates"]["duplicate_vector_groups"], 0)
        self.assertEqual(result["training_only_diagnostics"]["row_count"], 512)
        self.assertEqual(result["training_only_diagnostics"]["family_label_counts"], {"bowl": {"0": 256, "1": 256}})
        self.assertEqual(result["nominal_pose_strata"]["bowl"]["test"]["yaw_rad"]["strata"], 8)

    def test_missing_raw_trial_is_rejected_even_if_inputs_are_unordered(self):
        removed = self.raw.pop(15)["trial_id"]
        self.raw.reverse()
        with self.assertRaisesRegex(ValueError, "raw: planned trial ID mismatch.*" + removed):
            self.audit()

    def test_duplicate_or_unplanned_trial_is_rejected(self):
        self.csv.append(self.csv[0].copy())
        with self.assertRaisesRegex(ValueError, "CSV: duplicate trial_id"):
            self.audit()
        self.csv.pop()
        self.raw[0]["trial_id"] = "not-in-the-frozen-plan"
        with self.assertRaisesRegex(ValueError, "raw: planned trial ID mismatch"):
            self.audit()

    def test_changed_raw_control_does_not_pass_by_matching_only_condition_counts(self):
        self.raw[0]["settings"]["closing_force_n"] = 99.
        with self.assertRaisesRegex(ValueError, "raw settings differ from planned spec"):
            self.audit()

    def test_plan_with_repeated_control_is_not_eight_distinct_configurations(self):
        controls = self.protocol["condition_groups"][0]["controls"]
        controls[7] = controls[0].copy()
        with self.assertRaisesRegex(ValueError, "controls are not eight distinct"):
            self.audit()

    def test_crossed_condition_split_is_rejected_without_using_outcomes(self):
        self.csv[0]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "condition split differs from frozen protocol"):
            self.audit()

    def test_condition_cannot_be_divided_between_frozen_groups(self):
        self.csv[1]["group_id"] = "group_wrong"
        with self.assertRaisesRegex(ValueError, "condition maps to multiple frozen groups"):
            self.audit()

    def test_reconstruction_rejects_consistent_but_wrong_group_identifier(self):
        for row in self.csv[:16]:
            row["group_id"] = "arbitrary-consistent-id"
        with self.assertRaisesRegex(ValueError, "reconstructed label-independent group_id mismatch"):
            self.audit()

    def test_unpaired_nuisance_seed_is_rejected_even_when_raw_matches_plan(self):
        self.raw[2]["metadata"]["nuisance_seed"] = 9999
        self.protocol["trials"][2][2]["nuisance_seed"] = 9999
        with self.assertRaisesRegex(ValueError, "two paired nuisance draws"):
            self.audit()

    def test_shared_seed_requires_shared_bias_across_controls(self):
        self.raw[2]["settings"]["bias_x_m"] = .003
        self.protocol["trials"][2][1]["bias_x_m"] = .003
        with self.assertRaisesRegex(ValueError, "shared biases across controls"):
            self.audit()

    def test_identical_hidden_draws_are_not_legitimate_observable_conflicts(self):
        for index in range(1, 16, 2):
            self.raw[index]["settings"]["bias_x_m"] = .001
            self.protocol["trials"][index][1]["bias_x_m"] = .001
        with self.assertRaisesRegex(ValueError, "exact physical-input duplication"):
            self.audit()

    def test_planned_grid_loss_is_rejected_even_with_unchanged_group_counts(self):
        groups = self.protocol["condition_groups"]
        groups[1]["nominal_settings"]["sliding_friction"] = groups[0]["nominal_settings"]["sliding_friction"]
        with self.assertRaisesRegex(ValueError, "property grid must cover all 48 combinations once"):
            self.audit()

    def test_pose_bounds_do_not_substitute_for_splitwise_latin_hypercube_coverage(self):
        groups = self.protocol["condition_groups"]
        groups[1]["nominal_settings"]["x_m"] = groups[0]["nominal_settings"]["x_m"]
        with self.assertRaisesRegex(ValueError, "nominal x_m Latin-hypercube strata incomplete"):
            self.audit()

    def test_declared_family_split_counts_are_checked(self):
        self.protocol["family_group_split_counts"]["bowl"] = {"train": 31, "validation": 9, "test": 8}
        with self.assertRaisesRegex(ValueError, "bowl.group_split_counts"):
            self.audit()

    def test_training_diagnostics_do_not_mine_held_out_outcomes(self):
        before = self.audit()["training_only_diagnostics"]
        for raw, csv in zip(self.raw, self.csv):
            if csv["split"] != "train":
                raw["success"] = not raw["success"]
                csv["success"] = str(int(raw["success"]))
                raw["failure_kind"] = "held-out-only-category"
                raw["failure_reason"] = "held-out-only-reason"
        after = self.audit()["training_only_diagnostics"]
        self.assertEqual(before, after)
        self.assertNotIn("held-out-only-category", after["failure_kind_counts"])

    def test_csv_label_conflict_with_raw_is_not_a_latent_noise_exception(self):
        self.csv[0]["success"] = "1"
        with self.assertRaisesRegex(ValueError, "CSV label differs from raw success"):
            self.audit()


if __name__ == "__main__":
    unittest.main()
