"""Validation-only error/calibration consumers; no model fitting or fake model runs."""
import math
from pathlib import Path
import tempfile
import unittest

from baseline import FEATURES
from evaluate_models import ROOT, calibration_bins, error_cases, new_output

SOURCE = {"repository": "https://example.invalid/repo", "commit": "abc",
          "raw_trials_path": "study/collection/trials.json"}


def rows(labels):
    result = []
    for index, label in enumerate(labels):
        row = {name: 0. for name in FEATURES}
        row.update(target="bowl", row_index=str(index), trial_id=f"trial_{index}",
                   group_id=f"group_{index//2}", split="validation", success=str(label))
        result.append(row)
    return result


class UnifiedEvaluationDiagnosticsTests(unittest.TestCase):
    def test_all_error_cases_use_authoritative_labels_and_success_ties(self):
        truth = rows([0, 1, 0, 1])
        cases = error_cases(truth, [.1, .9, .5, .49], SOURCE)
        self.assertEqual([case["trial_id"] for case in cases], ["trial_2", "trial_3"])
        self.assertEqual([case["error_type"] for case in cases], ["missed_failure", "missed_success"])
        self.assertEqual(cases[0]["predicted_success"], 1)
        self.assertAlmostEqual(cases[0]["log_loss_contribution"], -math.log(.5))
        self.assertAlmostEqual(cases[1]["log_loss_contribution"], -math.log(.49))
        self.assertEqual(cases[0]["success"], "0")

    def test_wrong_probability_endpoints_have_finite_defined_log_loss(self):
        cases = error_cases(rows([0, 1]), [1., 0.], SOURCE)
        self.assertEqual(len(cases), 2)
        for case in cases:
            self.assertTrue(math.isfinite(case["log_loss_contribution"]))
            self.assertAlmostEqual(case["log_loss_contribution"], -math.log(2.220446049250313e-16))

    def test_calibration_edges_and_empty_bins_are_unambiguous(self):
        bins = calibration_bins(rows([0, 1, 0]), [0., .5, 1.])
        self.assertEqual([index for index, entry in enumerate(bins) if entry["count"]], [0, 5, 9])
        self.assertEqual(bins[0]["observed_success_fraction"], 0.)
        self.assertEqual(bins[5]["mean_p_success"], .5)
        self.assertEqual(bins[5]["observed_success_fraction"], 1.)
        self.assertEqual(bins[9]["mean_p_success"], 1.)
        self.assertTrue(bins[9]["right_closed"])
        self.assertIsNone(bins[1]["mean_p_success"])
        self.assertIsNone(bins[1]["observed_success_fraction"])

    def test_diagnostics_cannot_silently_score_test_rows(self):
        truth = rows([0, 1])
        truth[0]["split"] = "test"
        with self.assertRaises(ValueError):
            calibration_bins(truth, [.1, .8])
        with self.assertRaises(ValueError):
            error_cases(truth, [.1, .8], SOURCE)

    def test_invalid_probabilities_and_alignment_are_rejected(self):
        for probabilities in ([.2], [math.nan, .8], [-.1, .8], [.1, 1.1]):
            with self.subTest(probabilities=probabilities):
                with self.assertRaises(ValueError):
                    calibration_bins(rows([0, 1]), probabilities)
                with self.assertRaises(ValueError):
                    error_cases(rows([0, 1]), probabilities, SOURCE)

    def test_results_cannot_overwrite_or_enter_frozen_locations(self):
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory)/"keep"
            existing.write_text("unchanged")
            with self.assertRaises(ValueError):
                new_output(existing)
            self.assertEqual(existing.read_text(), "unchanged")
        for destination in ("data/new_eval", "protocols/new_eval", "submissions/new_eval",
                            "results/logistic_baseline_v1/new_eval"):
            with self.subTest(destination=destination):
                with self.assertRaises(ValueError):
                    new_output(ROOT/destination)
                self.assertFalse((ROOT/destination).exists())


if __name__ == "__main__":
    unittest.main()
