"""Tiny deterministic checks for whole-group, paired validation uncertainty."""
import copy
import json
import math
import unittest

import numpy as np

from baseline import validation_metrics
from evaluation_uncertainty import METRICS, grouped_bootstrap


def rows_for(groups):
    return [
        {"trial_id": f"{group}-{index}", "group_id": group, "target": family,
         "split": "validation", "success": str(label)}
        for group, family, labels in groups
        for index, label in enumerate(labels)
    ]


class GroupedBootstrapTests(unittest.TestCase):
    def assert_interval(self, interval, estimate, lower, upper, valid, undefined):
        self.assertAlmostEqual(interval["estimate"], estimate)
        self.assertAlmostEqual(interval["lower"], lower)
        self.assertAlmostEqual(interval["upper"], upper)
        self.assertEqual(interval["valid_resamples"], valid)
        self.assertEqual(interval["undefined_resamples"], undefined)

    def test_hand_computable_metrics_and_candidate_minus_reference(self):
        rows = rows_for([("only", "bowl", [0, 0, 1, 1])])
        result = grouped_bootstrap(
            rows, {"reference": [0.5] * 4, "candidate": [0.1, 0.2, 0.8, 0.9]},
            reference="reference", resamples=5, seed=7,
        )
        self.assertEqual(METRICS, ("log_loss", "brier", "roc_auc", "balanced_accuracy", "failure_recall"))
        candidate = {"log_loss": -math.log(0.72) / 2, "brier": 0.025,
                     "roc_auc": 1.0, "balanced_accuracy": 1.0, "failure_recall": 1.0}
        reference = {"log_loss": math.log(2), "brier": 0.25,
                     "roc_auc": 0.5, "balanced_accuracy": 0.5, "failure_recall": 0.0}
        summary = result["summary"]
        for metric in METRICS:
            with self.subTest(metric=metric):
                for model, expected in (("candidate", candidate), ("reference", reference)):
                    interval = summary["models"][model][metric]
                    self.assert_interval(interval, expected[metric], expected[metric], expected[metric], 5, 0)
                    self.assertIsNone(interval["estimate_undefined_reason"])
                    self.assertEqual(interval["undefined_reasons"], {})
                    self.assertEqual(interval["interval_condition"], "all_resamples_defined")
                contrast = candidate[metric] - reference[metric]
                self.assert_interval(summary["differences"]["candidate"][metric], contrast, contrast, contrast, 5, 0)
                expected_direction = "lower_is_better" if metric in {"log_loss", "brier"} else "higher_is_better"
                self.assertEqual(summary["metric_directions"][metric], expected_direction)
        self.assertNotIn("reference", summary["differences"])
        self.assertEqual(result["group_order"], ["only"])
        self.assertEqual(result["group_weights"], [[1]] * 5)
        json.dumps(result, allow_nan=False)

    def test_unequal_size_groups_and_family_counts_preserved_for_all_models(self):
        rows = rows_for([
            ("g2", "bowl", [0, 1, 1]), ("g1", "bowl", [0, 1]),
            ("g5", "glass", [0]), ("g3", "glass", [0, 0, 1]),
            ("g4", "glass", [1, 1]), ("g0", "pen", [0, 1]),
        ])
        models = {
            "reference": [0.7, 0.6, 0.8, 0.3, 0.4, 0.9, 0.1, 0.3, 0.8, 0.7, 0.6, 0.2, 0.7],
            "candidate": [0.2, 0.3, 0.8, 0.5, 0.9, 0.2, 0.7, 0.4, 0.9, 0.1, 0.8, 0.3, 0.6],
        }
        result = grouped_bootstrap(rows, models, reference="reference", resamples=32, seed=17)
        order = result["group_order"]
        self.assertEqual(order, ["g0", "g1", "g2", "g3", "g4", "g5"])
        protocol = result["summary"]["protocol"]
        self.assertEqual(protocol["family_group_counts"], {"bowl": 2, "glass": 3, "pen": 1})
        self.assertEqual(protocol["groups"], 6)
        self.assertEqual(protocol["rows"], 13)
        self.assertEqual(protocol["bit_generator"], "PCG64")
        self.assertEqual(protocol["numpy_version"], np.__version__)
        group_indices = {group: [index for index, row in enumerate(rows) if row["group_id"] == group]
                         for group in order}
        group_family = {row["group_id"]: row["target"] for row in rows}
        draws = {model: {metric: [] for metric in METRICS} for model in models}
        contrasts = {metric: [] for metric in METRICS}
        rng = np.random.default_rng(17)
        sizes = set()
        for weights in result["group_weights"]:
            self.assertTrue(all(type(weight) is int and weight >= 0 for weight in weights))
            self.assertEqual(sum(weights), 6)
            expected_weights = np.zeros(6, dtype=int)
            for family, expected in (("bowl", 2), ("glass", 3), ("pen", 1)):
                indices = [index for index, group in enumerate(order) if group_family[group] == family]
                self.assertEqual(sum(weights[index] for index in indices), expected)
                sampled = rng.choice(indices, size=expected, replace=True)
                expected_weights += np.bincount(sampled, minlength=6)
            self.assertEqual(weights, expected_weights.tolist())
            selected = [index for group, count in zip(order, weights)
                        for _ in range(count) for index in group_indices[group]]
            sizes.add(len(selected))
            sampled_rows = [rows[index] for index in selected]
            scores = {model: validation_metrics(sampled_rows, [probabilities[index] for index in selected])
                      for model, probabilities in models.items()}
            for metric in METRICS:
                for model in models:
                    draws[model][metric].append(scores[model][metric])
                contrasts[metric].append(scores["candidate"][metric] - scores["reference"][metric])
        self.assertGreater(len(sizes), 1)  # Row count may vary; group counts must not.
        for metric in METRICS:
            for model in models:
                lower, upper = np.quantile(draws[model][metric], [0.025, 0.975], method="linear")
                point = validation_metrics(rows, models[model])[metric]
                self.assert_interval(result["summary"]["models"][model][metric], point, lower, upper, 32, 0)
            lower, upper = np.quantile(contrasts[metric], [0.025, 0.975], method="linear")
            point = validation_metrics(rows, models["candidate"])[metric] - validation_metrics(rows, models["reference"])[metric]
            self.assert_interval(result["summary"]["differences"]["candidate"][metric], point, lower, upper, 32, 0)
        reversed_models = dict(reversed(list(models.items())))
        self.assertEqual(result, grouped_bootstrap(rows, reversed_models, reference="reference", resamples=32, seed=17))
        identical = grouped_bootstrap(rows, {"reference": models["reference"], "same": models["reference"]},
                                      reference="reference", resamples=8, seed=17)
        for metric in METRICS:
            self.assert_interval(identical["summary"]["differences"]["same"][metric], 0.0, 0.0, 0.0, 8, 0)

    def test_conditional_intervals_count_single_class_replicates(self):
        rows = rows_for([("failures", "bowl", [0, 0]), ("successes", "bowl", [1])])
        result = grouped_bootstrap(rows, {"reference": [0.5] * 3, "candidate": [0.25] * 3},
                                   reference="reference", resamples=64, seed=9)
        weights = result["group_weights"]
        both = sum(failures > 0 and successes > 0 for failures, successes in weights)
        with_failures = sum(failures > 0 for failures, successes in weights)
        self.assertGreater(both, 0)
        self.assertLess(both, 64)
        self.assertLess(with_failures, 64)
        for metric, valid, reason in (("roc_auc", both, "requires both classes"),
                                      ("balanced_accuracy", both, "requires both classes"),
                                      ("failure_recall", with_failures, "requires failures")):
            for model in ("reference", "candidate"):
                interval = result["summary"]["models"][model][metric]
                self.assertEqual(interval["valid_resamples"], valid)
                self.assertEqual(interval["undefined_resamples"], 64 - valid)
                self.assertEqual(interval["undefined_reasons"], {reason: 64 - valid})
                self.assertEqual(interval["interval_condition"], "conditional_on_metric_defined")
                self.assertIsNone(interval["estimate_undefined_reason"])
            contrast = result["summary"]["differences"]["candidate"][metric]
            expected = 1.0 if metric == "failure_recall" else 0.0
            self.assert_interval(contrast, expected, expected, expected, valid, 64 - valid)
            self.assertEqual(contrast["undefined_reasons"],
                             {f"candidate: {reason}; reference: {reason}": 64 - valid})
        for metric in ("log_loss", "brier"):
            self.assertEqual(result["summary"]["models"]["candidate"][metric]["valid_resamples"], 64)
        json.dumps(result, allow_nan=False)

    def test_all_undefined_metrics_stay_null_and_failure_only_support_is_defined(self):
        for label in (0, 1):
            with self.subTest(label=label):
                rows = rows_for([("a", "bowl", [label]), ("b", "glass", [label, label])])
                result = grouped_bootstrap(rows, {"reference": [0.5] * 3, "candidate": [0.25] * 3},
                                           reference="reference", resamples=4, seed=3)
                undefined = ["roc_auc", "balanced_accuracy"] + (["failure_recall"] if label == 1 else [])
                for metric in undefined:
                    for interval in (result["summary"]["models"]["candidate"][metric],
                                     result["summary"]["differences"]["candidate"][metric]):
                        for key in ("estimate", "lower", "upper"):
                            self.assertIsNone(interval[key])
                        self.assertEqual(interval["valid_resamples"], 0)
                        self.assertEqual(interval["undefined_resamples"], 4)
                        self.assertIsNotNone(interval["estimate_undefined_reason"])
                        self.assertEqual(sum(interval["undefined_reasons"].values()), 4)
                        self.assertEqual(interval["interval_condition"], "undefined_no_valid_resamples")
                if label == 0:
                    self.assert_interval(result["summary"]["models"]["candidate"]["failure_recall"], 1, 1, 1, 4, 0)
                json.dumps(result, allow_nan=False)

    def test_exact_probability_endpoints_are_valid_without_clipping_inputs(self):
        rows = rows_for([("a", "bowl", [0, 1])])
        result = grouped_bootstrap(rows, {"reference": [0.0, 1.0]}, reference="reference", resamples=2, seed=3)
        metrics = result["summary"]["models"]["reference"]
        self.assert_interval(metrics["brier"], 0, 0, 0, 2, 0)
        self.assert_interval(metrics["roc_auc"], 1, 1, 1, 2, 0)
        self.assertTrue(math.isfinite(metrics["log_loss"]["estimate"]))
        self.assertGreaterEqual(metrics["log_loss"]["estimate"], 0)
        self.assertEqual(result["summary"]["differences"], {})

    def test_invalid_probabilities_and_model_identifiers_are_rejected(self):
        rows = rows_for([("a", "bowl", [0, 1])])
        for probabilities in ([], [0.5], [0.1, 0.5, 0.9], [[0.1], [0.9]],
                              [float("nan"), 0.5], [0.5, float("inf")],
                              [float("-inf"), 0.5], [-0.01, 0.5], [0.5, 1.01],
                              ["invalid", 0.5], None, 0.5):
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                grouped_bootstrap(rows, {"reference": probabilities}, reference="reference", resamples=2, seed=3)
        for models, reference in (({}, "reference"), ([], "reference"), (None, "reference"),
                                  ({"": [0.1, 0.9]}, ""), ({" ": [0.1, 0.9]}, " "),
                                  ({1: [0.1, 0.9]}, 1), ({"candidate": [0.1, 0.9]}, "reference")):
            with self.subTest(models=models), self.assertRaises(ValueError):
                grouped_bootstrap(rows, models, reference=reference, resamples=2, seed=3)

    def test_invalid_groups_families_partitions_and_labels_are_rejected(self):
        rows = rows_for([("a", "bowl", [0, 1])])
        invalid_rows = [[], None, [None]]
        for field, value in (("group_id", ""), ("group_id", " "), ("group_id", None),
                             ("group_id", 12), ("target", ""), ("target", None),
                             ("target", 12), ("split", "train"), ("split", "test"),
                             ("split", "final"), ("success", "2"), ("success", 0.5),
                             ("success", True), ("success", None)):
            changed = copy.deepcopy(rows)
            changed[0][field] = value
            invalid_rows.append(changed)
        crossed = copy.deepcopy(rows)
        crossed[1]["target"] = "glass"
        invalid_rows.append(crossed)
        for field in ("group_id", "target", "success", "split"):
            changed = copy.deepcopy(rows)
            del changed[0][field]
            invalid_rows.append(changed)
        for bad_rows in invalid_rows:
            with self.subTest(rows=bad_rows), self.assertRaises(ValueError):
                grouped_bootstrap(bad_rows, {"reference": [0.1, 0.9]}, reference="reference", resamples=2, seed=3)

    def test_seed_and_resample_count_require_positive_nonboolean_integers(self):
        rows = rows_for([("a", "bowl", [0, 1])])
        for name in ("seed", "resamples"):
            for value in (0, -1, True, False, 1.5, "3", None):
                with self.subTest(parameter=name, value=value), self.assertRaises(ValueError):
                    grouped_bootstrap(rows, {"reference": [0.1, 0.9]}, reference="reference", **{name: value})


if __name__ == "__main__":
    unittest.main()
