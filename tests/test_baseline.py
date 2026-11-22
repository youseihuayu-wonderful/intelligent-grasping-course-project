"""Small deterministic boundary checks; no model training."""
import copy
import tempfile
import unittest
from pathlib import Path

from baseline import (FEATURES, feature_dict, load_dataset, remap_cv,
                      validate_partitions, validation_metrics)


def example_rows():
    rows = []
    for group in range(3):
        for label in (0, 1):
            rows.append({**{name: "1.0" for name in FEATURES[1:]},
                         "target": "bowl", "row_index": str(len(rows)),
                         "trial_id": f"trial-{group}-{label}", "group_id": f"g{group}",
                         "split": "train", "success": str(label)})
    return rows


def example_folds(rows):
    folds = []
    for group in range(3):
        validation = [i for i, row in enumerate(rows) if row["group_id"] == f"g{group}"]
        train = [i for i in range(len(rows)) if i not in validation]
        folds.append({"train_rows": train, "validation_rows": validation,
                      "train_groups": sorted({rows[i]["group_id"] for i in train}),
                      "validation_groups": [f"g{group}"]})
    return folds


class BaselineBoundaryTests(unittest.TestCase):
    def test_protected_columns_never_become_features(self):
        row = example_rows()[0]
        row.update(execution_sim_s="999", evidence_source="outcome", secret_label="1")
        before = feature_dict(row)
        row.update(success="1", trial_id="changed", execution_sim_s="0")
        self.assertEqual(before, feature_dict(row))
        self.assertEqual(set(before), set(FEATURES))

    def test_partition_group_leakage_is_rejected(self):
        rows = example_rows()
        rows[0]["split"] = "validation"
        with self.assertRaises(ValueError):
            validate_partitions(rows)

    def test_global_cv_ids_are_remapped_not_used_as_offsets(self):
        rows = example_rows()
        held_out = copy.deepcopy(rows[0])
        held_out.update(split="test", group_id="held-out", trial_id="held-out")
        rows.insert(0, held_out)
        for index, row in enumerate(rows):
            row["row_index"] = str(index)
        folds = example_folds(rows[1:])
        for fold in folds:
            for key in ("train_rows", "validation_rows"):
                fold[key] = [i + 1 for i in fold[key]]
        mapped = remap_cv(rows, folds)
        self.assertEqual(mapped[0][1], [0, 1])
        corrupted = copy.deepcopy(folds)
        corrupted[0]["train_rows"][0] = 0
        with self.assertRaises(ValueError):
            remap_cv(rows, corrupted)

    def test_cv_rejects_group_overlap_and_repeated_validation(self):
        rows = example_rows()
        folds = example_folds(rows)
        rows[2]["group_id"] = rows[0]["group_id"]
        for fold in folds:
            for side in ("train", "validation"):
                fold[f"{side}_groups"] = sorted({rows[i]["group_id"] for i in fold[f"{side}_rows"]})
        with self.assertRaises(ValueError):
            remap_cv(rows, folds)
        rows = example_rows()
        folds = example_folds(rows)
        folds[2] = copy.deepcopy(folds[1])
        with self.assertRaises(ValueError):
            remap_cv(rows, folds)

    def test_corrupt_frozen_csv_is_rejected(self):
        source = Path(__file__).resolve().parents[1] / "data" / "v1"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("manifest.json", "dataset.csv", "split_assignments.csv", "cv_manifest.json"):
                (root / name).write_bytes((source / name).read_bytes())
            with (root / "dataset.csv").open("ab") as handle:
                handle.write(b"\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_dataset(root)

    def test_failure_recall_and_undefined_metrics(self):
        rows = [{"split": "validation", "success": str(label)} for label in (0, 0, 1, 1)]
        result = validation_metrics(rows, [0.1, 0.8, 0.8, 0.9])
        self.assertEqual(result["failure_recall"], 0.5)
        self.assertEqual(result["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertEqual(result["balanced_accuracy"], 0.75)
        one_class = validation_metrics(rows[2:], [0.8, 0.9])
        self.assertIsNone(one_class["failure_recall"])
        self.assertIsNone(one_class["roc_auc"])
        rows[0]["split"] = "test"
        with self.assertRaises(ValueError):
            validation_metrics(rows, [0.1, 0.8, 0.8, 0.9])


if __name__ == "__main__":
    unittest.main()
