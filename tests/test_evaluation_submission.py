"""Isolated submission boundary tests; no model fitting, prediction, or private data."""
from contextlib import redirect_stderr
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from baseline import FEATURES, validation_metrics
from evaluation_submission import FIELDS, ROOT, create_submission, load_submission, main


def digest(content):
    return hashlib.sha256(content).hexdigest()


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.evidence = self.root / "evidence"
        self.evidence.mkdir()
        self.manifest_path = self.root / "submissions" / "candidate.json"
        self.manifest_path.parent.mkdir()
        self.rows = [dict(row_index=str(index), trial_id=f"trial-{index}",
                          group_id=f"condition-{group}", split="validation", success=str(label))
                     for index, group, label in ((8, 1, 0), (21, 1, 1), (32, 2, 0), (54, 2, 1))]
        self.probabilities = [0.0, 0.5, 0.8, 1.0]
        self.records = [dict(row, p_success=str(probability),
                             predicted_success=str(int(probability >= 0.5)))
                        for row, probability in zip(self.rows, self.probabilities)]
        self.audit = {"sha256": {name: digest(name.encode()) for name in
                                 ("dataset.csv", "split_assignments.csv", "cv_manifest.json", "manifest.json")}}
        self.paths = {name: self.evidence / filename for name, filename in
                      (("predictions", "predictions.csv"), ("run_report", "report.json"),
                       ("training_code", "train.py"), ("model_artifact", "model.joblib"))}
        self.paths["training_code"].write_text("# Declared producer source; never executed.\n", encoding="utf-8")
        # Deliberately not a pickle: only its bytes, not its internals, may be inspected.
        self.paths["model_artifact"].write_bytes(b"opaque artifact: not valid pickle\x00\xff")
        self.report = {
            "protocol": {"fit_partition": "train", "score_partition": "validation",
                         "features": list(FEATURES), "decision_threshold": 0.5,
                         "selection": "TRAIN-only frozen-fold negative log loss"},
            "audit": copy.deepcopy(self.audit),
            "code_sha256": digest(self.paths["training_code"].read_bytes()),
            "versions": {"python": "3.test", "estimator": "1.test"},
            "selection": {"C": 0.1},
            "cv_results": [{"C": 0.1, "mean_cv_neg_log_loss": -0.7}],
        }
        self.write_records(self.records)
        self.write_report(self.report)
        self.refresh_manifest()

    def write_records(self, records, fields=FIELDS):
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
        self.paths["predictions"].write_text(stream.getvalue(), encoding="utf-8")

    def write_report(self, report):
        self.paths["run_report"].write_text(json.dumps(report), encoding="utf-8")

    def refresh_manifest(self):
        manifest = {"schema_version": 1, "model_id": "candidate_1", "probability_class": 1,
                    "files": {name: {"path": "../evidence/" + path.name,
                                      "sha256": digest(path.read_bytes())}
                              for name, path in self.paths.items()}}
        self.write_manifest(manifest)
        return manifest

    def write_manifest(self, manifest):
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def load(self):
        return load_submission(self.manifest_path, self.rows, self.audit)

    def create(self, output, **changes):
        arguments = dict(model_id="candidate_1", output=output, validation_rows=self.rows,
                         package_audit=self.audit, **self.paths)
        arguments.update(changes)
        return create_submission(**arguments)

    def test_shuffled_rows_are_aligned_by_trial_not_file_position(self):
        self.write_records([self.records[i] for i in (3, 1, 0, 2)])
        self.refresh_manifest()
        result = self.load()
        self.assertEqual(result["probabilities"], self.probabilities)
        self.assertEqual(result["model_id"], "candidate_1")
        self.assertEqual(result["source_report"], self.report)
        self.assertFalse(result["provenance"]["training_execution_independently_verified"])
        self.assertFalse(result["provenance"]["model_artifact_deserialized"])
        self.assertEqual(result["provenance"]["manifest_sha256"], digest(self.manifest_path.read_bytes()))

    def test_missing_duplicate_and_nonvalidation_trials_are_rejected(self):
        variants = [self.records[:-1], self.records + [self.records[0]]]
        for trial, split in (("train-trial", "train"), ("test-trial", "test"),
                             ("pilot-trial", "validation"), ("unknown", "validation")):
            records = copy.deepcopy(self.records)
            records[-1].update(trial_id=trial, split=split)
            variants.append(records)
        for records in variants:
            with self.subTest(records=records):
                self.write_records(records)
                self.refresh_manifest()
                with self.assertRaises(ValueError):
                    self.load()

    def test_authoritative_identity_and_label_cannot_be_changed(self):
        for field, value in (("row_index", "0"), ("group_id", "other"),
                             ("split", "test"), ("success", "1"), ("success", "0.0")):
            with self.subTest(field=field, value=value):
                records = copy.deepcopy(self.records)
                records[0][field] = value
                self.write_records(records)
                self.refresh_manifest()
                with self.assertRaises(ValueError):
                    self.load()

    def test_invalid_probabilities_binary_predictions_and_ties(self):
        for value in ("", " ", "nan", "NaN", "inf", "-inf", "1e999", "-0.01", "1.01", "0_0", "text"):
            with self.subTest(probability=value):
                records = copy.deepcopy(self.records)
                records[0]["p_success"] = value
                self.write_records(records)
                self.refresh_manifest()
                with self.assertRaises(ValueError):
                    self.load()
        for index, prediction in ((1, "0"), (0, "1"), (0, "2"), (0, "0.0"), (0, "")):
            with self.subTest(index=index, prediction=prediction):
                records = copy.deepcopy(self.records)
                records[index]["predicted_success"] = prediction
                self.write_records(records)
                self.refresh_manifest()
                with self.assertRaises(ValueError):
                    self.load()
        self.write_records(self.records)
        self.refresh_manifest()
        self.assertEqual(self.load()["probabilities"], [0.0, 0.5, 0.8, 1.0])

    def test_bad_csv_header_cells_encoding_and_quotes_are_rejected(self):
        original = self.paths["predictions"].read_bytes()
        header, body = original.split(b"\n", 1)
        variants = [header.replace(b"trial_id", b"row_index") + b"\n" + body,
                    header + b",extra\n" + body,
                    header.replace(b",predicted_success", b"") + b"\n" + body,
                    original + b"\n", original + b"too,few,cells\n",
                    original + b"too,many,cells,a,b,c,d,e\n", b"\xff" + original,
                    header + b'\n"unterminated']
        for content in variants:
            with self.subTest(content=content):
                self.paths["predictions"].write_bytes(content)
                self.refresh_manifest()
                with self.assertRaises(ValueError):
                    self.load()
        self.paths["predictions"].write_bytes(b"\xef\xbb\xbf" + original)
        self.refresh_manifest()
        self.assertEqual(self.load()["probabilities"], self.probabilities)

    def test_every_bound_file_hash_is_checked(self):
        for name, path in self.paths.items():
            original = path.read_bytes()
            with self.subTest(file=name):
                self.refresh_manifest()
                path.write_bytes(original + b"corruption")
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    self.load()
                path.write_bytes(original)

    def test_frozen_data_split_fold_and_optional_manifest_hash_drift(self):
        for name in self.audit["sha256"]:
            with self.subTest(file=name):
                report = copy.deepcopy(self.report)
                report["audit"]["sha256"][name] = "0" * 64
                self.write_report(report)
                self.refresh_manifest()
                with self.assertRaisesRegex(ValueError, "Frozen package hash mismatch"):
                    self.load()
        report = copy.deepcopy(self.report)
        del report["audit"]["sha256"]["manifest.json"]
        self.write_report(report)
        self.refresh_manifest()
        self.assertEqual(self.load()["probabilities"], self.probabilities)

    def test_protocol_and_feature_order_drift_is_rejected(self):
        for key, value in (("fit_partition", "train+validation"), ("score_partition", "test"),
                           ("decision_threshold", 0.6), ("selection", " "),
                           ("features", list(reversed(FEATURES))),
                           ("features", list(FEATURES) + ["success"])):
            with self.subTest(field=key):
                report = copy.deepcopy(self.report)
                report["protocol"][key] = value
                self.write_report(report)
                self.refresh_manifest()
                with self.assertRaises(ValueError):
                    self.load()

    def test_missing_or_invalid_required_metadata_is_rejected(self):
        variants = []
        for field in ("protocol", "audit", "code_sha256", "versions", "selection"):
            report = copy.deepcopy(self.report)
            del report[field]
            variants.append(report)
        for field, value in (("code_sha256", "0" * 64), ("versions", {}),
                             ("versions", {"python": 3}), ("selection", {})):
            report = copy.deepcopy(self.report)
            report[field] = value
            variants.append(report)
        for name in ("dataset.csv", "split_assignments.csv", "cv_manifest.json"):
            report = copy.deepcopy(self.report)
            del report["audit"]["sha256"][name]
            variants.append(report)
        for field in ("fit_partition", "score_partition", "features", "decision_threshold", "selection"):
            report = copy.deepcopy(self.report)
            del report["protocol"][field]
            variants.append(report)
        for report in variants:
            with self.subTest(report=report):
                self.write_report(report)
                self.refresh_manifest()
                with self.assertRaises(ValueError):
                    self.load()

    def test_optional_reported_metrics_are_recomputed_with_strict_tolerance(self):
        metrics = validation_metrics(self.rows, self.probabilities)
        report = copy.deepcopy(self.report)
        report["validation"] = copy.deepcopy(metrics)
        report["validation"]["brier"] += 5e-13
        self.write_report(report)
        self.refresh_manifest()
        self.assertEqual(self.load()["source_report"]["validation"], report["validation"])
        variants = []
        for key, value in (("brier", metrics["brier"] + 1e-8),
                           ("undefined", {"roc_auc": "wrong"}),
                           ("confusion_matrix", [[1, 0], [0, 3]]),
                           ("confusion_matrix_labels", [1, 0]), ("failure_recall", True)):
            variant = copy.deepcopy(metrics)
            variant[key] = value
            variants.append(variant)
        missing = copy.deepcopy(metrics)
        del missing["log_loss"]
        variants.extend((missing, None))
        for variant in variants:
            with self.subTest(metrics=variant):
                report["validation"] = variant
                self.write_report(report)
                self.refresh_manifest()
                with self.assertRaisesRegex(ValueError, "Reported validation metrics"):
                    self.load()

    def test_single_class_undefined_metrics_must_match(self):
        self.rows = [dict(row, success="1") for row in self.rows]
        self.records = [dict(record, success="1") for record in self.records]
        self.write_records(self.records)
        self.report["validation"] = validation_metrics(self.rows, self.probabilities)
        self.write_report(self.report)
        self.refresh_manifest()
        self.assertIsNone(self.load()["source_report"]["validation"]["roc_auc"])
        self.report["validation"]["roc_auc"] = 0.0
        self.write_report(self.report)
        self.refresh_manifest()
        with self.assertRaisesRegex(ValueError, "Reported validation metrics"):
            self.load()

    def test_model_ids_and_manifest_schema_are_strict(self):
        original = self.refresh_manifest()
        for model_id in ("../escape", "/absolute", "Amodel", "has.dot", "", "a" * 65,
                         "constant_train_prior", "a/b", None):
            with self.subTest(model_id=model_id):
                manifest = copy.deepcopy(original)
                manifest["model_id"] = model_id
                self.write_manifest(manifest)
                with self.assertRaises(ValueError):
                    self.load()
        for key, value in (("schema_version", True), ("schema_version", 2),
                           ("probability_class", True), ("probability_class", 0), ("extra", 1)):
            with self.subTest(key=key, value=value):
                manifest = copy.deepcopy(original)
                manifest[key] = value
                self.write_manifest(manifest)
                with self.assertRaises(ValueError):
                    self.load()

    def test_manifest_paths_and_digest_declarations_are_strict(self):
        original = self.refresh_manifest()
        entries = [{"path": str(self.paths["predictions"]), "sha256": "0" * 64},
                   {"path": "C:\\escape.csv", "sha256": "0" * 64},
                   {"path": "../evidence/predictions.csv", "sha256": "A" * 64},
                   {"path": "../evidence/predictions.csv", "sha256": "0" * 63},
                   {"path": "../evidence/predictions.csv", "sha256": "0" * 64, "extra": 1},
                   {"path": "../evidence/predictions.csv"}]
        for entry in entries:
            with self.subTest(entry=entry):
                manifest = copy.deepcopy(original)
                manifest["files"]["predictions"] = entry
                self.write_manifest(manifest)
                with self.assertRaises(ValueError):
                    self.load()

    def test_corrupt_or_ambiguous_json_is_rejected_even_when_hash_bound(self):
        for content in (b'{"versions":{},"versions":{}}', b'{"value":NaN}',
                        b'{"value":1e999}', b'[]', b'{', b'\xff'):
            with self.subTest(content=content):
                self.paths["run_report"].write_bytes(content)
                self.refresh_manifest()
                with self.assertRaises(ValueError):
                    self.load()
        self.manifest_path.write_bytes(b'{"schema_version":1,"schema_version":1}')
        with self.assertRaises(ValueError):
            self.load()

    def test_authoritative_rows_must_be_nonempty_unique_and_validation_only(self):
        for rows in ([], self.rows + [self.rows[0]], [dict(self.rows[0], split="test")],
                     [dict(self.rows[0], success="2")]):
            with self.subTest(rows=rows):
                with self.assertRaises(ValueError):
                    load_submission(self.manifest_path, rows, self.audit)

    def test_create_is_portable_and_checks_before_writing(self):
        output = self.root / "new-submissions" / "candidate.json"
        manifest = self.create(output)
        self.assertTrue(all(not Path(entry["path"]).is_absolute()
                            for entry in manifest["files"].values()))
        destination = self.root / "relocated"
        destination.mkdir()
        shutil.copytree(self.evidence, destination / "evidence")
        shutil.copytree(output.parent, destination / output.parent.name)
        result = load_submission(destination / output.parent.name / output.name, self.rows, self.audit)
        self.assertEqual(result["probabilities"], self.probabilities)
        invalid_output = self.root / "must-not-exist" / "candidate.json"
        self.paths["predictions"].write_bytes(b"corrupt CSV")
        with self.assertRaises(ValueError):
            self.create(invalid_output)
        self.assertFalse(invalid_output.parent.exists())

    def test_create_refuses_overwrite_symlink_frozen_data_and_reference_runs(self):
        existing = self.root / "existing.json"
        existing.write_bytes(b"keep existing bytes")
        dangling = self.root / "dangling.json"
        dangling.symlink_to(self.root / "missing-target")
        linked = self.root / "linked.json"
        linked.symlink_to(existing)
        alias = self.root / "data-alias"
        alias.symlink_to(ROOT / "data", target_is_directory=True)
        for output in (existing, dangling, linked, ROOT / "data" / "forbidden-submission.json",
                       ROOT / "protocols" / "forbidden-submission.json",
                       ROOT / "results" / "logistic_baseline_v1" / "forbidden-submission.json",
                       alias / "forbidden-submission.json"):
            with self.subTest(output=output):
                with self.assertRaises(ValueError):
                    self.create(output)
        self.assertEqual(existing.read_bytes(), b"keep existing bytes")
        self.assertFalse((self.root / "missing-target").exists())

    def test_cli_existing_output_is_rejected_before_package_access(self):
        output = self.root / "existing.json"
        output.write_bytes(b"untouched")
        arguments = ["create", "--model-id", "candidate_1", "--output", str(output),
                     "--data-dir", str(self.root / "nonexistent-data")]
        for name, path in self.paths.items():
            arguments.extend(["--" + name.replace("_", "-"), str(path)])
        with self.assertRaises(SystemExit) as raised, redirect_stderr(io.StringIO()):
            main(arguments)
        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(output.read_bytes(), b"untouched")


if __name__ == "__main__":
    unittest.main()
