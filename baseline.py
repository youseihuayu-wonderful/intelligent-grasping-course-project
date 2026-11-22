"""Frozen grasp-and-hold baseline. Only TRAIN is fitted; only VALIDATION is scored."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import platform
from collections import Counter
import warnings
from pathlib import Path

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, log_loss, roc_auc_score)
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ("target", "scale", "mass_kg", "sliding_friction", "x_m", "y_m",
            "yaw_rad", "opening_m", "closing_force_n", "close_trigger_m",
            "descent_speed_m_s")
DATASET_SHA256 = "d218aca702d21fcd0bfe55d60b8022939e732a33c5cb81e42cf07f319716064a"
FROZEN_FILE_HASHES = {
    "dataset.csv": DATASET_SHA256,
    "cv_manifest.json": "7952deb8129dce436b3c1a26bc40a4d0f43186ff41b4b63bdbd96a11e18323a6",
    "split_assignments.csv": "5a45cd13537248b9d4228a00113bee962815fe28a34d4324ededf11afb561c93",
}
SEED = 5644
EXPECTED_COUNTS = {"train": (2560, 160, 1109), "validation": (640, 40, 243),
                   "test": (640, 40, 217)}


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def feature_dict(row: dict) -> dict:
    """Project onto the allowlist, never outcome, identity or execution fields."""
    result = {"target": row["target"]}
    if result["target"] not in {"bowl", "bottle", "spoon", "glass", "pen"}:
        raise ValueError("Unknown target")
    for name in FEATURES[1:]:
        value = float(row[name])
        if not math.isfinite(value):
            raise ValueError(f"Nonfinite feature: {name}")
        result[name] = value
    return result


def validate_partitions(rows: list[dict]) -> dict:
    """Check identities, labels and group-disjoint partitions; return counts."""
    trials, groups = set(), {}
    for index, row in enumerate(rows):
        if str(row["row_index"]) != str(index):
            raise ValueError("row_index must equal global CSV position")
        if not row["trial_id"] or row["trial_id"] in trials:
            raise ValueError("Missing or duplicate trial identity")
        trials.add(row["trial_id"])
        split, group = row["split"], row["group_id"]
        if split not in EXPECTED_COUNTS or not group:
            raise ValueError("Invalid partition/group")
        if groups.setdefault(group, split) != split:
            raise ValueError("Group crosses partitions")
        if str(row["success"]) not in {"0", "1"}:
            raise ValueError("Invalid binary label")
        feature_dict(row)
    counts = {}
    for split in EXPECTED_COUNTS:
        subset = [row for row in rows if row["split"] == split]
        successes = sum(int(row["success"]) for row in subset)
        counts[split] = {"rows": len(subset),
                         "groups": len({row["group_id"] for row in subset}),
                         "successes": successes, "failures": len(subset) - successes}
    return counts


def remap_cv(rows: list[dict], folds: list[dict]) -> list[tuple[list[int], list[int]]]:
    """Validate GLOBAL row IDs and map the frozen CV to TRAIN-local indices."""
    train = [i for i, row in enumerate(rows) if row["split"] == "train"]
    local = {global_id: i for i, global_id in enumerate(train)}
    universe, seen, mapped = set(train), Counter(), []
    if len(folds) != 3:
        raise ValueError("Expected exactly three frozen CV folds")
    for fold in folds:
        sides = []
        for side in ("train", "validation"):
            ids = fold[f"{side}_rows"]
            if not ids or any(type(i) is not int for i in ids):
                raise ValueError("CV row IDs must be nonempty integer lists")
            if len(set(ids)) != len(ids) or not set(ids) <= universe:
                raise ValueError("Duplicate or non-TRAIN CV rows")
            group_ids = {rows[i]["group_id"] for i in ids}
            declared = fold[f"{side}_groups"]
            if len(declared) != len(set(declared)) or group_ids != set(declared):
                raise ValueError("CV group manifest mismatch")
            if {int(rows[i]["success"]) for i in ids} != {0, 1}:
                raise ValueError("CV partition must contain both classes")
            sides.append((set(ids), group_ids, [local[i] for i in ids]))
        a, b = sides
        if a[0] & b[0] or a[0] | b[0] != universe or a[1] & b[1]:
            raise ValueError("CV rows/groups overlap or coverage is incomplete")
        seen.update(b[0])
        mapped.append((a[2], b[2]))
    if set(seen) != universe or any(count != 1 for count in seen.values()):
        raise ValueError("Each TRAIN row must be validated exactly once")
    return mapped


def load_dataset(data_dir: Path | str) -> dict:
    """Validate the frozen package and return rows, TRAIN-local CV and audit."""
    root = Path(data_dir)
    manifest_bytes = (root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    expected_schema = {"dataset_version": "v1", "dataset_file": "dataset.csv",
                       "features": list(FEATURES), "categorical_features": ["target"],
                       "numeric_features": list(FEATURES[1:]), "label": "success",
                       "group_column": "group_id", "split_column": "split",
                       "split_assignments_file": "split_assignments.csv",
                       "cv_manifest_file": "cv_manifest.json", "cv_seed": SEED,
                       "cv_folds": 3, "decision_threshold": 0.5}
    for key, value in expected_schema.items():
        if manifest.get(key) != value:
            raise ValueError(f"Frozen manifest contract mismatch: {key}")
    if manifest.get("dataset_sha256") != DATASET_SHA256:
        raise ValueError("Frozen dataset hash mismatch")
    contents, hashes = {}, {"manifest.json": sha256(manifest_bytes)}
    for stem, filename in (("dataset", "dataset.csv"),
                           ("split_assignments", "split_assignments.csv"),
                           ("cv_manifest", "cv_manifest.json")):
        content = (root / filename).read_bytes()
        hashes[filename] = sha256(content)
        if (hashes[filename] != manifest.get(f"{stem}_sha256")
                or hashes[filename] != FROZEN_FILE_HASHES[filename]):
            raise ValueError(f"Frozen file hash mismatch: {filename}")
        contents[stem] = content.decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(contents["dataset"])))
    counts = validate_partitions(rows)
    for split, (n, groups, successes) in EXPECTED_COUNTS.items():
        expected = dict(rows=n, groups=groups, successes=successes, failures=n-successes)
        if counts[split] != expected or any(
            manifest["splits"][split].get(key) != value for key, value in expected.items()
        ):
            raise ValueError(f"Frozen partition counts mismatch: {split}")
    assignments = list(csv.DictReader(io.StringIO(contents["split_assignments"])))
    projected = [{key: row[key] for key in ("row_index", "trial_id", "group_id", "split")}
                 for row in rows]
    if assignments != projected:
        raise ValueError("Frozen split assignments differ from dataset")
    cv = remap_cv(rows, json.loads(contents["cv_manifest"]))
    return {"rows": rows, "cv": cv, "manifest": manifest,
            "audit": {"counts": counts, "sha256": hashes, "cv_folds": len(cv),
                      "cv_index_space": "global CSV IDs remapped to TRAIN-local indices",
                      "historical_test_already_inspected": True}}


def validation_metrics(rows: list[dict], probabilities) -> dict:
    """Evaluate VALIDATION only. Failure recall is recall of label zero."""
    if not rows or any(row["split"] != "validation" for row in rows):
        raise ValueError("Only nonempty VALIDATION rows may be scored")
    y = np.asarray([int(row["success"]) for row in rows])
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Invalid binary labels")
    p = np.asarray(probabilities, dtype=float)
    if p.shape != y.shape or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid success probabilities")
    predicted = (p >= 0.5).astype(int)
    matrix = confusion_matrix(y, predicted, labels=[0, 1])
    both = len(np.unique(y)) == 2
    failures = int((y == 0).sum())
    return {"roc_auc": float(roc_auc_score(y, p)) if both else None,
            "balanced_accuracy": float(balanced_accuracy_score(y, predicted)) if both else None,
            "failure_recall": float(matrix[0, 0] / failures) if failures else None,
            "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "confusion_matrix": matrix.tolist(), "confusion_matrix_labels": [0, 1],
            "undefined": {**({"roc_auc": "requires both classes",
                              "balanced_accuracy": "requires both classes"} if not both else {}),
                          **({"failure_recall": "requires failures"} if not failures else {})}}

def run_baseline(package: dict, output: Path) -> None:
    """Select C by TRAIN CV, refit TRAIN, then report VALIDATION only."""
    output.mkdir(parents=True, exist_ok=False)
    rows = package["rows"]
    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    x_train = [feature_dict(row) for row in train]
    y_train = [int(row["success"]) for row in train]
    pipeline = Pipeline([("vectorizer", DictVectorizer(sparse=False)),
                         ("scaler", StandardScaler()),
                         ("logistic", LogisticRegression(solver="lbfgs", max_iter=5000,
                                                         tol=1e-8, random_state=SEED))])
    search = GridSearchCV(pipeline, {"logistic__C": [0.1, 1.0, 10.0]},
                          scoring="neg_log_loss", cv=package["cv"], n_jobs=1,
                          error_score="raise", refit=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        warnings.simplefilter("error", RuntimeWarning)
        search.fit(x_train, y_train)
        model = search.best_estimator_
        positive = list(model.classes_).index(1)
        probabilities = model.predict_proba([feature_dict(row) for row in validation])[:, positive]
    prior = sum(y_train) / len(y_train)
    cv_results = []
    for i, params in enumerate(search.cv_results_["params"]):
        cv_results.append({"C": params["logistic__C"],
                           "mean_cv_neg_log_loss": float(search.cv_results_["mean_test_score"][i]),
                           "std_cv_neg_log_loss": float(search.cv_results_["std_test_score"][i]),
                           **{f"fold_{k}_neg_log_loss": float(search.cv_results_[f"split{k}_test_score"][i])
                              for k in range(3)}})
    report = {"protocol": {"fit_partition": "train", "score_partition": "validation",
                            "selection": "maximum TRAIN-CV negative log loss; fold-local preprocessing",
                            "features": list(FEATURES), "decision_threshold": 0.5,
                            "seed": SEED, "historical_test_already_inspected": True},
              "audit": package["audit"], "code_sha256": sha256(Path(__file__).read_bytes()),
              "versions": {"python": platform.python_version(), "numpy": np.__version__,
                           "scikit-learn": sklearn.__version__, "scipy": scipy.__version__,
                           "joblib": joblib.__version__},
              "selection": {"C": search.best_params_["logistic__C"],
                            "best_cv_neg_log_loss": float(search.best_score_)},
              "cv_results": cv_results,
              "validation": validation_metrics(validation, probabilities),
              "constant_train_prior": {"p_success": prior,
                                       "validation": validation_metrics(validation, np.full(len(validation), prior))},
              "limitations": ["Historical test results were already published and inspected; no test scoring here.",
                              "Simulation grasp-and-hold labels do not establish storage-task or real-world validity.",
                              "Undefined metrics are JSON null; confusion rows are actual labels and columns predicted labels."]}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (output / "validation_predictions.csv").open("w", newline="") as handle:
        fields = ["row_index", "trial_id", "group_id", "split", "success", "p_success", "predicted_success"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, p in zip(validation, probabilities, strict=True):
            writer.writerow({**{key: row[key] for key in fields[:5]},
                             "p_success": float(p), "predicted_success": int(p >= 0.5)})
    with (output / "cv_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cv_results[0]))
        writer.writeheader()
        writer.writerows(cv_results)
    joblib.dump(model, output / "model.joblib")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data" / "v1")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only and args.output is not None:
        parser.error("--validate-only does not write outputs; omit --output")
    if not args.validate_only and args.output is None:
        parser.error("--output is required unless --validate-only")
    package = load_dataset(args.data_dir)
    if args.validate_only:
        print(json.dumps(package["audit"], indent=2, allow_nan=False))
    else:
        run_baseline(package, args.output)


if __name__ == "__main__":
    main()
