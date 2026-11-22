"""Strict, reproducible multi-model evaluation on frozen v1 VALIDATION only."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import sys
import time
import warnings

import numpy as np

from baseline import FEATURES, load_dataset, validation_metrics
from evaluation_submission import load_submission
from evaluation_uncertainty import METRICS, grouped_bootstrap

ROOT = Path(__file__).resolve().parent
CONSTANT = "constant_train_prior"
FIELDS = ("row_index", "trial_id", "group_id", "split", "success", "p_success", "predicted_success")
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 5644


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def new_output(path):
    path = Path(path)
    require(not path.exists() and not path.is_symlink(), "Evaluation output must be a new path")
    resolved = path.resolve()
    for protected in ("data", "protocols", "submissions", "results/logistic_baseline_v1"):
        require(not resolved.is_relative_to((ROOT/protected).resolve()),
                "Do not write evaluation results inside frozen data/protocols/submissions or the reference run")
    return path


def save_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def save_csv(path, rows, fields):
    with Path(path).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def support(rows):
    labels = Counter(int(row["success"]) for row in rows)
    return {"rows": len(rows), "groups": len({row["group_id"] for row in rows}),
            "failures_0": labels[0], "successes_1": labels[1]}


def calibration_bins(rows, probabilities):
    require(len(rows) == len(probabilities) and bool(rows), "Calibration needs aligned nonempty rows")
    buckets = [[] for _ in range(10)]
    for row, p in zip(rows, probabilities, strict=True):
        require(row["split"] == "validation" and math.isfinite(p) and 0 <= p <= 1
                and row["success"] in ("0", "1", 0, 1),
                "Calibration requires validation labels and valid probabilities")
        buckets[min(int(p*10), 9)].append((float(p), int(row["success"])))
    result = []
    for index, pairs in enumerate(buckets):
        result.append({"bin": index, "lower": index/10, "upper": (index+1)/10,
                       "right_closed": index == 9, "count": len(pairs),
                       "mean_p_success": math.fsum(p for p, _ in pairs)/len(pairs) if pairs else None,
                       "observed_success_fraction": sum(y for _, y in pairs)/len(pairs) if pairs else None})
    return result


def error_cases(rows, probabilities, source):
    require(len(rows) == len(probabilities), "Error cases need aligned probabilities")
    directory = PurePosixPath(source["raw_trials_path"]).parent/"trials"
    cases = []
    epsilon = np.finfo(np.float64).eps
    for row, probability in zip(rows, probabilities, strict=True):
        require(row["split"] == "validation" and row["success"] in ("0", "1", 0, 1)
                and math.isfinite(probability) and 0 <= probability <= 1,
                "Error cases require validation labels and valid probabilities")
        truth, predicted = int(row["success"]), int(probability >= .5)
        if predicted == truth:
            continue
        clipped = min(max(float(probability), epsilon), 1-epsilon)
        contribution = -math.log(clipped if truth else 1-clipped)
        cases.append({**{name: row[name] for name in FIELDS[:5]},
                      "p_success": float(probability), "predicted_success": predicted,
                      **{name: row[name] for name in FEATURES},
                      "error_type": "missed_failure" if truth == 0 else "missed_success",
                      "log_loss_contribution": contribution,
                      "source_trial_url": f"{source['repository']}/blob/{source['commit']}/{directory}/{row['trial_id']}.json"})
    return sorted(cases, key=lambda row: int(row["row_index"]))


def markdown_summary(point, rows, reference):
    lines = ["# Unified validation evaluation", "",
             "Development validation only; no historical-test, pilot or private-final scoring.", "",
             "| Model | Log loss ↓ | Brier ↓ | ROC-AUC ↑ | Balanced accuracy ↑ | Failure recall ↑ |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for model_id, scores in sorted(point.items()):
        values = ["undefined" if scores[name] is None else f"{scores[name]:.6f}" for name in METRICS]
        lines.append("| " + " | ".join([model_id, *values]) + " |")
    lines.extend(["", f"Rows: {len(rows)}; condition groups: {len({row['group_id'] for row in rows})}; paired-difference reference: `{reference}`.",
                  "", "Read `uncertainty.json` for paired, family-stratified whole-group intervals, defined-replicate counts and limitations. `bootstrap_group_weights.csv` preserves every draw.",
                  "", "All submitted models use the same authoritative row identities and scoring rules. Training-feature/CV claims are checked declarations tied to hashed files, not an independent audit of arbitrary training execution. The evaluator never unpickles model artifacts.",
                  "", "Only submitted real models are listed. The constant reference uses training prevalence only. No model ranking here is a fresh final-test or physical-execution claim.", ""])
    return "\n".join(lines)


def evaluate(submission_paths, output, *, data_dir=None, reference=CONSTANT):
    started = time.monotonic()
    output = new_output(output)
    data_root = ROOT/"data/v1" if data_dir is None else Path(data_dir).resolve()
    require(not output.resolve().is_relative_to(data_root), "Output cannot be inside the input data package")
    package = load_dataset(data_root)
    rows = [row for row in package["rows"] if row["split"] == "validation"]
    train = [row for row in package["rows"] if row["split"] == "train"]
    require(bool(submission_paths), "At least one real model submission is required")
    submissions = {}
    for path in submission_paths:
        submission = load_submission(Path(path), rows, package["audit"])
        submission["provenance"]["manifest_path_relative_to_repository"] = os.path.relpath(Path(path).resolve(), ROOT)
        model_id = submission["model_id"]
        require(model_id not in submissions and model_id != CONSTANT, "Duplicate or reserved model ID")
        submissions[model_id] = submission
    prior = sum(int(row["success"]) for row in train)/len(train)
    probabilities = {name: item["probabilities"] for name, item in submissions.items()}
    probabilities[CONSTANT] = [prior]*len(rows)
    require(reference in probabilities, "Requested paired reference was not submitted")
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        point = {name: validation_metrics(rows, p) for name, p in sorted(probabilities.items())}
        bootstrap = grouped_bootstrap(rows, probabilities, reference=reference,
                                      resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED)
        by_family = {}
        for target in sorted({row["target"] for row in rows}):
            indices = [index for index, row in enumerate(rows) if row["target"] == target]
            family_rows = [rows[index] for index in indices]
            by_family[target] = {"support": support(family_rows), "models": {
                name: validation_metrics(family_rows, [p[index] for index in indices])
                for name, p in sorted(probabilities.items())}}
    diagnostics = {name: error_cases(rows, p, package["manifest"]["source"])
                   for name, p in sorted(probabilities.items())}
    calibration = {name: calibration_bins(rows, p) for name, p in sorted(probabilities.items())}
    # All inputs and scores have passed before any valid-result directory exists.
    output.mkdir(parents=True, exist_ok=False)
    (output/"predictions").mkdir()
    (output/"errors").mkdir()
    for name, p in sorted(probabilities.items()):
        canonical = [{**{field: row[field] for field in FIELDS[:5]},
                      "p_success": float(value), "predicted_success": int(value >= .5)}
                     for row, value in zip(rows, p, strict=True)]
        save_csv(output/"predictions"/f"{name}.csv", canonical, FIELDS)
        fields = [*FIELDS, *FEATURES, "error_type", "log_loss_contribution", "source_trial_url"]
        save_csv(output/"errors"/f"{name}.csv", diagnostics[name], fields)
    counts = support(rows)
    save_csv(output/"comparison.csv", [{"model_id": name, **counts,
             **{metric: scores[metric] for metric in METRICS}} for name, scores in sorted(point.items())],
             ["model_id", *counts, *METRICS])
    weights = bootstrap["group_weights"]
    group_order = bootstrap["group_order"]
    save_csv(output/"bootstrap_group_weights.csv", [dict(resample=index, **dict(zip(group_order, draw, strict=True)))
             for index, draw in enumerate(weights)], ["resample", *group_order])
    save_json(output/"uncertainty.json", bootstrap["summary"])
    save_json(output/"calibration_bins.json", calibration)
    save_json(output/"family_metrics.json", by_family)
    save_json(output/"confusion_matrices.json", {name: {"matrix": scores["confusion_matrix"],
              "true_row_labels": [0, 1], "predicted_column_labels": [0, 1]} for name, scores in point.items()})
    save_json(output/"error_summary.json", {name: {"misclassified_rows": len(cases),
              "error_type_counts": dict(Counter(case["error_type"] for case in cases)),
              "highest_log_loss_errors": sorted(cases, key=lambda case: (-case["log_loss_contribution"], int(case["row_index"])))[:10]}
              for name, cases in diagnostics.items()})
    (output/"summary.md").write_text(markdown_summary(point, rows, reference), encoding="utf-8")
    artifacts = {path.relative_to(output).as_posix(): file_sha256(path)
                 for path in sorted(output.rglob("*")) if path.is_file()}
    report = {
        "status": "passed_validation_prediction_and_metric_checks",
        "protocol": {"dataset": "frozen_v1", "score_partition": "validation", "features": list(FEATURES),
                     "probability_class": 1, "decision_threshold": .5, "threshold_tie_class": 1,
                     "log_loss_clipping_epsilon": float(np.finfo(np.float64).eps),
                     "primary_metric": "log_loss", "metric_keys": list(METRICS),
                     "lower_is_better": ["log_loss", "brier"], "reference": reference,
                     "bootstrap_resamples": BOOTSTRAP_RESAMPLES, "bootstrap_seed": BOOTSTRAP_SEED,
                     "historical_test_rows_scored": 0, "pilot_rows_scored": 0,
                     "sealed_final_plan_opened": False, "models_trained": 0,
                     "model_artifacts_deserialized": False},
        "frozen_data_audit": package["audit"], "support": counts,
        "models": {name: {
            "metrics": point[name], "provenance": submissions[name]["provenance"],
            "declared_training_metadata": {
                key: submissions[name]["source_report"][key]
                for key in ("protocol", "versions", "selection")},
            "declared_cv_candidate_count": (len(submissions[name]["source_report"]["cv_results"])
                if isinstance(submissions[name]["source_report"].get("cv_results"), list) else None),
        } for name in sorted(submissions)},
        "constant_train_prior": {"p_success": prior, "fit_support": support(train), "metrics": point[CONSTANT]},
        "artifacts_sha256": artifacts,
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     **{name: importlib.metadata.version(name) for name in ("scikit-learn", "scipy", "joblib")}},
        "implementation_sha256": {name: file_sha256(ROOT/name) for name in
            ("evaluate_models.py", "evaluation_submission.py", "evaluation_uncertainty.py", "baseline.py", "uv.lock")},
        "command": sys.argv, "wall_s": time.monotonic()-started,
        "limitations": [
            "This is repeatedly usable development validation, not an untouched final assessment or proof of real-robot success.",
            "Only submitted models have results. No LASSO/Random Forest/XGBoost run is fabricated or inferred from another model.",
            "Artifact hashes and feature/training declarations are checked, but arbitrary training execution, fold-local transformations and prediction origin are not independently proven from CSV/report files.",
            "No artifact is unpickled. Review producer code and training evidence before claiming fully controlled or leakage-free model development.",
            "Tuning budgets and hardware may differ; provenance records actual supplied information, not a certification of equal training effort.",
            "Group-bootstrap intervals are approximate descriptive uncertainty, not row-IID or multiplicity-adjusted significance guarantees.",
            "Failure recall alone can favour a constant all-failure classifier; interpret it with the other metrics and supports.",
            "Calibration bins are descriptive diagnostics, not a fitted calibrator or proof of calibrated probabilities.",
        ],
    }
    save_json(output/"report.json", report)
    print(json.dumps({"status": report["status"], "support": counts,
                      "models": {name: {metric: value[metric] for metric in METRICS} for name, value in point.items()},
                      "bootstrap_resamples": BOOTSTRAP_RESAMPLES, "output": str(output)}, indent=2, allow_nan=False))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, default=ROOT/"data/v1")
    parser.add_argument("--reference", default=CONSTANT)
    args = parser.parse_args()
    evaluate(args.submission, args.output, data_dir=args.data_dir, reference=args.reference)


if __name__ == "__main__":
    main()
