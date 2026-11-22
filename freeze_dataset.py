"""Export the exact published v1 dataset and folds from an upstream checkout.

This copies original bytes; it does not collect data, relabel trials, or resplit.
Run with --source-dir pointing to results/grasp_classifier_five_objects_20261121
in a checkout containing the pinned upstream revision.
"""

import argparse
import csv
import hashlib
import io
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE_REPOSITORY = "https://github.com/youseihuayu-wonderful/Intelligent-Grasping-Predicting-Success-and-Selecting-Approach-Speeds-of-Robotic-Arms"
SOURCE_COMMIT = "7a2c5642113d0d3b9c5b5d5205feda843b2c043a"
SOURCE_PATH = "results/grasp_classifier_five_objects_20261121"
DATASET_SHA256 = "d218aca702d21fcd0bfe55d60b8022939e732a33c5cb81e42cf07f319716064a"
SOURCE_FILES = {
    "dataset.csv": ("dataset.csv", "af54a6320ff8feca5169e29ea56c3cc5fb38334e"),
    "cv_manifest.json": ("cv_manifest.json", "139b898202733bddd56e8067b32b9cb489ed2478"),
    "environment.json": ("source_environment.json", "2078253a51d0017eec10d90079ae6d77818df3ea"),
    "collection/summary.json": ("source_collection_summary.json", "505246323e38c9ee06349543bf2a84fb62d8a269"),
}
FEATURES = (
    "target", "scale", "mass_kg", "sliding_friction", "x_m", "y_m",
    "yaw_rad", "opening_m", "closing_force_n", "close_trigger_m", "descent_speed_m_s",
)
PARTITIONS = ("train", "validation", "test")
TARGETS = {"bowl", "bottle", "spoon", "glass", "pen"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def frozen_payloads(source):
    payloads, provenance = {}, []
    for relative, (destination, expected_blob) in SOURCE_FILES.items():
        data = (source / relative).read_bytes()
        blob = hashlib.sha1(
            b"blob " + str(len(data)).encode("ascii") + b"\0" + data,
            usedforsecurity=False,
        ).hexdigest()
        require(blob == expected_blob, f"Source does not match the pinned upstream Git blob: {relative}")
        payloads[destination] = data
        provenance.append({
            "upstream_path": f"{SOURCE_PATH}/{relative}",
            "upstream_git_blob_sha1": blob,
            "sha256": sha256(data),
            "bytes": len(data),
            "local_file": destination,
        })
    require(sha256(payloads["dataset.csv"]) == DATASET_SHA256, "Dataset SHA256 differs from v1")
    return payloads, provenance


def audit_rows(data):
    rows = list(csv.DictReader(io.StringIO(data.decode("utf-8"), newline="")))
    require(len(rows) == 3840, "The pinned v1 dataset must contain 3840 rows")
    group_splits, trial_ids = {}, set()
    for index, row in enumerate(rows):
        require(row["row_index"] == str(index), "row_index must match the original zero-based CSV order")
        require(row["trial_id"] and row["trial_id"] not in trial_ids, "trial_id must be nonempty and unique")
        trial_ids.add(row["trial_id"])
        require(row["split"] in PARTITIONS and row["group_id"], "Invalid split or missing group")
        require(row["success"] in ("0", "1"), "success must be 0 or 1")
        require(row["target"] in TARGETS, "Unknown object family")
        for name in FEATURES[1:]:
            require(math.isfinite(float(row[name])), f"Nonfinite predictor {name}")
        split = group_splits.setdefault(row["group_id"], row["split"])
        require(split == row["split"], "A configuration group crosses data partitions")
    summary = {}
    for split in PARTITIONS:
        part = [row for row in rows if row["split"] == split]
        successes = sum(int(row["success"]) for row in part)
        summary[split] = {
            "rows": len(part), "groups": len({row["group_id"] for row in part}),
            "successes": successes, "failures": len(part) - successes,
            "targets": dict(sorted(Counter(row["target"] for row in part).items())),
        }
    return rows, summary


def audit_cv(data, rows):
    folds = json.loads(data)
    require(isinstance(folds, list) and len(folds) == 3, "Expected the original three CV folds")
    training = {index for index, row in enumerate(rows) if row["split"] == "train"}
    coverage = Counter()
    for fold in folds:
        left, right = fold["train_rows"], fold["validation_rows"]
        require(all(type(index) is int for index in left + right), "CV row IDs must be integers")
        train, validation = set(left), set(right)
        require(len(train) == len(left) and len(validation) == len(right), "Duplicate CV row IDs")
        require(train and validation and not train & validation, "Empty or overlapping CV partitions")
        require(train | validation == training, "CV must partition training rows only")
        train_groups = {rows[index]["group_id"] for index in train}
        validation_groups = {rows[index]["group_id"] for index in validation}
        require(not train_groups & validation_groups, "CV leaks configuration groups")
        require(train_groups == set(fold["train_groups"]), "Recorded CV training groups differ")
        require(validation_groups == set(fold["validation_groups"]), "Recorded CV validation groups differ")
        require(len({rows[index]["success"] for index in train}) == 2, "CV training fold lacks a class")
        require(len({rows[index]["success"] for index in validation}) == 2, "CV validation fold lacks a class")
        coverage.update(validation)
    require(set(coverage) == training and set(coverage.values()) == {1}, "Each training row must be held out once in CV")


def freeze(source, output):
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite a frozen dataset: {output}")
    payloads, provenance = frozen_payloads(source)
    rows, splits = audit_rows(payloads["dataset.csv"])
    audit_cv(payloads["cv_manifest.json"], rows)
    assignments = io.StringIO(newline="")
    writer = csv.DictWriter(assignments, fieldnames=["row_index", "trial_id", "group_id", "split"], lineterminator="\n")
    writer.writeheader()
    writer.writerows({name: row[name] for name in writer.fieldnames} for row in rows)
    payloads["split_assignments.csv"] = assignments.getvalue().encode("utf-8")
    ranges = {}
    for name in FEATURES[1:]:
        values = sorted({float(row[name]) for row in rows})
        ranges[name] = {"min": values[0], "max": values[-1], "distinct_values": len(values)}
        if len(values) <= 16:
            ranges[name]["levels"] = values
    manifest = {
        "schema_version": 1, "dataset_version": "v1",
        "dataset_file": "dataset.csv", "dataset_sha256": DATASET_SHA256,
        "features": list(FEATURES), "categorical_features": ["target"],
        "numeric_features": list(FEATURES[1:]), "label": "success",
        "group_column": "group_id", "split_column": "split",
        "split_assignments_file": "split_assignments.csv",
        "split_assignments_sha256": sha256(payloads["split_assignments.csv"]),
        "cv_manifest_file": "cv_manifest.json",
        "cv_manifest_sha256": sha256(payloads["cv_manifest.json"]),
        "cv_seed": 5644, "cv_folds": 3, "cv_row_index_convention": "Global zero-based CSV row_index, not train-local offsets",
        "decision_threshold": 0.5, "positive_class": 1, "failure_class": 0,
        "rows": len(rows), "groups": len({row["group_id"] for row in rows}),
        "targets": sorted(TARGETS), "splits": splits, "feature_ranges": ranges,
        "source": {
            "repository": SOURCE_REPOSITORY, "commit": SOURCE_COMMIT,
            "dataset_path": f"{SOURCE_PATH}/dataset.csv", "original_files": provenance,
            "raw_trials_path": f"{SOURCE_PATH}/collection/trials.json",
            "label_implementation_path": "grasp_study_physics.py",
        },
        "files": {name: {"sha256": sha256(data), "bytes": len(data)} for name, data in payloads.items()},
        "label_rule": "Controller succeeded and an independent observer continuously verifies bilateral_grip, clear_of_table, target_lifted and relative_pose_stable for at least 2 simulation seconds in hold/done.",
        "test_policy": "Historical upstream test results have already been published and inspected. Preserve this split for reference; no test fitting, scoring or prediction in the shared baseline. This is not a fresh blind final test.",
        "limits": [
            "No new simulation trials, label changes, resplitting or model-result copying occur in this export.",
            "CSV evidence_source paths are preserved original-machine provenance, not portable paths. Use the pinned upstream raw-trial links instead.",
            "Configured speed, force and friction are not measured constant TCP velocity, measured pinch force or measured effective contact friction.",
            "Grasp-and-hold labels do not measure drawer-storage mission success or establish sim-to-real performance.",
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, data in payloads.items():
        with (output / name).open("xb") as stream:
            stream.write(data)
    with (output / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"dataset": str(output / "dataset.csv"), "sha256": DATASET_SHA256,
                      "rows": len(rows), "splits": splits}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "data/v1")
    args = parser.parse_args()
    freeze(args.source_dir.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
