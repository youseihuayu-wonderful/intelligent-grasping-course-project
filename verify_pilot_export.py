"""Cross-check the pilot CSV with the frozen legacy extractor and historical evidence."""
import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

from audit_data_quality import FrozenEvidence, ROOT, STUDY, decode_json, require
from baseline import FEATURES, load_dataset
from design_independent_test import payload
from run_development_pilot import sha_file


def physical_key(row):
    return json.dumps({"settings": row["settings"], "initial_state_fingerprint": row["initial_state_fingerprint"]},
                      sort_keys=True, separators=(",", ":"), allow_nan=False)


def feature_key(row):
    return (row["target"], *(float(row[name]) for name in FEATURES[1:]))


def verify(source_repo, evidence_dir, export_dir):
    evidence, exported = Path(evidence_dir), Path(export_dir)
    require(not (exported/"export_verification.json").exists(), "Export verification output must be new")
    reader = FrozenEvidence(source_repo)
    historical_raw = reader.json(f"{STUDY}/collection/trials.json")["rows"]
    legacy_bytes = reader.read("grasp_classifier_data.py")
    old_path = sys.path[:]
    try:
        sys.path.insert(0, str(Path(source_repo).resolve()))
        specification = importlib.util.spec_from_file_location("pilot_legacy_feature_reference", Path(source_repo)/"grasp_classifier_data.py")
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
    frozen = load_dataset(ROOT/"data/v1")
    new_rows = list(csv.DictReader((exported/"dataset.csv").open()))
    raw_rows = decode_json((evidence/"collection/trials.json").read_bytes())["rows"]
    require(len(new_rows) == len(raw_rows) == 320, "Pilot export is incomplete")
    by_id = {row["trial_id"]: row for row in raw_rows}
    require(len(by_id) == 320 and {row["trial_id"] for row in new_rows} == set(by_id), "Missing/extra/repeated CSV trial IDs")
    old_ids = {row["trial_id"] for row in frozen["rows"]}
    old_groups = {row["group_id"] for row in frozen["rows"]}
    old_physics = {physical_key(row) for row in historical_raw}
    old_features = {feature_key(row) for row in frozen["rows"]}
    seen_physics = set()
    group_counts = Counter()
    max_difference = 0.
    for index, row in enumerate(new_rows):
        raw = by_id[row["trial_id"]]
        require(int(row["row_index"]) == index and row["trial_id"] not in old_ids
                and row["group_id"] not in old_groups, "Pilot/history identity overlap or reordered CSV")
        require(row["role"] == "development_pilot_only" and row["split"] == "train", "Pilot incorrectly presented as a test set")
        reference = module.features_from_row(raw)
        require(set(reference) == set(FEATURES) and reference["target"] == row["target"], "Wrong observable feature schema")
        for name in FEATURES[1:]:
            value = float(row[name])
            require(math.isfinite(value) and math.isclose(value, reference[name], rel_tol=1e-12, abs_tol=1e-12),
                    f"CSV predictor differs from original pre-execution extractor: {name}")
            max_difference = max(max_difference, abs(value-reference[name]))
        require(row["success"] in ("0", "1") and type(raw["success"]) is bool
                and int(row["success"]) == int(raw["success"]), "CSV label differs from original physical outcome")
        signature = physical_key(raw)
        require(signature not in old_physics and signature not in seen_physics, "Repeated full physical evidence")
        require(feature_key(row) not in old_features, "Actual settled observable vector overlaps historical v1")
        seen_physics.add(signature)
        group_counts[row["group_id"]] += 1
    require(len(group_counts) == 20 and set(group_counts.values()) == {16}, "Broken paired condition groups")
    audit_report = decode_json((exported/"report.json").read_bytes())
    require(audit_report["status"] == "passed_pilot_evidence_quality_checks"
            and audit_report["verified_labelled_rows"] == 320 and audit_report["audit_issues"] == 0,
            "Independent evidence audit did not pass")
    record = {"status": "passed_independent_export_and_overlap_checks", "rows_checked": 320,
              "groups": 20, "trials_per_group": 16,
              "reference_extractor_sha256": hashlib.sha256(legacy_bytes).hexdigest(),
              "max_predictor_difference_from_legacy_extractor": max_difference,
              "historical_trial_id_overlap": 0, "historical_group_overlap": 0,
              "historical_full_physical_input_overlap": 0, "pilot_full_physical_duplicates": 0,
              "historical_actual_observable_vector_overlap": 0,
              "dataset_sha256": sha_file(exported/"dataset.csv"),
              "historical_raw_sha256": reader.files[f"{STUDY}/collection/trials.json"]["sha256"],
              "final_test_plan_opened": False, "models_trained_or_scored": 0,
              "implementation_sha256": sha_file(Path(__file__))}
    with (exported/"export_verification.json").open("xb") as stream:
        stream.write(payload(record))
    print(json.dumps(record, indent=2, sort_keys=True, allow_nan=False))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    args = parser.parse_args()
    verify(args.source_repo, args.evidence_dir, args.export_dir)


if __name__ == "__main__":
    main()
