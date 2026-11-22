"""Audit frozen v1 against original recorded trials; never fit, simulate or relabel."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

from audit_coverage import audit_coverage
from audit_trace import audit_trace
from baseline import FEATURES, load_dataset

ROOT = Path(__file__).resolve().parent
SOURCE_COMMIT = "7a2c5642113d0d3b9c5b5d5205feda843b2c043a"
STUDY = "results/grasp_classifier_five_objects_20261121"
SOURCE_REPOSITORY = (
    "https://github.com/youseihuayu-wonderful/"
    "Intelligent-Grasping-Predicting-Success-and-Selecting-Approach-Speeds-of-Robotic-Arms"
)
CORE_SOURCES = ("collect_grasp_classifiers.py", "grasp_study_physics.py",
                "grasp_study_objects.py", "grasp.py", "grasp_classifier_data.py")
REASONS = {
    "抬升过程中失去双侧夹持接触": "lost_bilateral_contact_during_lift",
    "夹爪未建立稳定的双侧接触": "stable_bilateral_contact_not_established",
    "机械臂未在时限内到达目标位姿": "pose_target_timeout",
    "Physical postconditions lost after completion": "postcompletion_physical_predicates_lost",
    **{f"{name}掉落、滑移或未达到要求的离桌高度": "hold_postcondition_failure"
       for name in ("黑碗", "酒瓶", "Spoon", "Glass tumbler", "Pen")},
}
LIMITS = [
    "Recorded-evidence consistency, not a guarantee of simulator correctness or real-robot validity.",
    "The observer checked every 0.002 simulation seconds, but trajectories retain approximately 0.020-second samples plus phase/terminal samples. Unsaved physics steps cannot be independently reconstructed here.",
    "Contact/stability booleans share the controller's predicates. Saved trajectories lack full state and contact-force history; this audit is not independent contact sensing or a physics rerun.",
    "Relative stability means target-to-TCP position displacement, not a full orientation guarantee.",
    "Failure kinds are controller diagnostic categories with collision-episode precedence, not independently proven causal root causes or verified object drops.",
    "Only training outcomes are used for exploratory feature/failure summaries and case selection; historical test labels are read for integrity only. That test was already public, not a fresh blind final test.",
    "Eight configurations and two paired latent-error draws per condition are correlated trials, not sixteen independent object conditions.",
    "Object family also identifies geometry, grasp profile and some contact settings; descriptive associations are not causal feature effects.",
    "Mass/friction are prescribed simulator inputs. Speed, force and opening are commands, not measured constant TCP speed, grip force or final gap.",
    "Scale/settings and recorded compiled diagnostics are checked; collision geometry is not independently regenerated. No new trials, labels, classifier results or storage-task success claims are created.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    require(type(value) in (int, float) and math.isfinite(value), "Expected a finite numeric scalar")
    return float(value)


def decode_json(content):
    def parse_float(text):
        value = float(text)
        require(math.isfinite(value), f"Nonfinite JSON number: {text}")
        return value

    def reject_constant(text):
        raise ValueError(f"Nonfinite JSON constant: {text}")

    return json.loads(content, parse_float=parse_float, parse_constant=reject_constant)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def git_blob_hash(content):
    digest = hashlib.sha1()
    digest.update(f"blob {len(content)}\0".encode())
    digest.update(content)
    return digest.hexdigest()


class FrozenEvidence:
    """Read evidence only when its bytes match the declared immutable Git snapshot."""

    def __init__(self, source_repo):
        self.root = Path(source_repo).resolve(strict=True)
        listing = subprocess.check_output([
            "git", "-C", str(self.root), "ls-tree", "-r", "-z", "--full-tree",
            SOURCE_COMMIT, "--", STUDY, *CORE_SOURCES,
        ])
        self.blobs = {}
        for entry in listing.split(b"\0"):
            if entry:
                info, path = entry.split(b"\t", 1)
                mode, kind, blob = info.decode().split()
                require(kind == "blob" and mode in ("100644", "100755"),
                        f"Unsupported frozen evidence entry: {path!r}")
                self.blobs[path.decode()] = blob
        self.files = {}

    def read(self, relative, *, frozen_source=False):
        require(relative in self.blobs, f"Evidence absent from frozen snapshot: {relative}")
        if frozen_source:
            content = subprocess.check_output([
                "git", "-C", str(self.root), "cat-file", "blob", self.blobs[relative],
            ])
        else:
            path = self.root / relative
            require(path.resolve(strict=True).is_relative_to(self.root),
                    f"Evidence escapes source repository: {relative}")
            content = path.read_bytes()
        require(git_blob_hash(content) == self.blobs[relative],
                f"Evidence bytes differ from frozen snapshot: {relative}")
        self.files[relative] = {
            "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
            "upstream_git_blob_sha1": self.blobs[relative],
            "read_method": "frozen_git_blob" if frozen_source else "snapshot_matched_worktree_bytes",
        }
        return content

    def json(self, relative):
        return decode_json(self.read(relative))


def audit_inputs(csv_row, row, protocol):
    """Independently derive predictors and check recorded compiled input diagnostics."""
    require(row["trial_status"] == "completed", "Unusable initialization/status was exported")
    require(type(row["success"]) is bool, "Raw label must be Boolean")
    require(row["warnings"] == [], "Numerical warning trial was exported")
    require(int(csv_row["success"]) == int(row["success"]), "CSV label differs from raw label")
    settings = row["settings"]
    pose = row["initial_pose"]
    position, quaternion = pose["position_m"], pose["quaternion_wxyz"]
    require(len(position) == 3 and len(quaternion) == 4, "Invalid initial pose shape")
    position = [number(value) for value in position]
    quaternion = [number(value) for value in quaternion]
    norm = math.hypot(*quaternion)
    require(math.isclose(norm, 1, rel_tol=0, abs_tol=1e-5), "Initial quaternion is not unit length")
    w, x, y, z = (value / norm for value in quaternion)
    expected = {name: settings[name] for name in FEATURES}
    expected.update(x_m=position[0], y_m=position[1],
                    yaw_rad=math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z)))
    require(csv_row["target"] == expected["target"], "CSV object family differs from raw input")
    errors = {}
    for name in FEATURES[1:]:
        actual = float(csv_row[name])
        desired = number(expected[name])
        require(math.isfinite(actual) and math.isclose(actual, desired, rel_tol=1e-12, abs_tol=1e-12),
                f"CSV feature differs from pre-execution evidence: {name}")
        errors[name] = abs(actual - desired)
    require(math.isclose(float(csv_row["execution_sim_s"]), number(row["execution_sim_s"]),
                         rel_tol=0, abs_tol=1e-9), "CSV diagnostic duration differs from raw duration")
    variant = row["variant_features"]
    mass = number(variant["actual_mass_kg"])
    require(math.isclose(mass, number(settings["mass_kg"]), rel_tol=1e-12, abs_tol=1e-12),
            "Recorded compiled mass differs from prescribed mass")
    for name in ("actual_principal_inertia_kg_m2", "collision_world_aabb_size_m"):
        require(len(variant[name]) == 3 and all(number(v) > 0 for v in variant[name]),
                f"Invalid recorded compiled diagnostic: {name}")
    for name in ("object_collision_sliding_friction", "finger_collision_sliding_friction"):
        require(bool(variant[name]) and all(
            math.isclose(number(v), settings["sliding_friction"], rel_tol=0, abs_tol=1e-12)
            for v in variant[name]), f"Recorded configured friction differs: {name}")
    require(bool(variant["object_collision_friction"]) and all(
        len(v) == 3 and all(math.isclose(number(a), b, rel_tol=0, abs_tol=1e-12)
                            for a, b in zip(v, (settings["sliding_friction"], .005, .0001)))
        for v in variant["object_collision_friction"]), "Recorded full friction vector differs")
    condim = protocol["contact_model"]["object_collision_condim"][settings["target"]]
    require(bool(variant["object_collision_contact_dimensions"]) and all(
        type(value) is int and value == condim for value in variant["object_collision_contact_dimensions"]),
        "Recorded object contact dimensions differ from protocol")
    require(type(variant["mujoco_enableflags"]) is int and variant["mujoco_enableflags"] & 16,
            "Recorded MULTICCD flag is absent")
    contacts = row["actual_finger_target_sliding_frictions"]
    dimensions = row["actual_finger_target_contact_dimensions"]
    require(all(math.isclose(number(v), settings["sliding_friction"], rel_tol=0, abs_tol=1e-6)
                for v in contacts), "Observed contact friction differs from prescribed setting")
    require(all(type(v) is int and v == condim for v in dimensions),
            "Observed finger-target contact dimensions differ from family configuration")
    if row["success"]:
        require(bool(contacts) and bool(dimensions), "Positive label has no recorded finger-target contact")
    offset = protocol["study_objects"][settings["target"]]["grasp_offset_body_m_at_scale_1"]
    require(len(variant["grasp_offset_body_m"]) == 3 and all(
        math.isclose(number(actual), number(base)*settings["scale"] + settings[f"bias_{axis}_m"],
                     rel_tol=0, abs_tol=1e-12)
        for axis, base, actual in zip("xyz", offset, variant["grasp_offset_body_m"])),
        "Recorded grasp offset differs from scaled profile plus latent execution error")
    episodes = row["collisions"]["episodes"]
    require(type(episodes) is int and episodes >= 0, "Invalid collision episode count")
    if row["success"]:
        kind, reason = None, "success"
        require(row["failure_reason"] is None, "Positive label has a controller failure reason")
    else:
        kind = ("unplanned_contact" if episodes else
                "grasp_contact_or_retention" if row["last_active_phase"] in ("close", "lift", "hold")
                else "motion_or_preparation")
        require(row["failure_reason"] in REASONS, "Unrecognized frozen controller failure diagnostic")
        reason = REASONS[row["failure_reason"]]
    require(row["failure_kind"] == kind, "Recorded failure kind violates source diagnostic rule")
    return {"max_feature_absolute_error": max(errors.values()),
            "failure_reason_code": reason, "failure_kind": kind,
            "recorded_mass_absolute_error_kg": abs(mass - settings["mass_kg"])}


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def run_audit(source_repo, data_dir, output):
    started = time.monotonic()
    package = load_dataset(data_dir)
    manifest, csv_rows = package["manifest"], package["rows"]
    require(manifest["source"]["commit"] == SOURCE_COMMIT, "Unexpected source snapshot")
    reader = FrozenEvidence(source_repo)
    protocol = reader.json(f"{STUDY}/protocol.json")
    raw_rows = reader.json(f"{STUDY}/collection/trials.json")["rows"]
    old_audit = reader.json(f"{STUDY}/data_audit.json")
    old_splits = reader.json(f"{STUDY}/split_manifest.json")
    environment = reader.json(f"{STUDY}/environment.json")
    collection = reader.json(f"{STUDY}/collection/summary.json")
    for item in manifest["source"]["original_files"]:
        content = reader.read(item["upstream_path"])
        require(content == (Path(data_dir) / item["local_file"]).read_bytes(),
                f"Course copy differs from frozen original: {item['local_file']}")
    source_checks = {}
    for name in CORE_SOURCES:
        content = reader.read(name, frozen_source=True)
        expected = environment["input_sha256"].get(name)
        if name == "collect_grasp_classifiers.py":
            expected = environment["classifier_collector_sha256"]
        elif name == "grasp_study_physics.py":
            expected = environment["physical_collector_sha256"]
        actual = hashlib.sha256(content).hexdigest()
        if expected is not None:
            require(actual == expected, f"Frozen implementation differs from collection environment: {name}")
        source_checks[name] = {"sha256": actual, "recorded_collection_sha256": expected,
                              "collection_hash_verified": expected is not None}
    coverage = audit_coverage(csv_rows, raw_rows, protocol)
    require(len(old_audit["row_records"]) == len(raw_rows) == len(csv_rows), "Original audit count differs")
    require(len(old_splits["rows"]) == len(csv_rows), "Original split manifest count differs")
    require(old_audit["excluded_count"] == old_audit["duplicate_count"] == 0,
            "Frozen v1 unexpectedly contains excluded or deduplicated source attempts")
    require(old_audit["input_count"] == old_audit["included_count"] == len(raw_rows),
            "Original audit summary is inconsistent")
    source_reasons = coverage["training_only_diagnostics"].pop("failure_reason_counts")
    normalized_reasons = Counter()
    for reason, count in source_reasons.items():
        require(reason in REASONS, "Unrecognized training failure diagnostic")
        normalized_reasons[REASONS[reason]] += count
    coverage["training_only_diagnostics"]["controller_failure_diagnostic_codes"] = dict(normalized_reasons)
    raw_by_id = {row["trial_id"]: row for row in raw_rows}
    source_index = {row["trial_id"]: index for index, row in enumerate(raw_rows)}
    require(len(raw_by_id) == len(raw_rows), "Duplicate raw trial IDs")
    expected_trial_paths = {f"{STUDY}/collection/trials/{trial_id}.json" for trial_id in raw_by_id}
    expected_trace_paths = {f"{STUDY}/collection/traces/{trial_id}.json.gz" for trial_id in raw_by_id}
    for directory, expected in (("trials", expected_trial_paths), ("traces", expected_trace_paths)):
        prefix = f"{STUDY}/collection/{directory}/"
        require({path for path in reader.blobs if path.startswith(prefix)} == expected,
                f"Frozen {directory} inventory differs from recorded attempts")
    audited, cases = [], []
    training = defaultdict(Counter)
    counts = Counter()
    chosen_cases = set()
    for csv_row in csv_rows:
        trial_id = csv_row["trial_id"]
        require(re.fullmatch(r"[A-Za-z0-9_.-]+", trial_id) is not None, "Unsafe trial ID")
        row = raw_by_id[trial_id]
        index = source_index[trial_id]
        old = old_audit["row_records"][index]
        require(old["trial_id"] == trial_id and old["source_index"] == index
                and old["clean_index"] == int(csv_row["row_index"])
                and old["decision"] == "included" and old["reasons"] == []
                and old["label"] is row["success"], "Original row audit disagrees with frozen export")
        settings = {k: v if k == "target" else number(v) for k, v in row["settings"].items()}
        physical_key = canonical({"settings": settings,
                                  "initial_state": {"fingerprint": row["initial_state_fingerprint"]}})
        require(hashlib.sha256(physical_key.encode()).hexdigest() == old["physical_fingerprint"],
                "Original physical fingerprint cannot be independently reproduced")
        split_record = old_splits["rows"][int(csv_row["row_index"])]
        require(all(str(split_record[key]) == csv_row[key]
                    for key in ("row_index", "trial_id", "group_id", "split"))
                and split_record["success"] is row["success"]
                and split_record["condition_id"] == row["metadata"]["condition_id"]
                and split_record["source"] == csv_row["evidence_source"],
                "Original split manifest disagrees with frozen row")
        trial_path = f"{STUDY}/collection/trials/{trial_id}.json"
        trace_path = f"{STUDY}/collection/traces/{trial_id}.json.gz"
        require(reader.json(trial_path) == row, "Individual trial record differs from aggregate")
        trace = decode_json(gzip.decompress(reader.read(trace_path)))
        trace_check = audit_trace(row, trace)
        input_check = audit_inputs(csv_row, row, protocol)
        result = {key: csv_row[key] for key in ("row_index", "trial_id", "group_id", "split", "target")}
        result.update(success=int(row["success"]), raw_record_path=trial_path,
                      raw_record_sha256=reader.files[trial_path]["sha256"], trace_path=trace_path,
                      trace_sha256=reader.files[trace_path]["sha256"], **input_check,
                      **{key: value for key, value in trace_check.items()
                         if key not in ("trial_id", "limitations")})
        audited.append(result)
        counts["rows"] += 1
        counts["successes" if row["success"] else "failures"] += 1
        counts["trace_samples"] += trace_check["sample_count"]
        if csv_row["split"] == "train":
            target = csv_row["target"]
            training["controller_failure_diagnostics"][input_check["failure_reason_code"]] += 1
            training["failure_kinds"][input_check["failure_kind"] or "success"] += 1
            training["last_active_phase"][row["last_active_phase"]] += 1
            if row["collisions"]["episodes"]:
                training["rows_with_collision_episodes"]["success" if row["success"] else "failure"] += 1
            keys = [("family_label", target, row["success"])]
            if not row["success"]:
                keys.append(("controller_diagnostic", input_check["failure_reason_code"]))
            new_keys = [key for key in keys if key not in chosen_cases]
            if new_keys:
                chosen_cases.update(new_keys)
                cases.append({"selection_reasons": [list(key) for key in new_keys],
                              "evidence": result, "features": {name: csv_row[name] for name in FEATURES},
                              "final_postconditions": row["postconditions"],
                              "diagnostic_is_not_proven_causal_root_cause": True})
        if counts["rows"] % 480 == 0:
            print(f"Audited {counts['rows']}/{len(csv_rows)} rows; {counts['trace_samples']} sampled states", flush=True)
    require(collection["attempted"] == collection["labelled"] == counts["rows"]
            and collection["successes"] == counts["successes"]
            and collection["failures"] == counts["failures"]
            and collection["statuses"] == {"completed": counts["rows"]}
            and collection["numerical_warning_trials"] == 0, "Collection summary disagrees with raw evidence")
    require(old_audit["class_counts"] == {"0": counts["failures"], "1": counts["successes"]},
            "Original audit class counts disagree with raw evidence")
    require(package["audit"]["sha256"] == load_dataset(data_dir)["audit"]["sha256"],
            "Frozen course files changed during audit")
    report = {
        "schema_version": 1, "status": "passed_recorded_evidence_consistency",
        "scope": "All frozen v1 rows, individual raw trials and sampled trajectories; no simulation rerun",
        "source_repository": SOURCE_REPOSITORY, "source_commit": SOURCE_COMMIT,
        "source_study": STUDY, "frozen_package_validation": package["audit"],
        "counts": dict(counts), "evidence_files_checked": len(reader.files),
        "core_source_provenance": source_checks, "coverage": coverage,
        "training_only_diagnostics": {key: dict(sorted(value.items())) for key, value in training.items()},
        "training_case_count": len(cases), "labels_modified": 0,
        "max_feature_absolute_error": max(row["max_feature_absolute_error"] for row in audited),
        "max_trace_sample_gap_s": max(row["max_sample_gap_s"] for row in audited),
        "software": {"python": sys.version, "platform": platform.platform()},
        "implementation_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                  for name in ("audit_data_quality.py", "audit_coverage.py", "audit_trace.py", "baseline.py", "uv.lock")},
        "wall_s": time.monotonic() - started, "limits": LIMITS,
    }
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "report.json", report)
    save_json(output / "evidence_manifest.json", {
        "source_repository": SOURCE_REPOSITORY, "source_commit": SOURCE_COMMIT,
        "files": dict(sorted(reader.files.items())),
    })
    save_json(output / "training_cases.json", cases)
    with (output / "trial_audit.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audited[0]))
        writer.writeheader()
        writer.writerows({key: canonical(value) if isinstance(value, (dict, list)) else value
                          for key, value in row.items()} for row in audited)
    print(json.dumps({"status": report["status"], "counts": report["counts"],
                      "evidence_files_checked": report["evidence_files_checked"],
                      "wall_s": report["wall_s"], "output": str(output)}, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("Output must be a new path; existing files/directories and symlinks are refused")
    if args.output.resolve().is_relative_to(args.data_dir.resolve()):
        parser.error("Audit output must not be written inside the frozen data directory")
    run_audit(args.source_repo, args.data_dir, args.output)


if __name__ == "__main__":
    main()
