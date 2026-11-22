"""Execute only the frozen development pilot; preserve every attempt and raw outcome."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import platform
import sys
import time
import traceback

from design_independent_test import (PILOT_SEED, ROOT, SOURCE_COMMIT, digest, make_policy,
                                     payload, source_inputs, validate_plan)
from audit_data_quality import decode_json, require
from pilot_physics import run_trial

PILOT_PLAN_SHA256 = "ca384e9b365eddaa56c6aa09762181eeca3706773645ced3db01bed508a8f2b9"
POLICY_SHA256 = "db190249e6d202a0a07ee474aabd13132593be13dc5eaa68485a7b6638c48e99"


def sha_file(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload(value))


def load_pilot(source_repo, plan_dir):
    plan_dir = Path(plan_dir)
    policy_bytes = (plan_dir / "policy.json").read_bytes()
    plan_bytes = (plan_dir / "pilot_plan.json").read_bytes()
    require(digest(plan_bytes) == PILOT_PLAN_SHA256 and digest(policy_bytes) == POLICY_SHA256,
            "Pilot plan/policy differs from the preregistered bytes")
    plan, policy = decode_json(plan_bytes), decode_json(policy_bytes)
    require(plan["role"] == "pilot" and plan["seed_hex"] == PILOT_SEED,
            "This runner refuses final-test or unregistered plans")
    historical, environment, source_files = source_inputs(source_repo)
    require(policy == make_policy(historical, environment), "Planning code/source/policy drift")
    validation = validate_plan(plan, historical, policy)
    require(validation["planned_attempts"] == 320 and validation["condition_groups"] == 20,
            "Wrong development pilot budget")
    return plan, policy, historical, environment, source_files


def runtime_fingerprint(source_repo, expected):
    source = Path(source_repo).resolve(strict=True)
    files = dict(expected["input_sha256"])
    files["collect_grasp_classifiers.py"] = expected["classifier_collector_sha256"]
    files["grasp_study_physics.py"] = expected["physical_collector_sha256"]
    observed = {name: sha_file(source/name) for name in files}
    mismatches = [name for name in files if observed[name] != files[name]]
    packages = {name: importlib.metadata.version(name) for name in expected["packages"]}
    require(not mismatches, f"Physical runtime source/asset drift: {mismatches}")
    require(packages == expected["packages"], "Run under the original simulation lockfile; package versions differ")
    require(sys.version_info[:2] == (3, 12), "Original physical runtime requires Python 3.12")
    return {"input_sha256": observed, "runtime_input_files_checked": len(files),
            "input_mismatches": [], "packages": packages,
            "python": sys.version, "platform": platform.platform()}


def protected_hashes(plan_dir):
    root = Path(plan_dir).resolve()
    paths = [ROOT/"data/v1"/name for name in ("dataset.csv", "split_assignments.csv", "cv_manifest.json")]
    paths += [root/name for name in ("policy.json", "pilot_plan.json", "final_commitment.json", "design_validation.json")]
    paths += [ROOT/"design_independent_test.py"]
    return {str(path.relative_to(ROOT)): sha_file(path) for path in paths}


def validate_output(output, source_repo):
    path = Path(output)
    resolved = path.resolve()
    require(not path.exists() and not path.is_symlink(), "Pilot evidence output must be new")
    require(not any((parent/".git").exists() for parent in (resolved, *resolved.parents)),
            "Write large pilot evidence outside Git; publish its lossless archive separately")
    private_root = ROOT.parent/"intelligent-grasping-private-evaluation"
    require(not resolved.is_relative_to(private_root.resolve()), "Do not write pilot data into final-test custody")
    require(not resolved.is_relative_to(Path(source_repo).resolve()), "Do not mutate original simulation evidence")
    return resolved


def collect(source_repo, plan_dir, output, workers=3):
    require(type(workers) is int and workers > 0, "workers must be a positive integer")
    output = validate_output(output, source_repo)
    plan, policy, historical, expected, source_files = load_pilot(source_repo, plan_dir)
    before = runtime_fingerprint(source_repo, expected)
    protected = protected_hashes(plan_dir)
    output.mkdir(parents=True, exist_ok=False)
    save(output/"plan.json", plan)
    save(output/"policy.json", policy)
    save(output/"source_environment.json", expected)
    save(output/"runtime_before.json", before)
    save(output/"source_artifacts.json", {"source_commit": SOURCE_COMMIT, "files": source_files})
    implementation = {name: sha_file(ROOT/name) for name in ("run_development_pilot.py", "pilot_physics.py")}
    save(output/"run_manifest.json", {
        "status": "started", "role": "development_pilot_only", "plan_sha256": PILOT_PLAN_SHA256,
        "policy_sha256": POLICY_SHA256, "source_commit": SOURCE_COMMIT,
        "implementation_sha256": implementation, "workers": workers,
        "protected_input_sha256": protected, "planned_attempts": len(plan["trials"]),
        "observer": "Read-only per-physics-step contacts/references; original run_one/control/label unchanged",
        "final_test_plan_opened": False, "final_test_trials_executed": 0,
    })
    started = time.monotonic()
    outcomes = {}
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        pending = {pool.submit(run_trial, spec, str(Path(source_repo).resolve()), str(output/"collection")): spec
                   for spec in plan["trials"]}
        for future in as_completed(pending):
            spec = pending[future]
            try:
                result = future.result()
            except Exception as error:
                # A worker failure is a preserved nonlabel, never a failed grasp.
                raw = {"trial_id": spec[0], "settings": spec[1], "metadata": spec[2],
                       "trial_status": "worker_error", "success": None,
                       "error_type": type(error).__name__, "error_message": str(error)}
                result = {"raw": raw, "instrumentation": {"trial_id": spec[0], "status": "internal_error",
                          "error_type": type(error).__name__, "error_message": str(error),
                          "traceback": traceback.format_exc(), "dense_samples": 0,
                          "read_only_checks_passed": False, "python_warnings": [],
                          "python_numeric_warning_count": 0}}
                save(output/"collection/errors"/f"{spec[0]}.json", result)
            require(result["raw"]["trial_id"] == spec[0] and spec[0] not in outcomes,
                    "Worker returned wrong or repeated trial identity")
            outcomes[spec[0]] = result
            raw, meta = result["raw"], result["instrumentation"]
            print(json.dumps({"completed": len(outcomes), "planned": len(plan["trials"]),
                              "trial_id": spec[0], "trial_status": raw["trial_status"],
                              "success": raw.get("success"),
                              "python_numeric_warnings": meta.get("python_numeric_warning_count", 0)}), flush=True)
    rows = [outcomes[spec[0]]["raw"] for spec in plan["trials"]]
    metas = [outcomes[spec[0]]["instrumentation"] for spec in plan["trials"]]
    save(output/"collection/trials.json", {"rows": rows})
    save(output/"collection/instrumentation.json", {"rows": metas})
    usable = [raw for raw, meta in zip(rows, metas)
              if raw["trial_status"] == "completed" and type(raw.get("success")) is bool
              and not raw.get("warnings") and not meta.get("python_numeric_warning_count")
              and meta.get("status") == "completed" and meta.get("read_only_checks_passed")]
    after = runtime_fingerprint(source_repo, expected)
    save(output/"runtime_after.json", after)
    unchanged = protected == protected_hashes(plan_dir) and before["input_sha256"] == after["input_sha256"]
    summary = {"planned_attempts": len(plan["trials"]), "returned_attempt_records": len(rows),
               "raw_statuses": dict(Counter(row["trial_status"] for row in rows)),
               "usable_label_candidates": len(usable), "unusable_attempts": len(rows)-len(usable),
               "raw_successes": sum(row.get("success") is True for row in rows),
               "raw_failures": sum(row.get("success") is False for row in rows),
               "mujoco_warning_trials": sum(bool(row.get("warnings")) for row in rows),
               "python_numeric_warning_trials": sum(bool(meta.get("python_numeric_warning_count")) for meta in metas),
               "dense_samples": sum(meta.get("dense_samples", 0) for meta in metas),
               "protected_inputs_unchanged": unchanged, "wall_s": time.monotonic()-started,
               "final_test_trials_executed": 0,
               "status": "collection_complete_pending_independent_audit" if len(usable) == 320 and unchanged
                         else "collection_has_unusable_attempts_or_input_drift"}
    save(output/"collection/summary.json", summary)
    print("PILOT_COLLECTION_FINISHED " + json.dumps(summary), flush=True)
    require(unchanged and len(usable) == 320, "Pilot evidence retained, but collection cannot pass its quality gate")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--plan-dir", type=Path, default=ROOT/"protocols/independent_test_v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    collect(args.source_repo, args.plan_dir, args.output, args.workers)


if __name__ == "__main__":
    main()
