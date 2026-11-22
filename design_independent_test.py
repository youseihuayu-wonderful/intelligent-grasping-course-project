"""Predeclare a public development pilot and a private, unexecuted final test plan."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import product
import hashlib
import json
import math
from pathlib import Path
import random
import secrets

from audit_data_quality import FrozenEvidence, ROOT, SOURCE_COMMIT, STUDY, decode_json, require
from baseline import FEATURES, FROZEN_FILE_HASHES

CONTEXT = FEATURES[:7]
COMMANDS = FEATURES[7:]
BIASES = ("bias_x_m", "bias_y_m", "bias_z_m")
PROTOCOL_ID = "course_independent_test_v1"
PILOT_SEED = hashlib.sha256(b"course development pilot v1:564403").hexdigest()
FORBIDDEN_OUTCOMES = {"success", "failure_kind", "failure_reason", "final_pose",
                      "execution_sim_s", "verified_hold_s", "p_success", "predicted_success"}


def payload(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def digest(content):
    return hashlib.sha256(content).hexdigest()


def key(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def property_key(settings):
    return tuple(settings[name] for name in ("target", "scale", "mass_kg", "sliding_friction"))


def group_id(condition, nominal):
    tokens = sorted(["condition:" + condition, "nominal:" + key(nominal)])
    return "group_" + digest(key(tokens).encode())


def assert_unlabelled(value):
    if isinstance(value, dict):
        require(not (set(value) & FORBIDDEN_OUTCOMES), "Outcome/prediction field in an unexecuted plan")
        for item in value.values():
            assert_unlabelled(item)
    elif isinstance(value, list):
        for item in value:
            assert_unlabelled(item)


def make_policy(historical, environment):
    return {
        "schema_version": 1, "protocol_id": PROTOCOL_ID,
        "status": "preregistered_design_only_not_collected",
        "design_code_sha256": digest(Path(__file__).read_bytes()),
        "source_commit": SOURCE_COMMIT,
        "historical_protocol_payload_sha256": digest(payload(historical)),
        "frozen_v1_sha256": dict(FROZEN_FILE_HASHES),
        "source_environment_payload_sha256": digest(payload(environment)),
        "physics_collector_sha256": environment["physical_collector_sha256"],
        "features": list(FEATURES),
        "target_population": "New poses and latent execution errors in the known five-family simulator domain; not unseen shapes or sim-to-real",
        "scale_levels": historical["scale_levels"],
        "sliding_friction_levels": historical["sliding_friction_levels"],
        "family_mass_levels_kg": {name: obj["mass_levels_kg"] for name, obj in historical["study_objects"].items()},
        "pose_bounds": historical["pose_bounds"], "minimum_matched_property_xy_separation_m": .005,
        "pose_sampling": "Latin hypercube within each family/role; reject whole candidate pose designs only for predeclared proximity to historical/pilot matched-property poses, never outcomes",
        "maximum_pose_design_draws": 1000,
        "control_levels": historical["control_levels"],
        "baseline_control": historical["baseline_control"],
        "controls_per_condition": 8, "paired_draws_per_condition": 2,
        "latent_bias_uniform_bounds_m": [-.004, .004],
        "control_sampling": "Baseline, high-force anchor, two rotating one-command contrasts, both nonbaseline speeds and two rotating multi-command configurations; full 54-command pool per final-test family",
        "pilot": {"condition_groups": 20, "groups_per_family": 4, "planned_attempts": 320,
                  "split": "train", "use": "development_only_never_final_test",
                  "property_sampling": "Four fixed light/high-friction to heavy/low-friction corners; all family mass and friction levels and all three scales, not a representative prevalence sample"},
        "final": {"condition_groups": 240, "groups_per_family": 48, "planned_attempts": 3840,
                  "split": "test", "property_sampling": "Complete 3 scale x 4 family mass x 4 friction grid once per family",
                  "secret_seed_bits": 256, "release": "Inputs, seed and labels remain outside Git until models, predictions and final evaluation are frozen"},
        "label_rule": {"positive_class": 1, "negative_class": 0,
                       "required_controller_status": "succeeded",
                       "continuous_observer_hold_s": 2., "minimum_lift_m": .10,
                       "relative_position_displacement_less_than_m": .01,
                       "predicates": ["bilateral_grip", "clear_of_table", "target_lifted", "relative_pose_stable"],
                       "physics_step_s": .002,
                       "sampling_limit": "Retain existing raw evidence; sampled traces alone cannot independently reconstruct unsaved contacts"},
        "collection_rules": {
            "models_frozen_before_final_collection": True,
            "freeze_required": ["model_artifact_hashes", "training_code_revision", "feature_allowlist",
                                "v1_and_cv_hashes", "hyperparameters", "preprocessing_and_selection",
                                "calibration_if_any", "dependency_versions", "threshold_and_tie_rule"],
            "fixed_attempt_budget_no_outcome_stopping": True,
            "no_label_balance_target": True,
            "ordinary_controller_failure": "Keep as label 0; never discard because unfavorable",
            "unusable_attempt": "Preserve invalid initialization, internal errors and numerical warnings as explicit nonlabels; no replacements or selective retries",
            "incomplete_or_unusable_final_batch": "Block confirmatory comparison, preserve all evidence and diagnose. Any changed environment/protocol needs a new version; a replacement run must be a complete declared batch, not cherry-picked trials",
            "pilot_does_not_modify_final_plan": True,
        },
        "evaluation": {
            "model_ids": ["logistic", "lasso_logistic", "random_forest", "xgboost"],
            "constant_reference": "Frozen training-only success prevalence",
            "fit_policy": "Training CV for tuning; validation for documented development; final pipelines fitted on v1 train only for this comparison",
            "primary_metric": "log_loss", "secondary_metrics": ["brier_score", "roc_auc", "balanced_accuracy", "failure_recall", "confusion_matrix", "class_support"],
            "probability_class": 1, "threshold": .5, "threshold_tie_class": 1,
            "probability_dtype": "float64", "log_loss_clipping_epsilon": 2.220446049250313e-16,
            "predictions_locked_before_test_labels_joined": True,
            "all_models_same_trial_ids": True,
            "bootstrap": {"resamples": 2000, "seed": 5644, "unit": "condition_group",
                          "strata": "target_family", "paired_across_models": True,
                          "interval": "95% percentile; descriptive, not multiplicity-adjusted superiority testing"},
            "undefined_metric": "Report null with reason; do not invent zero or collect extra data to change support",
            "scope": "Classifier probability comparison on predeclared candidates; no claim of faster physical execution or globally optimal control",
        },
    }


def legacy_exclusions(historical, other_plans):
    groups = list(historical["condition_groups"])
    seeds = {spec[2]["nuisance_seed"] for spec in historical["trials"]}
    identifiers = {group["condition_id"] for group in groups}
    for plan in other_plans:
        groups.extend(plan["condition_groups"])
        seeds.update(spec[2]["nuisance_seed"] for spec in plan["trials"])
        identifiers.update(group["condition_id"] for group in plan["condition_groups"])
    positions = defaultdict(list)
    for group in groups:
        nominal = group["nominal_settings"]
        positions[property_key(nominal)].append((nominal["x_m"], nominal["y_m"]))
    return positions, seeds, identifiers


def properties(policy, target, role):
    masses = policy["family_mass_levels_kg"][target]
    scales, frictions = policy["scale_levels"], policy["sliding_friction_levels"]
    if role == "final":
        return list(product(scales, masses, frictions))
    return [(scales[i % 3], masses[i], frictions[3-i]) for i in range(4)]


def build_plan(policy, historical, role, seed_hex, *, other_plans=()):
    require(role in ("pilot", "final"), "Unknown plan role")
    require(isinstance(seed_hex, str) and len(seed_hex) == 64 and all(c in "0123456789abcdef" for c in seed_hex),
            "Planning seed must contain 256 bits encoded as lowercase hex")
    rng = random.Random(int(seed_hex, 16))
    positions, used_seeds, _ = legacy_exclusions(historical, other_plans)
    baseline = policy["baseline_control"]
    pool = [dict(zip(COMMANDS, values)) for values in product(*(policy["control_levels"][name] for name in COMMANDS))]
    joint = [control for control in pool if sum(control[name] != baseline[name] for name in COMMANDS) > 1]
    rng.shuffle(joint)
    contrasts = [{**baseline, "opening_m": .04}, {**baseline, "closing_force_n": 2.},
                 {**baseline, "close_trigger_m": .01}, {**baseline, "close_trigger_m": .02}]
    groups, trials = [], []
    for target_index, target in enumerate(sorted(policy["family_mass_levels_kg"])):
        grid = properties(policy, target, role)
        n = len(grid)
        for _ in range(policy["maximum_pose_design_draws"]):
            poses = [{} for _ in grid]
            for axis in ("x_m", "y_m", "yaw_rad"):
                low, high = policy["pose_bounds"][axis]
                strata = list(range(n))
                rng.shuffle(strata)
                for pose, stratum in zip(poses, strata):
                    pose[axis] = low + (high-low) * (stratum+rng.random()) / n
            if all(all(math.dist((pose["x_m"], pose["y_m"]), old) >= policy["minimum_matched_property_xy_separation_m"]
                       for old in positions[(target, scale, mass, friction)])
                   for (scale, mass, friction), pose in zip(grid, poses)):
                break
        else:
            raise ValueError(f"Could not satisfy preregistered nominal pose separation for {target}")
        for index, ((scale, mass, friction), pose) in enumerate(zip(grid, poses)):
            condition = f"{PROTOCOL_ID}.{role}.{target}_{index:02d}"
            nominal = dict(target=target, scale=scale, mass_kg=mass, sliding_friction=friction, **pose)
            ordinal = target_index*n + index
            controls = [dict(baseline), {**baseline, "closing_force_n": 8.},
                        contrasts[index % 4].copy(), contrasts[(index+1) % 4].copy(),
                        {**baseline, "descent_speed_m_s": .04}, {**baseline, "descent_speed_m_s": .20},
                        joint[(ordinal*2) % len(joint)].copy(), joint[(ordinal*2+1) % len(joint)].copy()]
            groups.append({"condition_id": condition, "group_id": group_id(condition, nominal),
                           "split": policy[role]["split"], "nominal_settings": nominal, "controls": controls})
            for draw in range(2):
                seed = rng.getrandbits(32)
                while seed in used_seeds:
                    seed = rng.getrandbits(32)
                used_seeds.add(seed)
                error_rng = random.Random(seed)
                error = {name: error_rng.uniform(-.004, .004) for name in BIASES}
                for control_id, control in enumerate(controls):
                    trial_id = f"{condition}.control_{control_id}.draw_{draw}"
                    trials.append([trial_id, {**nominal, **control, **error}, {
                        "design": "independent_test_design", "protocol_id": PROTOCOL_ID,
                        "condition_id": condition, "split": policy[role]["split"],
                        "control_id": control_id, "nuisance_seed": seed,
                        "error_observability": "Latent execution uncertainty; excluded from predictive inputs",
                    }])
    return {"schema_version": 1, "protocol_id": PROTOCOL_ID, "role": role,
            "status": "planned_not_collected", "seed_hex": seed_hex,
            "policy_sha256": digest(payload(policy)),
            "condition_group_count": len(groups), "planned_attempts": len(trials),
            "controls_per_condition": 8, "paired_draws_per_condition": 2,
            "condition_groups": groups, "trials": trials}


def validate_plan(plan, historical, policy, *, other_plans=()):
    assert_unlabelled(plan)
    role = plan.get("role")
    require(role in ("pilot", "final") and plan.get("protocol_id") == PROTOCOL_ID, "Wrong plan identity")
    require(plan.get("status") == "planned_not_collected", "Plan is not an unexecuted design")
    require(plan.get("policy_sha256") == digest(payload(policy)), "Plan/policy commitment mismatch")
    # Regeneration binds every draw/setting to the recorded seed without labels.
    expected = build_plan(policy, historical, role, plan["seed_hex"], other_plans=other_plans)
    require(plan == expected, "Plan differs from its deterministic seed, policy or exclusion inputs")
    require(plan["condition_group_count"] == policy[role]["condition_groups"]
            and plan["planned_attempts"] == policy[role]["planned_attempts"], "Wrong fixed trial budget")
    positions, old_seeds, old_ids = legacy_exclusions(historical, other_plans)
    conditions, contexts, groups, trial_ids = {}, set(), set(), set()
    families = Counter()
    family_controls = defaultdict(set)
    min_separation = math.inf
    pool = {tuple(values) for values in product(*(policy["control_levels"][name] for name in COMMANDS))}
    for condition in plan["condition_groups"]:
        cid, nominal = condition["condition_id"], condition["nominal_settings"]
        require(cid not in old_ids and cid not in conditions, "Reused historical/pilot condition ID")
        require(set(nominal) == set(CONTEXT) and all(type(nominal[name]) in (int, float)
                and math.isfinite(nominal[name]) for name in CONTEXT[1:]), "Malformed nominal context")
        target = nominal["target"]
        require(target in policy["family_mass_levels_kg"], "Unknown object family")
        require((nominal["scale"], nominal["mass_kg"], nominal["sliding_friction"]) in properties(policy, target, role),
                "Property combination outside preregistered role grid")
        require(key(nominal) not in contexts, "Duplicate nominal context")
        contexts.add(key(nominal))
        require(condition["group_id"] == group_id(cid, nominal) and condition["group_id"] not in groups,
                "Invalid or repeated label-independent group ID")
        groups.add(condition["group_id"])
        require(condition["split"] == policy[role]["split"], "Plan role leaks into another partition")
        for old in positions[property_key(nominal)]:
            distance = math.dist((nominal["x_m"], nominal["y_m"]), old)
            min_separation = min(min_separation, distance)
            require(distance >= policy["minimum_matched_property_xy_separation_m"], "Reused/too-close matched-property historical pose")
        controls = [tuple(control[name] for name in COMMANDS) for control in condition["controls"]]
        require(len(controls) == len(set(controls)) == 8 and set(controls) <= pool, "Invalid candidate commands")
        family_controls[target].update(controls)
        families[target] += 1
        conditions[cid] = condition
    require(set(families) == set(policy["family_mass_levels_kg"])
            and set(families.values()) == {policy[role]["groups_per_family"]}, "Unequal or missing family groups")
    for target in families:
        members = [g["nominal_settings"] for g in conditions.values() if g["nominal_settings"]["target"] == target]
        actual = Counter((g["scale"], g["mass_kg"], g["sliding_friction"]) for g in members)
        require(actual == Counter(properties(policy, target, role)), "Incomplete or repeated property grid")
        for axis, (low, high) in policy["pose_bounds"].items():
            require(all(low <= g[axis] < high for g in members), "Nominal pose outside domain")
            strata = [math.floor((g[axis]-low)/(high-low)*len(members)) for g in members]
            require(sorted(strata) == list(range(len(members))), "Missing or repeated pose strata")
        if role == "final":
            require(family_controls[target] == pool and len(pool) == 54, "Final family lacks full candidate-pool coverage")
    draws = defaultdict(lambda: defaultdict(dict))
    seed_owner = {}
    for spec in plan["trials"]:
        require(isinstance(spec, list) and len(spec) == 3, "Collector requires [trial_id, settings, metadata]")
        trial_id, settings, metadata = spec
        require(trial_id not in trial_ids and set(settings) == set(FEATURES + BIASES), "Repeated trial ID or invalid settings")
        trial_ids.add(trial_id)
        cid, control_id, seed = metadata["condition_id"], metadata["control_id"], metadata["nuisance_seed"]
        require(cid in conditions and type(control_id) is int and 0 <= control_id < 8, "Invalid condition/control reference")
        require(type(seed) is int and 0 <= seed < 2**32 and seed not in old_seeds, "Reused or invalid nuisance seed")
        require(seed not in seed_owner or seed_owner[seed] == cid, "Nuisance seed crosses condition groups")
        seed_owner[seed] = cid
        condition = conditions[cid]
        require(metadata["split"] == policy[role]["split"] and metadata["protocol_id"] == PROTOCOL_ID, "Trial partition/protocol mismatch")
        require({name: settings[name] for name in CONTEXT} == condition["nominal_settings"]
                and {name: settings[name] for name in COMMANDS} == condition["controls"][control_id], "Trial settings differ from frozen condition")
        error_rng = random.Random(seed)
        biases = tuple(error_rng.uniform(-.004, .004) for _ in BIASES)
        require(tuple(settings[name] for name in BIASES) == biases, "Latent bias does not match nuisance seed")
        require(control_id not in draws[cid][seed], "Repeated control within nuisance draw")
        draws[cid][seed][control_id] = biases
    require(set(draws) == set(conditions), "Missing trial condition")
    for paired in draws.values():
        require(len(paired) == 2 and all(set(controls) == set(range(8)) and len(set(controls.values())) == 1
                                        for controls in paired.values()), "Draws are not two shared-error eight-control pairs")
    return {"role": role, "condition_groups": len(conditions), "planned_attempts": len(trial_ids),
            "groups_per_family": dict(families),
            "distinct_controls_per_family": {name: len(values) for name, values in family_controls.items()},
            "minimum_matched_property_xy_separation_m": min_separation if math.isfinite(min_separation) else None,
            "historical_condition_or_nuisance_seed_reuse": 0,
            "collector_spec_contract_checked": True, "physical_initialization_checked": False,
            "outcomes_present": False, "simulation_trials_executed": 0}


def check_output_paths(public_dir, private_dir):
    public = Path(public_dir).resolve()
    private = Path(private_dir).resolve()
    require(public.is_relative_to(ROOT) and not public.is_relative_to(ROOT / "data"),
            "Public design must be inside this course repository, outside frozen data")
    require(not private.is_relative_to(public) and not public.is_relative_to(private), "Public and private output paths overlap")
    require(not any((parent / ".git").exists() for parent in (private, *private.parents)),
            "Private final plan must be outside every Git checkout")
    for path in (Path(public_dir), Path(private_dir)):
        require(not path.exists() and not path.is_symlink(), "Design output paths must be new, not existing paths or symlinks")
    return public, private


def source_inputs(source_repo):
    reader = FrozenEvidence(source_repo)
    historical = reader.json(f"{STUDY}/protocol.json")
    environment = reader.json(f"{STUDY}/environment.json")
    for name, expected in FROZEN_FILE_HASHES.items():
        require(digest((ROOT / "data/v1" / name).read_bytes()) == expected, "Frozen v1 file changed")
    return historical, environment, reader.files


def verify_bundle(public_dir, private_dir, historical, environment):
    public, private = Path(public_dir), Path(private_dir)
    private = private.resolve()
    require(not any((parent / ".git").exists() for parent in (private, *private.parents)),
            "Private final plan must be outside every Git checkout")
    require(private.stat().st_mode & 0o077 == 0
            and (private / "final_plan.json").stat().st_mode & 0o077 == 0,
            "Private final plan permissions must exclude other users")
    require({p.name for p in public.iterdir()} == {"policy.json", "pilot_plan.json", "final_commitment.json", "design_validation.json"},
            "Unexpected/missing public design artifact; final inputs must not be published")
    policy = decode_json((public / "policy.json").read_bytes())
    require(policy == make_policy(historical, environment), "Published policy differs from pinned design code/source")
    pilot_bytes = (public / "pilot_plan.json").read_bytes()
    pilot = decode_json(pilot_bytes)
    final_bytes = (private / "final_plan.json").read_bytes()
    final = decode_json(final_bytes)
    commitment = decode_json((public / "final_commitment.json").read_bytes())
    require(set(commitment) == {"schema_version", "protocol_id", "policy_sha256", "pilot_plan_sha256", "final_plan_sha256", "final_plan_bytes", "condition_groups", "planned_attempts", "status"},
            "Unexpected final commitment field / possible private-input disclosure")
    require(commitment["final_plan_sha256"] == digest(final_bytes)
            and commitment["final_plan_bytes"] == len(final_bytes)
            and commitment["policy_sha256"] == digest((public / "policy.json").read_bytes())
            and commitment["pilot_plan_sha256"] == digest(pilot_bytes), "Final plan or public artifact commitment mismatch")
    require(commitment["protocol_id"] == PROTOCOL_ID and commitment["schema_version"] == 1
            and commitment["condition_groups"] == policy["final"]["condition_groups"]
            and commitment["planned_attempts"] == policy["final"]["planned_attempts"]
            and commitment["status"] == "planned_not_collected_private_inputs_withheld", "Wrong final commitment identity/budget")
    require(all(final["seed_hex"].encode() not in path.read_bytes() for path in public.iterdir()), "Final planning seed leaked into public artifacts")
    result = {"pilot": validate_plan(pilot, historical, policy),
              "final": validate_plan(final, historical, policy, other_plans=(pilot,)),
              "final_plan_sha256": digest(final_bytes), "policy_sha256": digest(payload(policy)),
              "status": "passed_design_checks_only_not_physics", "labels_generated": 0,
              "models_trained_or_scored": 0}
    require(decode_json((public / "design_validation.json").read_bytes()) == result,
            "Saved design validation differs from independently read bundle")
    return result


def build_bundle(source_repo, public_dir, private_dir):
    public, private = check_output_paths(public_dir, private_dir)
    historical, environment, _ = source_inputs(source_repo)
    policy = make_policy(historical, environment)
    pilot = build_plan(policy, historical, "pilot", PILOT_SEED)
    final = build_plan(policy, historical, "final", secrets.token_hex(32), other_plans=(pilot,))
    final_bytes, pilot_bytes = payload(final), payload(pilot)
    result = {"pilot": validate_plan(pilot, historical, policy),
              "final": validate_plan(final, historical, policy, other_plans=(pilot,)),
              "final_plan_sha256": digest(final_bytes), "policy_sha256": digest(payload(policy)),
              "status": "passed_design_checks_only_not_physics", "labels_generated": 0,
              "models_trained_or_scored": 0}
    commitment = {"schema_version": 1, "protocol_id": PROTOCOL_ID,
                  "policy_sha256": digest(payload(policy)), "pilot_plan_sha256": digest(pilot_bytes),
                  "final_plan_sha256": digest(final_bytes), "final_plan_bytes": len(final_bytes),
                  "condition_groups": len(final["condition_groups"]), "planned_attempts": len(final["trials"]),
                  "status": "planned_not_collected_private_inputs_withheld"}
    private.mkdir(parents=True, mode=0o700, exist_ok=False)
    private.chmod(0o700)
    with (private / "final_plan.json").open("xb") as stream:
        stream.write(final_bytes)
    (private / "final_plan.json").chmod(0o600)
    public.mkdir(parents=True, exist_ok=False)
    for name, content in (("policy.json", payload(policy)), ("pilot_plan.json", pilot_bytes),
                          ("final_commitment.json", payload(commitment)), ("design_validation.json", payload(result))):
        with (public / name).open("xb") as stream:
            stream.write(content)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify"))
    parser.add_argument("--source-repo", required=True, type=Path)
    parser.add_argument("--public-dir", required=True, type=Path)
    parser.add_argument("--private-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "build":
        result = build_bundle(args.source_repo, args.public_dir, args.private_dir)
    else:
        historical, environment, _ = source_inputs(args.source_repo)
        result = verify_bundle(args.public_dir, args.private_dir, historical, environment)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
