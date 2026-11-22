"""Independent, label-blind protocol checks and training-only descriptive diagnostics."""

from collections import Counter, defaultdict
from itertools import product
import hashlib
import json
import math


_CONTEXT = ("target", "scale", "mass_kg", "sliding_friction", "x_m", "y_m", "yaw_rad")
_COMMANDS = ("opening_m", "closing_force_n", "close_trigger_m", "descent_speed_m_s")
_FEATURES = _CONTEXT + _COMMANDS
_BIASES = ("bias_x_m", "bias_y_m", "bias_z_m")
_SPLITS = ("train", "validation", "test")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _number(value, name, *, csv=False):
    _require(not isinstance(value, bool), f"{name}: boolean is not a numeric setting")
    if csv:
        _require(isinstance(value, str), f"{name}: expected CSV numeric string")
        try:
            result = float(value)
        except ValueError as error:
            raise ValueError(f"{name}: invalid numeric value {value!r}") from error
    else:
        _require(isinstance(value, (float, int)), f"{name}: expected numeric scalar")
        result = float(value)
    _require(math.isfinite(result), f"{name}: nonfinite numeric value")
    return 0.0 if result == 0 else result


def _mapping(value, name):
    _require(isinstance(value, dict), f"{name}: expected dictionary")
    return value


def _field(mapping, key, name):
    _require(key in mapping, f"{name}: missing {key}")
    return mapping[key]


def _text(value, name):
    _require(isinstance(value, str) and bool(value), f"{name}: expected nonempty string")
    return value


def _settings(value, names, name):
    _mapping(value, name)
    _require(set(value) == set(names), f"{name}: unexpected or missing setting keys")
    return {key: (_text(value[key], f"{name}.{key}") if key == "target"
                  else _number(value[key], f"{name}.{key}")) for key in names}


def _index(rows, name):
    _require(isinstance(rows, list), f"{name}: expected list")
    indexed = {}
    for row in rows:
        _mapping(row, name)
        trial_id = _text(_field(row, "trial_id", name), f"{name}.trial_id")
        _require(trial_id not in indexed, f"{name}: duplicate trial_id {trial_id}")
        indexed[trial_id] = row
    return indexed


def _same_ids(actual, expected, name):
    missing, extra = sorted(expected - actual), sorted(actual - expected)
    _require(not missing and not extra,
             f"{name}: planned trial ID mismatch; missing={missing[:3]}, extra={extra[:3]}")


def _counts_equal(actual, declared, name):
    _mapping(declared, name)
    _require(all(type(count) is int and count >= 0 for count in declared.values()),
             f"{name}: counts must be nonnegative integers")
    _require(dict(actual) == declared, f"{name}: actual {dict(actual)} != declared {declared}")


def _duplicate_summary(buckets, labels, splits):
    repeated = [ids for ids in buckets.values() if len(ids) > 1]
    return {
        "distinct_vectors": len(buckets),
        "duplicate_vector_groups": len(repeated),
        "rows_in_duplicate_vectors": sum(map(len, repeated)),
        "redundant_rows": sum(len(ids) - 1 for ids in repeated),
        "conflicting_label_groups": sum(len({labels[i] for i in ids}) > 1 for ids in repeated),
        "cross_split_groups": sum(len({splits[i] for i in ids}) > 1 for ids in repeated),
    }


def audit_coverage(csv_rows: list[dict[str, str]], raw_rows: list[dict], protocol: dict) -> dict:
    """Validate the frozen prospective plan; never use labels to assess coverage.

    Observable duplicate vectors are descriptive, not errors: the deliberately
    hidden paired execution biases can yield different outcomes. Exact complete
    physical settings plus an initial-state fingerprint repeated in two trials
    are instead rejected, as are condition/group or nominal-context leakage.
    Outcome-dependent exploration below is restricted to the training partition.
    """
    _mapping(protocol, "protocol")
    csv_by_id, raw_by_id = _index(csv_rows, "CSV"), _index(raw_rows, "raw")
    trials = _field(protocol, "trials", "protocol")
    _require(isinstance(trials, list), "protocol.trials: expected list")
    planned = {}
    for trial in trials:
        _require(isinstance(trial, (list, tuple)) and len(trial) == 3,
                 "protocol.trials: expected [trial_id, settings, metadata]")
        trial_id, settings, metadata = trial
        _text(trial_id, "planned trial_id")
        _require(trial_id not in planned, f"protocol: duplicate planned trial_id {trial_id}")
        planned[trial_id] = (settings, metadata)
    _same_ids(set(raw_by_id), set(planned), "raw")
    _same_ids(set(csv_by_id), set(planned), "CSV")
    _require(type(protocol.get("planned_attempts")) is int
             and len(planned) == protocol["planned_attempts"], "planned_attempts mismatch")
    _require(protocol.get("controls_per_condition") == 8, "controls_per_condition must be eight")
    _require(protocol.get("paired_draws_per_condition") == 2, "paired_draws_per_condition must be two")

    families = _mapping(_field(protocol, "study_objects", "protocol"), "study_objects")
    _require(bool(families), "study_objects: empty family declaration")
    groups = _field(protocol, "condition_groups", "protocol")
    _require(isinstance(groups, list), "condition_groups: expected list")
    conditions, nominal_owner = {}, {}
    group_counts, family_counts = Counter(), {family: Counter() for family in families}
    grids, family_controls = {family: Counter() for family in families}, defaultdict(set)
    poses = defaultdict(lambda: defaultdict(list))
    bounds = _mapping(_field(protocol, "pose_bounds", "protocol"), "pose_bounds")
    _require(set(bounds) == {"x_m", "y_m", "yaw_rad"}, "pose_bounds: expected x/y/yaw axes")
    for axis, interval in bounds.items():
        _require(isinstance(interval, list) and len(interval) == 2, f"pose_bounds.{axis}: expected interval")
        low, high = [_number(v, f"pose_bounds.{axis}") for v in interval]
        _require(low < high, f"pose_bounds.{axis}: invalid interval")
    levels = _mapping(_field(protocol, "control_levels", "protocol"), "control_levels")
    _require(set(levels) == set(_COMMANDS), "control_levels: missing or unexpected command")
    for axis in _COMMANDS:
        _require(isinstance(levels[axis], list) and bool(levels[axis]), f"control_levels.{axis}: empty levels")
        values = [_number(v, f"control_levels.{axis}") for v in levels[axis]]
        _require(len(values) == len(set(values)), f"control_levels.{axis}: repeated levels")
    control_pool = _field(protocol, "control_pool", "protocol")
    _require(isinstance(control_pool, list), "control_pool: expected list")
    pool = [tuple(_settings(control, _COMMANDS, "control_pool").values()) for control in control_pool]
    expected_pool = set(product(*(levels[axis] for axis in _COMMANDS)))
    _require(len(pool) == len(set(pool)) and set(pool) == expected_pool and len(pool) == 54,
             "control_pool: must contain the complete 54 distinct command configurations")
    for group in groups:
        _mapping(group, "condition_group")
        condition = _text(_field(group, "condition_id", "condition_group"), "condition_id")
        _require(condition not in conditions, f"duplicate condition_id {condition}")
        split = group.get("split")
        _require(split in _SPLITS, f"{condition}: invalid declared split")
        nominal = _settings(_field(group, "nominal_settings", condition), _CONTEXT, condition)
        family = nominal["target"]
        _require(family in families, f"{condition}: undeclared family {family}")
        token = _canonical(nominal)
        _require(token not in nominal_owner, f"{condition}: repeated nominal context / partition leakage")
        nominal_owner[token] = condition
        controls = _field(group, "controls", condition)
        _require(isinstance(controls, list) and len(controls) == 8, f"{condition}: expected eight controls")
        normalized = [_settings(control, _COMMANDS, f"{condition}.control") for control in controls]
        command_vectors = [tuple(control.values()) for control in normalized]
        _require(len(set(command_vectors)) == 8, f"{condition}: controls are not eight distinct configurations")
        _require(set(command_vectors) <= expected_pool, f"{condition}: control outside declared levels")
        family_controls[family].update(command_vectors)
        conditions[condition] = (split, nominal, normalized)
        group_counts[split] += 1
        family_counts[family][split] += 1
        grids[family][tuple(nominal[key] for key in ("scale", "mass_kg", "sliding_friction"))] += 1
        for axis, (low, high) in bounds.items():
            value = nominal[axis]
            _require(low <= value < high, f"{condition}: nominal {axis} outside pose bounds")
            poses[(family, split)][axis].append(value)
    _require(type(protocol.get("condition_group_count")) is int
             and len(conditions) == protocol["condition_group_count"], "condition_group_count mismatch")
    _counts_equal(group_counts, _field(protocol, "group_split_counts", "protocol"), "group_split_counts")
    declared_family_counts = _mapping(_field(protocol, "family_group_split_counts", "protocol"),
                                      "family_group_split_counts")
    _require(set(declared_family_counts) == set(families), "family_group_split_counts: family mismatch")
    coverage, strata = {}, {}
    for family, specification in families.items():
        _mapping(specification, f"study_objects.{family}")
        level_lists = [protocol.get("scale_levels"), specification.get("mass_levels_kg"),
                       protocol.get("sliding_friction_levels")]
        for values in level_lists:
            _require(isinstance(values, list) and bool(values), f"{family}: missing property levels")
            checked = [_number(v, f"{family}.property_level") for v in values]
            _require(len(checked) == len(set(checked)), f"{family}: duplicate property levels")
        expected_grid = set(product(*level_lists))
        _require(len(expected_grid) == 48 and set(grids[family]) == expected_grid
                 and set(grids[family].values()) == {1}, f"{family}: planned property grid must cover all 48 combinations once")
        _counts_equal(family_counts[family], declared_family_counts[family], f"{family}.group_split_counts")
        _require(family_controls[family] == expected_pool, f"{family}: selected controls do not cover the full 54-configuration pool")
        coverage[family] = {"condition_count": sum(family_counts[family].values()),
                            "property_grid_combinations": len(grids[family]),
                            "group_split_counts": dict(family_counts[family]),
                            "distinct_control_configurations": len(family_controls[family])}
        strata[family] = {}
        for split in _SPLITS:
            axes = poses[(family, split)]
            _require(bool(axes), f"{family}/{split}: missing nominal pose strata")
            strata[family][split] = {}
            for axis, values in axes.items():
                low, high = bounds[axis]
                n = len(values)
                occupied = [math.floor((value - low) / (high - low) * n) for value in values]
                _require(sorted(occupied) == list(range(n)), f"{family}/{split}: nominal {axis} Latin-hypercube strata incomplete")
                strata[family][split][axis] = {"strata": n, "occupied_once": True,
                                              "minimum": min(values), "maximum": max(values)}

    condition_rows, condition_groups, group_conditions = defaultdict(list), {}, {}
    labels, splits, observable, physical = {}, {}, defaultdict(list), defaultdict(list)
    split_counts = Counter()
    training = []
    for trial_id, (planned_settings, planned_metadata) in planned.items():
        name = f"trial {trial_id}"
        settings = _settings(planned_settings, _FEATURES + _BIASES, f"{name}.planned settings")
        metadata = _mapping(planned_metadata, f"{name}.planned metadata")
        raw, csv = raw_by_id[trial_id], csv_by_id[trial_id]
        actual_settings = _settings(_field(raw, "settings", name), _FEATURES + _BIASES, f"{name}.raw settings")
        _require(actual_settings == settings, f"{name}: raw settings differ from planned spec")
        _mapping(_field(raw, "metadata", name), f"{name}.raw metadata")
        _require(_canonical(raw["metadata"]) == _canonical(metadata), f"{name}: raw metadata differ from planned spec")
        condition = _text(_field(metadata, "condition_id", name), f"{name}.condition_id")
        _require(condition in conditions, f"{name}: undeclared condition")
        split, nominal, controls = conditions[condition]
        _require(metadata.get("split") == split and csv.get("split") == split,
                 f"{name}: condition split differs from frozen protocol / partition leakage")
        _require(metadata.get("protocol_id") == protocol.get("protocol_id"), f"{name}: protocol_id mismatch")
        control_id = metadata.get("control_id")
        _require(type(control_id) is int and 0 <= control_id < 8, f"{name}: invalid control_id")
        _require({key: settings[key] for key in _CONTEXT} == nominal, f"{name}: nominal settings differ from condition")
        _require({key: settings[key] for key in _COMMANDS} == controls[control_id], f"{name}: control settings differ from condition")
        _require(all(abs(settings[key]) <= .004 for key in _BIASES), f"{name}: nuisance bias outside +/-4 mm")
        seed = metadata.get("nuisance_seed")
        _require(type(seed) is int and 0 <= seed < 2**32, f"{name}: invalid nuisance_seed")
        condition_rows[condition].append((trial_id, control_id, seed, tuple(settings[key] for key in _BIASES)))
        group_id = _text(_field(csv, "group_id", name), f"{name}.group_id")
        _require(condition not in condition_groups or condition_groups[condition] == group_id,
                 f"{name}: condition maps to multiple frozen groups")
        _require(group_id not in group_conditions or group_conditions[group_id] == condition,
                 f"{name}: frozen group maps to multiple conditions / partition leakage")
        condition_groups[condition], group_conditions[group_id] = group_id, condition
        tokens = sorted(["nominal:" + _canonical(nominal), "condition:" + condition])
        expected_group = "group_" + hashlib.sha256(_canonical(tokens).encode()).hexdigest()
        _require(group_id == expected_group, f"{name}: reconstructed label-independent group_id mismatch")
        feature_values = tuple(_text(_field(csv, key, name), f"{name}.{key}") if key == "target"
                               else _number(_field(csv, key, name), f"{name}.{key}", csv=True)
                               for key in _FEATURES)
        for key in ("target", "scale", "mass_kg", "sliding_friction") + _COMMANDS:
            _require(feature_values[_FEATURES.index(key)] == settings[key], f"{name}: CSV {key} differs from raw settings")
        _require(type(raw.get("success")) is bool, f"{name}: raw success must be boolean")
        _require(csv.get("success") in ("0", "1") and int(csv["success"]) == int(raw["success"]),
                 f"{name}: CSV label differs from raw success")
        labels[trial_id], splits[trial_id] = int(raw["success"]), split
        observable[feature_values].append(trial_id)
        fingerprint = _text(_field(raw, "initial_state_fingerprint", name), f"{name}.initial_state_fingerprint")
        physical[(_canonical(settings), fingerprint)].append(trial_id)
        split_counts[split] += 1
        if split == "train":
            failure_kind = _field(raw, "failure_kind", name)
            failure_reason = _field(raw, "failure_reason", name)
            _require(failure_kind is None or isinstance(failure_kind, str), f"{name}: invalid failure_kind")
            _require(failure_reason is None or isinstance(failure_reason, str), f"{name}: invalid failure_reason")
            training.append((trial_id, condition, feature_values, failure_kind, failure_reason))
    _require(set(condition_rows) == set(conditions), "planned trials omit declared condition groups")
    physical_summary = _duplicate_summary(physical, labels, splits)
    _require(physical_summary["duplicate_vector_groups"] == 0,
             "exact physical-input duplication (complete settings + initial_state_fingerprint): "
             f"{physical_summary}; possible partition leakage or conflicting duplicate labels")
    for condition, members in condition_rows.items():
        draws = defaultdict(dict)
        for trial_id, control_id, seed, biases in members:
            _require(control_id not in draws[seed], f"{condition}: repeated control in nuisance draw")
            draws[seed][control_id] = biases
        _require(len(members) == 16 and len(draws) == 2, f"{condition}: expected eight controls x two paired nuisance draws")
        for controls in draws.values():
            _require(set(controls) == set(range(8)) and len(set(controls.values())) == 1,
                     f"{condition}: nuisance draws are not paired with shared biases across controls")
        _require(len({next(iter(controls.values())) for controls in draws.values()}) == 2,
                 f"{condition}: nuisance draws have identical biases")
    _counts_equal(split_counts, _field(protocol, "planned_split_attempts", "protocol"), "planned_split_attempts")

    outcomes, failure_kinds, failure_reasons = defaultdict(Counter), Counter(), Counter()
    by_condition, by_level = defaultdict(Counter), defaultdict(lambda: defaultdict(Counter))
    numeric = {key: {0: [], 1: []} for key in _FEATURES[1:]}
    for trial_id, condition, features, failure_kind, failure_reason in training:
        label = labels[trial_id]
        outcomes[features[0]][str(label)] += 1
        by_condition[condition][str(label)] += 1
        for index, key in enumerate(_FEATURES):
            if key in ("target", "scale", "mass_kg", "sliding_friction") + _COMMANDS:
                by_level[key][str(features[index])][str(label)] += 1
            if key != "target":
                numeric[key][label].append(features[index])
        if not label:
            failure_kinds["null" if failure_kind is None else failure_kind] += 1
            failure_reasons["null" if failure_reason is None else failure_reason] += 1
    numeric_summary = {
        key: {str(label): {"count": len(values), "minimum": min(values) if values else None,
                           "maximum": max(values) if values else None,
                           "mean": math.fsum(values) / len(values) if values else None}
              for label, values in classes.items()} for key, classes in numeric.items()
    }
    return {
        "trial_count": len(planned), "condition_count": len(conditions),
        "group_count": len(group_conditions), "split_attempts": dict(split_counts),
        "group_split_counts": dict(group_counts), "families": coverage,
        "nominal_pose_strata": strata, "controls_per_condition": 8,
        "paired_draws_per_condition": 2, "group_ids_reconstructed": True,
        "observable_feature_duplicates": _duplicate_summary(observable, labels, splits),
        "exact_physical_input_duplicates": physical_summary,
        "duplicate_interpretation": "Observable 11-feature duplicates may disagree because paired latent execution biases are excluded; this alone is not corruption.",
        "training_only_diagnostics": {
            "scope": "train only; descriptive, not causal; no validation/test outcome-conditioned feature exploration",
            "row_count": len(training), "family_label_counts": {k: dict(v) for k, v in sorted(outcomes.items())},
            "condition_label_counts": {k: dict(v) for k, v in sorted(by_condition.items())},
            "feature_level_label_counts": {key: {value: dict(counts) for value, counts in sorted(values.items())}
                                          for key, values in by_level.items()},
            "numeric_features_by_label": numeric_summary,
            "failure_kind_counts": dict(failure_kinds), "failure_reason_counts": dict(failure_reasons),
            "failure_category_caveat": "Recorded heuristic categories/co-occurrences, not demonstrated causal root causes or necessarily robot-obstacle collisions.",
        },
    }
