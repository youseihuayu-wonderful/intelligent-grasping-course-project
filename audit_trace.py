"""Consistency checks for recorded samples, not an independent physics replay.

Contacts/clearance are recorded predicates, not independently reconstructed from
forces or full simulator state. Samples omit physics steps; timer consistency
cannot prove uninterrupted contact or stability between samples. Stability here
is relative position, not orientation. The controller's ``verified_hold`` is a
shorter internal predicate, distinct from the independent two-second observer.
"""
import math


_PHYSICAL = ("bilateral_grip", "clear_of_table", "target_lifted", "relative_pose_stable")
_PREDICATES = (*_PHYSICAL, "verified_hold")
_PHASES = ("idle", "prepare", "approach", "descend", "close", "lift", "hold", "done")
_TOL = 1e-8
_SAMPLE_INTERVAL_S = 0.02


def _mapping(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name}: expected an object")
    return value


def _required(mapping, key, name):
    if key not in mapping:
        raise ValueError(f"{name}: missing {key}")
    return mapping[key]


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}: expected a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name}: expected a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name}: expected a finite number")
    return result


def _vector(value, length, name):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name}: expected a {length}-element vector")
    values = [_number(component, name) for component in value]
    if length == 4 and not math.isclose(sum(x * x for x in values), 1.0, abs_tol=_TOL, rel_tol=0):
        raise ValueError(f"{name}: quaternion must have unit norm")
    return values


def _predicates(value, name):
    value = _mapping(value, name)
    for key in _PREDICATES:
        if type(_required(value, key, name)) is not bool:
            raise ValueError(f"{name}.{key}: expected a Boolean")
    return {key: value[key] for key in _PREDICATES}


def _near(actual, expected, name):
    if not math.isclose(actual, expected, abs_tol=_TOL, rel_tol=0):
        raise ValueError(f"{name}: values disagree ({actual!r} versus {expected!r})")


def _pose_match(pose, sample, name):
    pose = _mapping(pose, name)
    pose_name = _required(pose, "name", name)
    if not isinstance(pose_name, str) or not pose_name:
        raise ValueError(f"{name}.name: expected a nonempty object name")
    position = _vector(_required(pose, "position_m", name), 3, f"{name}.position_m")
    quaternion = _vector(_required(pose, "quaternion_wxyz", name), 4, f"{name}.quaternion_wxyz")
    for actual, expected in zip(sample["target_position_m"], position):
        _near(actual, expected, f"{name}.position_m")
    # The recorder copies the same simulator quaternion, including its sign.
    for actual, expected in zip(sample["target_quaternion"], quaternion):
        _near(actual, expected, f"{name}.quaternion_wxyz")


def _eligible(sample):
    return (sample["status"] in ("running", "succeeded")
            and sample["phase"] in ("hold", "done")
            and all(sample["grasp_predicates"][key] for key in _PHYSICAL))


def audit_trace(row: dict, trace: dict) -> dict:
    """Return JSON-safe per-trial evidence; reject missing/contradictory evidence."""
    row = _mapping(row, "row")
    trace = _mapping(trace, "trace")
    trial_id = _required(row, "trial_id", "row")
    if not isinstance(trial_id, str) or not trial_id:
        raise ValueError("row.trial_id: expected a nonempty string")
    if _required(trace, "trial_id", "trace") != trial_id:
        raise ValueError("trace.trial_id: disagrees with raw trial_id")
    success = _required(row, "success", "row")
    if type(success) is not bool:
        raise ValueError("row.success: expected a Boolean")
    if _required(row, "trial_status", "row") != "completed":
        raise ValueError("row.trial_status: expected completed")
    if _required(row, "warnings", "row") != []:
        raise ValueError("row.warnings: expected an empty list")
    samples = _required(trace, "samples", "trace")
    if not isinstance(samples, list) or not samples:
        raise ValueError("trace.samples: expected a nonempty list")

    terminal = _mapping(samples[-1], "trace.samples[-1]")
    terminal_time = _number(_required(terminal, "time_s", "trace.samples[-1]"),
                            "trace.samples[-1].time_s")

    unique = []
    duplicate_count = 0
    max_gap = 0.0
    phases = []
    phase_first = {}
    reset_intervals = 0
    sampled_relative_anchor = None
    relative_deviations = {}
    for index, sample in enumerate(samples):
        name = f"trace.samples[{index}]"
        sample = _mapping(sample, name)
        now = _number(_required(sample, "time_s", name), f"{name}.time_s")
        if now < 0:
            raise ValueError(f"{name}.time_s: must be nonnegative")
        phase = _required(sample, "phase", name)
        if not isinstance(phase, str) or phase not in (*_PHASES, "failed"):
            raise ValueError(f"{name}.phase: unknown grasp phase")
        status = _required(sample, "status", name)
        expected_status = {"idle": "idle", "done": "succeeded", "failed": "failed"}.get(phase, "running")
        if status != expected_status:
            raise ValueError(f"{name}.status: inconsistent with phase {phase}")
        for key, length in (("tcp_position_m", 3), ("target_position_m", 3), ("target_quaternion", 4)):
            _vector(_required(sample, key, name), length, f"{name}.{key}")
        predicates = _predicates(_required(sample, "grasp_predicates", name), f"{name}.grasp_predicates")
        hold = _number(_required(sample, "verified_hold_s", name), f"{name}.verified_hold_s")
        if hold < 0:
            raise ValueError(f"{name}.verified_hold_s: must be nonnegative")
        eligible = _eligible(sample)
        if hold > 0 and not eligible:
            raise ValueError(f"{name}.verified_hold_s: positive observer timer while ineligible")

        if unique:
            previous = unique[-1]
            gap = now - previous["time_s"]
            if gap < 0:
                raise ValueError(f"{name}.time_s: samples are out of order")
            if gap == 0:
                if sample != previous:
                    raise ValueError(f"{name}.time_s: contradictory same-time samples")
                if now != terminal_time or status not in ("succeeded", "failed"):
                    raise ValueError(f"{name}.time_s: duplicate is not terminal")
                duplicate_count += 1
                continue
            max_gap = max(max_gap, gap)
            if gap > _SAMPLE_INTERVAL_S + _TOL:
                raise ValueError(f"{name}.time_s: gap exceeds recorder's 0.020s sampling interval")
            previous_hold = previous["verified_hold_s"]
            if hold > previous_hold + gap + _TOL:
                raise ValueError(f"{name}.verified_hold_s: observer timer grows faster than elapsed time")
            if hold > gap + _TOL:
                if not _eligible(previous):
                    raise ValueError(f"{name}.verified_hold_s: timer spans an ineligible sample")
                _near(hold, previous_hold + gap, f"{name}.observer timer evolution")
            elif previous_hold > 0 and hold < previous_hold + gap - _TOL:
                reset_intervals += 1
            if previous["status"] == "failed":
                raise ValueError(f"{name}.status: samples continue after controller failure")
            if previous["status"] == "succeeded" and status not in ("succeeded", "failed"):
                raise ValueError(f"{name}.status: controller leaves terminal success")
        if phase not in phases:
            if phase != "failed" and phases and _PHASES.index(phase) <= _PHASES.index(phases[-1]):
                raise ValueError(f"{name}.phase: phase order reverses")
            phases.append(phase)
            phase_first[phase] = now
        elif unique and phase != unique[-1]["phase"]:
            raise ValueError(f"{name}.phase: previously completed phase reappears")
        if phase in ("hold", "done"):
            relative = [a - b for a, b in zip(sample["target_position_m"], sample["tcp_position_m"])]
            if not relative_deviations:
                _near(hold, 0.0, f"{name}.observer timer at forced hold entry")
                if phase != "hold":
                    raise ValueError(f"{name}.phase: done without a recorded hold entry")
            if predicates["relative_pose_stable"] and sampled_relative_anchor is None:
                sampled_relative_anchor = relative
            deviation = (math.dist(relative, sampled_relative_anchor)
                         if sampled_relative_anchor is not None else None)
            relative_deviations[now] = deviation
            # The controller captures its reference BEFORE the next 2ms physics
            # step; the forced hold sample is AFTER it. That reference is not
            # saved. Two points each <1cm from it must be <2cm apart, but a
            # displacement >1cm from the first saved sample is not a contradiction.
            if predicates["relative_pose_stable"] and deviation >= 0.02 + _TOL:
                raise ValueError(f"{name}.relative_pose_stable: sampled stable points exceed the 2cm necessary bound")
        unique.append(sample)

    if unique[0]["phase"] != "idle" or "prepare" not in phase_first:
        raise ValueError("trace: missing initial settling/prepare evidence")
    prepare = next(sample for sample in unique if sample["phase"] == "prepare")
    start = prepare["time_s"]
    final = unique[-1]
    if final["status"] not in ("succeeded", "failed"):
        raise ValueError("trace: final controller status is not terminal")
    final_hold = final["verified_hold_s"]
    observed_success = final["status"] == "succeeded" and final_hold >= 2.0
    if success != observed_success:
        raise ValueError("row.success: disagrees with final controller status and independent two-second observer")
    if final["status"] == "succeeded" and not observed_success:
        raise ValueError("trace: succeeded controller ends before independent two-second observation")
    if success and not all(final["grasp_predicates"][key] for key in _PREDICATES):
        raise ValueError("trace: positive label lacks final physical/controller hold predicates")
    postconditions = _predicates(_required(row, "postconditions", "row"), "row.postconditions")
    if postconditions != {key: final["grasp_predicates"][key] for key in _PREDICATES}:
        raise ValueError("row.postconditions: disagree with final sampled predicates")
    _near(_number(_required(row, "verified_hold_s", "row"), "row.verified_hold_s"), final_hold, "row.verified_hold_s")
    duration = _number(_required(row, "execution_sim_s", "row"), "row.execution_sim_s")
    _near(duration, final["time_s"] - start, "row.execution_sim_s (excluding settling)")
    _pose_match(_required(row, "initial_pose", "row"), prepare, "row.initial_pose")
    _pose_match(_required(row, "final_pose", "row"), final, "row.final_pose")
    if _required(row, "phase_order", "row") != phases:
        raise ValueError("row.phase_order: disagrees with sampled phase order")
    recorded_first = _mapping(_required(row, "phase_first_sim_s", "row"), "row.phase_first_sim_s")
    if set(recorded_first) != set(phases):
        raise ValueError("row.phase_first_sim_s: phase keys disagree with trace")
    for phase in phases:
        value = recorded_first[phase]
        if phase == "idle":
            if value is not None:
                raise ValueError("row.phase_first_sim_s.idle: settling must have null execution time")
        else:
            _near(_number(value, f"row.phase_first_sim_s.{phase}"), phase_first[phase] - start,
                  f"row.phase_first_sim_s.{phase}")

    window_start = final["time_s"] - final_hold if final_hold > 0 else None
    window = []
    if window_start is not None:
        if window_start < start - _TOL:
            raise ValueError("trace: final observer hold starts before execution")
        for sample in unique:
            if sample["time_s"] >= window_start - _TOL:
                if not _eligible(sample):
                    raise ValueError("trace: final observer hold window includes an ineligible sample")
                _near(sample["verified_hold_s"], max(0.0, sample["time_s"] - window_start),
                      "trace final sampled hold window timer")
                window.append(sample)
    window_gaps = [b["time_s"] - a["time_s"] for a, b in zip(window, window[1:])]
    return {
        "trial_id": trial_id,
        "sample_count": len(samples),
        "distinct_time_sample_count": len(unique),
        "duplicate_terminal_samples": duplicate_count,
        "max_sample_gap_s": max_gap,
        "final_controller_status": final["status"],
        "final_observer_hold_s": final_hold,
        "observed_success": observed_success,
        "positive_label_consistent": success == observed_success,
        "sampled_hold_observations": {
            "eligible_sample_count": sum(_eligible(sample) for sample in unique),
            "final_window_sample_count": len(window),
            "final_window_start_s": window_start,
            "final_window_max_sample_gap_s": max(window_gaps, default=0.0),
            "final_window_max_relative_displacement_from_sampled_anchor_m": max(
                (relative_deviations[sample["time_s"]] for sample in window), default=None),
            "relative_anchor_is_not_the_unsaved_controller_reference": True,
            "sampled_timer_reset_intervals": reset_intervals,
            "unsampled_continuity_independently_verified": False,
        },
        "limitations": [
            "Recorded contacts/clearance are not independently reconstructed from contact forces or full state.",
            "Unsampled physics steps cannot establish independent uninterrupted two-second retention.",
            "The controller stability reference precedes the first hold sample by a physics step and is not saved; sampled-anchor displacement is descriptive, with only a necessary 2cm pairwise bound checked.",
            "Relative stability concerns position, not full orientation.",
            "Controller verified_hold is distinct from the independent two-second observer timer.",
        ],
    }
