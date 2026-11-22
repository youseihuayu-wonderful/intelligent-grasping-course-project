"""Independent predicate and retention-clock audit of dense pilot telemetry."""
import math
from collections.abc import Iterable


_PHYSICAL = ("bilateral_grip", "clear_of_table", "target_lifted", "relative_pose_stable")
_PREDICATES = (*_PHYSICAL, "verified_hold")
_PHASES = ("idle", "prepare", "approach", "descend", "close", "lift", "hold", "done", "failed")
_COLUMNS = ["target_geom_id", "other_geom_id", "other_body_id", "efc_address", "normal_force_n"]
_DT = 0.002
_TOL = 1e-8
_SETTLE_STEPS = 500


def _required(obj, key, name):
    if key not in obj:
        raise ValueError(f"{name}: missing {key}")
    return obj[key]


def _mapping(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name}: expected an object")
    return value


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


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name}: expected an integer >= {minimum}")
    return value


def _vector(value, size, name):
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{name}: expected a {size}-element vector")
    result = [_number(v, name) for v in value]
    if size == 4 and not math.isclose(math.hypot(*result), 1.0, rel_tol=0, abs_tol=_TOL):
        raise ValueError(f"{name}: quaternion must have unit norm")
    return result


def _predicates(value, name):
    value = _mapping(value, name)
    if set(value) != set(_PREDICATES):
        raise ValueError(f"{name}: expected exactly five controller predicates")
    for key in _PREDICATES:
        if type(value[key]) is not bool:
            raise ValueError(f"{name}.{key}: expected a Boolean")
    return value


def _near(actual, expected, name):
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=_TOL):
        raise ValueError(f"{name}: values disagree ({actual!r} versus {expected!r})")


def _pose_match(value, sample, name):
    value = _mapping(value, name)
    if not isinstance(_required(value, "name", name), str) or not value["name"]:
        raise ValueError(f"{name}.name: expected a nonempty string")
    for key, size, sample_key in (("position_m", 3, "target_position_m"),
                                  ("quaternion_wxyz", 4, "target_quaternion_wxyz")):
        vector = _vector(_required(value, key, name), size, f"{name}.{key}")
        for actual, expected in zip(vector, sample[sample_key]):
            _near(actual, expected, f"{name}.{key}")


def audit_physics_steps(row: dict, records: Iterable[dict]) -> dict:
    """Reject incomplete/contradictory evidence; return JSON-safe audit results.

    Consumes the stream once, keeping only its last step and aggregate counters.
    Controller verified_hold is checked as recorded metadata, never used as the
    independent observer clock or as a replacement for reconstructed physics.
    """
    row = _mapping(row, "row")
    trial_id = _required(row, "trial_id", "row")
    if not isinstance(trial_id, str) or not trial_id:
        raise ValueError("row.trial_id: expected a nonempty string")
    if _required(row, "trial_status", "row") != "completed":
        raise ValueError("row.trial_status: expected completed")
    success = _required(row, "success", "row")
    if type(success) is not bool:
        raise ValueError("row.success: expected a Boolean")
    try:
        stream = iter(records)
        header = _mapping(next(stream), "header")
    except (TypeError, StopIteration) as error:
        raise ValueError("records: missing header/steps") from error
    for key, expected in (("record_type", "header"), ("trial_id", trial_id),
                          ("contact_row_columns", _COLUMNS)):
        if _required(header, key, "header") != expected:
            raise ValueError(f"header.{key}: disagrees with frozen schema/trial ID")
    if _integer(_required(header, "schema_version", "header"), "header.schema_version") != 1:
        raise ValueError("header.schema_version: expected 1")
    for key, expected in (("physics_step_s", _DT), ("finger_normal_force_threshold_n", 0.01),
                          ("minimum_lift_m", 0.10), ("relative_position_tolerance_m", 0.01)):
        if _number(_required(header, key, "header"), f"header.{key}") != expected:
            raise ValueError(f"header.{key}: disagrees with frozen physics rule")
    target = _integer(_required(header, "target_body_id", "header"), "header.target_body_id")
    table = _integer(_required(header, "table_body_id", "header"), "header.table_body_id")
    fingers = _required(header, "finger_body_ids", "header")
    if not isinstance(fingers, list) or len(fingers) != 2:
        raise ValueError("header.finger_body_ids: expected two distinct body IDs")
    fingers = {_integer(body, "header.finger_body_ids") for body in fingers}
    if len(fingers) != 2 or len({target, table, *fingers}) != 4:
        raise ValueError("header: target, table and finger body IDs must be distinct")

    count = contacts_count = active_count = inactive_count = eligible_count = resets = 0
    table_steps = 0
    true_counts = dict.fromkeys(_PHYSICAL, 0)
    geom_bodies = {}
    phases = []
    first_times = {}
    previous = initial = None
    hold_started = None
    hold_s = 0.0
    window_count = 0
    min_lift = max_displacement = None
    max_gap = 0.0
    for count, record in enumerate(stream, 1):
        name = f"steps[{count}]"
        record = _mapping(record, name)
        if _required(record, "record_type", name) != "step":
            raise ValueError(f"{name}.record_type: expected step")
        # Steps inherit the header ID. If a producer supplies an explicit ID,
        # it must not silently identify a different trial.
        if "trial_id" in record and record["trial_id"] != trial_id:
            raise ValueError(f"{name}.trial_id: disagrees with header")
        if _integer(_required(record, "step_index", name), f"{name}.step_index", 1) != count:
            raise ValueError(f"{name}.step_index: dropped, duplicated or reordered physics tick")
        now = _number(_required(record, "time_s", name), f"{name}.time_s")
        _near(now, count * _DT, f"{name}.time_s: contiguous 2ms grid")
        if previous is not None:
            gap = now - previous["time_s"]
            _near(gap, _DT, f"{name}.time_s: contiguous 2ms interval")
            max_gap = max(max_gap, gap)
            if previous["status"] == "failed" or (previous["status"] == "succeeded" and hold_s >= 2.0):
                raise ValueError(f"{name}: physics continues after observer termination")
        phase = _required(record, "phase", name)
        if not isinstance(phase, str) or phase not in _PHASES:
            raise ValueError(f"{name}.phase: unknown phase")
        status = _required(record, "status", name)
        expected_status = {"idle": "idle", "done": "succeeded", "failed": "failed"}.get(phase, "running")
        if status != expected_status:
            raise ValueError(f"{name}.status: inconsistent with phase")
        if count <= _SETTLE_STEPS and phase != "idle":
            raise ValueError(f"{name}.phase: non-idle during one-second settling")
        if count > _SETTLE_STEPS and phase == "idle":
            raise ValueError(f"{name}.phase: idle after execution started")
        if count == _SETTLE_STEPS + 1 and phase != "prepare":
            raise ValueError(f"{name}.phase: missing prepare after settling")
        if previous is not None and previous["status"] == "succeeded" and status not in ("succeeded", "failed"):
            raise ValueError(f"{name}.status: controller leaves terminal success")
        if not phases or phase != phases[-1]:
            if phase in phases or (phases and _PHASES.index(phase) <= _PHASES.index(phases[-1])):
                raise ValueError(f"{name}.phase: phase order reverses")
            phases.append(phase)
            first_times[phase] = now
        position = _vector(_required(record, "target_position_m", name), 3, f"{name}.target_position_m")
        tcp = _vector(_required(record, "tcp_position_m", name), 3, f"{name}.tcp_position_m")
        _vector(_required(record, "target_quaternion_wxyz", name), 4, f"{name}.target_quaternion_wxyz")
        relative = [_number(a - b, f"{name}.target_relative_m") for a, b in zip(position, tcp)]
        reference = _required(record, "reference_z_m", name)
        if reference is not None:
            reference = _number(reference, f"{name}.reference_z_m")
        anchor = _required(record, "hold_relative_m", name)
        if anchor is not None:
            anchor = _vector(anchor, 3, f"{name}.hold_relative_m")
        for key, value in (("reference_z_m", reference), ("hold_relative_m", anchor)):
            old = previous[key] if previous is not None else None
            if old is not None and value != old:
                raise ValueError(f"{name}.{key}: saved reference changed or disappeared")
            if old is None and value is not None:
                capture_phase = "prepare" if key == "reference_z_m" else "lift"
                if previous is None or previous["phase"] != capture_phase or phase == capture_phase:
                    raise ValueError(f"{name}.{key}: reference captured outside phase transition")
                expected = (previous["target_position_m"][2] if key == "reference_z_m" else
                            [a - b for a, b in zip(previous["target_position_m"], previous["tcp_position_m"])])
                if value != expected:
                    raise ValueError(f"{name}.{key}: disagrees with exact pre-step capture pose")
        if phase in ("approach", "descend", "close", "lift", "hold", "done") and reference is None:
            raise ValueError(f"{name}.reference_z_m: missing controller reference")
        if phase in ("hold", "done") and anchor is None:
            raise ValueError(f"{name}.hold_relative_m: missing controller reference")
        lift = (_number(position[2] - reference, f"{name}.lift_m") if reference is not None else None)
        displacement = (_number(math.dist(relative, anchor), f"{name}.relative_displacement_m")
                        if anchor is not None else None)
        contacts = _required(record, "target_contacts", name)
        if not isinstance(contacts, list):
            raise ValueError(f"{name}.target_contacts: expected a list")
        touching = set()
        table_support = False
        for index, contact in enumerate(contacts):
            contact_name = f"{name}.target_contacts[{index}]"
            if not isinstance(contact, list) or len(contact) != 5:
                raise ValueError(f"{contact_name}: expected five columns")
            target_geom, other_geom, body = [_integer(value, contact_name) for value in contact[:3]]
            address = _integer(contact[3], f"{contact_name}.efc_address", -1)
            if target_geom == other_geom or body == target:
                raise ValueError(f"{contact_name}: impossible target/other IDs")
            for geom, owner in ((target_geom, target), (other_geom, body)):
                if geom in geom_bodies and geom_bodies[geom] != owner:
                    raise ValueError(f"{contact_name}: geometry/body ID mapping changed")
                geom_bodies[geom] = owner
            force = contact[4]
            if force is not None:
                force = _number(force, f"{contact_name}.normal_force_n")
            if address < 0 and force is not None:
                raise ValueError(f"{contact_name}: inactive contact force must be null")
            if address >= 0 and body in fingers and force is None:
                raise ValueError(f"{contact_name}: active finger requires a measured normal force")
            contacts_count += 1
            if address >= 0:
                active_count += 1
            else:
                inactive_count += 1
            table_support |= body == table
            if body in fingers and address >= 0 and force > 0.01:
                touching.add(body)
        physical = {
            "bilateral_grip": touching == fingers,
            "clear_of_table": not table_support,
            "target_lifted": lift is not None and lift >= 0.10,
            "relative_pose_stable": displacement is not None and displacement < 0.01,
        }
        predicates = _predicates(_required(record, "controller_predicates", name), f"{name}.controller_predicates")
        for key in _PHYSICAL:
            if predicates[key] != physical[key]:
                raise ValueError(f"{name}.{key}: controller label disagrees with reconstructed physics")
            true_counts[key] += int(physical[key])
        if status == "succeeded" and not all(predicates.values()):
            raise ValueError(f"{name}: succeeded controller lacks physical/controller postconditions")
        if _required(record, "read_only_state_unchanged", name) is not True:
            raise ValueError(f"{name}.read_only_state_unchanged: mutation guard did not pass")
        table_steps += int(table_support)
        eligible = status in ("running", "succeeded") and phase in ("hold", "done") and all(physical.values())
        if eligible:
            eligible_count += 1
            if hold_started is None:
                hold_started = now
                window_count = 0
                min_lift = max_displacement = None
            hold_s = now - hold_started
            window_count += 1
            min_lift = lift if min_lift is None else min(min_lift, lift)
            max_displacement = displacement if max_displacement is None else max(max_displacement, displacement)
        else:
            resets += int(hold_started is not None)
            hold_started = None
            hold_s = 0.0
            window_count = 0
            min_lift = max_displacement = None
        if count == _SETTLE_STEPS:
            initial = record
        previous = record

    if initial is None or count <= _SETTLE_STEPS or previous is None:
        raise ValueError("records: missing settling/execution steps")
    final = previous
    if final["status"] not in ("succeeded", "failed"):
        raise ValueError("records: final controller status is not terminal")
    observed_success = final["status"] == "succeeded" and hold_s >= 2.0
    if final["status"] == "succeeded" and not observed_success:
        raise ValueError("records: controller success is shorter than independent two-second retention")
    if success != observed_success:
        raise ValueError("row.success: disagrees with independent two-second observer")
    _near(_number(_required(row, "verified_hold_s", "row"), "row.verified_hold_s"), hold_s, "row.verified_hold_s")
    duration = _number(_required(row, "execution_sim_s", "row"), "row.execution_sim_s")
    _near(duration, final["time_s"] - initial["time_s"], "row.execution_sim_s (excluding one-second settling)")
    _pose_match(_required(row, "initial_pose", "row"), initial, "row.initial_pose")
    _pose_match(_required(row, "final_pose", "row"), final, "row.final_pose")
    if row["initial_pose"]["name"] != row["final_pose"]["name"]:
        raise ValueError("row: initial/final pose object names disagree")
    postconditions = _predicates(_required(row, "postconditions", "row"), "row.postconditions")
    if postconditions != final["controller_predicates"]:
        raise ValueError("row.postconditions: disagree with final reconstructed/recorded predicates")
    if _required(row, "phase_order", "row") != phases:
        raise ValueError("row.phase_order: disagrees with full-step phase order")
    phase_first = _mapping(_required(row, "phase_first_sim_s", "row"), "row.phase_first_sim_s")
    if set(phase_first) != set(phases):
        raise ValueError("row.phase_first_sim_s: phase keys disagree")
    for phase in phases:
        if phase == "idle":
            if phase_first[phase] is not None:
                raise ValueError("row.phase_first_sim_s.idle: settling time must be null")
        else:
            # start_task enters prepare at the last settling timestamp, without
            # a physics tick; the first dense prepare record is one tick later.
            expected = 0.0 if phase == "prepare" else first_times[phase] - initial["time_s"]
            _near(_number(phase_first[phase], f"row.phase_first_sim_s.{phase}"), expected,
                  f"row.phase_first_sim_s.{phase}")
    return {
        "trial_id": trial_id,
        "schema_version": 1,
        "physics_step_s": _DT,
        "step_count": count,
        "settling_step_count": _SETTLE_STEPS,
        "execution_step_count": count - _SETTLE_STEPS,
        "contact_record_count": contacts_count,
        "active_contact_count": active_count,
        "inactive_contact_count": inactive_count,
        "table_support_step_count": table_steps,
        "physical_predicate_true_step_counts": true_counts,
        "read_only_checks_passed": count,
        "max_step_gap_s": max_gap,
        "final_controller_status": final["status"],
        "final_observer_hold_s": hold_s,
        "observed_success": observed_success,
        "full_step_label_consistent": True,
        "positive_label_consistent": success == observed_success,
        "hold_observations": {
            "eligible_step_count": eligible_count,
            "timer_reset_count": resets,
            "final_window_step_count": window_count,
            "final_window_start_s": hold_started,
            "minimum_lift_m": min_lift,
            "maximum_relative_displacement_from_saved_reference_m": max_displacement,
            "all_physics_ticks_present": True,
        },
        "limitations": [
            "Independent reconstruction uses recorded MuJoCo contacts/forces and controller-supplied references, not independent sensors or a full dynamics replay.",
            "Read-only guards attest recorded qpos/qvel/ctrl/time checks, not every possible simulator field.",
            "Relative stability concerns position, not orientation; this is not sim-to-real proof.",
            "Controller verified_hold is recorded metadata, not the independently reconstructed two-second timer.",
        ],
    }
