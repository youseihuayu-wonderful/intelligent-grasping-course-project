"""Deterministic dense contact/reference evidence; no simulator or mocks."""
import copy
import json
import math
import unittest

from pilot_evidence_audit import audit_physics_steps


def evidence(success=True):
    header = {
        "record_type": "header", "schema_version": 1, "trial_id": "dense-trial",
        "physics_step_s": 0.002, "target_body_id": 1, "finger_body_ids": [2, 3],
        "table_body_id": 4,
        "contact_row_columns": ["target_geom_id", "other_geom_id", "other_body_id",
                                "efc_address", "normal_force_n"],
        "finger_normal_force_threshold_n": 0.01, "minimum_lift_m": 0.10,
        "relative_position_tolerance_m": 0.01,
    }
    records = [header]
    transitions = {1: "idle", 501: "prepare", 502: "approach", 503: "descend",
                   504: "close", 505: "lift", 506: "hold", 1006: "done"}
    phase = "idle"
    index = 0
    while True:
        index += 1
        phase = transitions.get(index, phase)
        if not success and index == 505:
            phase = "failed"
        elevated = index >= 505
        holding = phase in ("hold", "done")
        position = [0.5005 if holding else 0.5, -0.08, 0.90 if elevated else 0.75]
        tcp = [0.5, -0.08, 1.0 if elevated else 0.85]
        reference = 0.75 if index >= 502 else None
        anchor = [0.0, 0.0, 0.90 - 1.0] if holding else None
        contacts = [] if elevated else [[10, 13, 4, -1, None]]
        if index >= 504:
            contacts.extend([[10, 11, 2, 0, 0.3], [10, 12, 3, 3, 0.4],
                             [10, 14, 2, -1, None]])
        predicates = {
            "bilateral_grip": index >= 504,
            "clear_of_table": elevated,
            "target_lifted": elevated,
            "relative_pose_stable": holding,
            "verified_hold": phase == "done",
        }
        records.append({
            "record_type": "step", "step_index": index, "time_s": index * 0.002,
            "phase": phase,
            "status": {"idle": "idle", "done": "succeeded", "failed": "failed"}.get(phase, "running"),
            "target_position_m": position, "target_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "tcp_position_m": tcp, "reference_z_m": reference, "hold_relative_m": anchor,
            "target_contacts": contacts, "controller_predicates": predicates,
            "read_only_state_unchanged": True,
        })
        if phase == "failed" or (phase == "done" and records[-1]["time_s"] - records[506]["time_s"] >= 2.0):
            break
    final = records[-1]
    phases = list(dict.fromkeys(record["phase"] for record in records[1:]))
    first = {phase: next(record["time_s"] for record in records[1:] if record["phase"] == phase)
             for phase in phases}

    def pose(record):
        return {"name": "synthetic_object", "position_m": record["target_position_m"].copy(),
                "quaternion_wxyz": record["target_quaternion_wxyz"].copy()}

    row = {
        "trial_id": "dense-trial", "trial_status": "completed", "success": success,
        "verified_hold_s": final["time_s"] - records[506]["time_s"] if success else 0.0,
        "execution_sim_s": final["time_s"] - records[500]["time_s"],
        "initial_pose": pose(records[500]), "final_pose": pose(final),
        "postconditions": copy.deepcopy(final["controller_predicates"]),
        "phase_order": phases,
        "phase_first_sim_s": {phase: None if phase == "idle" else
                              0.0 if phase == "prepare" else first[phase] - records[500]["time_s"]
                              for phase in phases},
    }
    return row, records


class AuditPhysicsStepsTests(unittest.TestCase):
    def test_streaming_success_reconstructs_every_tick_and_saved_anchor(self):
        row, records = evidence()
        result = audit_physics_steps(row, (record for record in records))
        self.assertEqual(result["step_count"], len(records) - 1)
        self.assertEqual(result["read_only_checks_passed"], len(records) - 1)
        self.assertEqual(result["settling_step_count"], 500)
        self.assertTrue(result["observed_success"])
        self.assertTrue(result["full_step_label_consistent"])
        self.assertGreaterEqual(result["final_observer_hold_s"], 2.0)
        window = result["hold_observations"]
        self.assertEqual(window["final_window_step_count"], len(records) - 506)
        self.assertAlmostEqual(window["minimum_lift_m"], 0.15)
        self.assertAlmostEqual(window["maximum_relative_displacement_from_saved_reference_m"], 0.0005)
        self.assertGreater(result["inactive_contact_count"], 0)
        self.assertTrue(records[1006]["controller_predicates"]["verified_hold"])
        self.assertLess(records[1006]["time_s"] - records[506]["time_s"], 2.0)
        json.dumps(result, allow_nan=False)

    def test_completed_failure_is_a_valid_negative_label(self):
        result = audit_physics_steps(*evidence(False))
        self.assertFalse(result["observed_success"])
        self.assertTrue(result["full_step_label_consistent"])
        self.assertEqual(result["final_observer_hold_s"], 0.0)
        self.assertIsNone(result["hold_observations"]["minimum_lift_m"])
        self.assertIsNone(result["hold_observations"]["maximum_relative_displacement_from_saved_reference_m"])
        json.dumps(result, allow_nan=False)

    def test_dropped_duplicated_reordered_or_retimed_tick_is_rejected(self):
        for change in ("drop", "duplicate", "swap", "time", "index", "early_termination"):
            with self.subTest(change=change):
                row, records = evidence()
                if change == "drop":
                    del records[900]
                elif change == "duplicate":
                    records.insert(900, copy.deepcopy(records[900]))
                elif change == "swap":
                    records[900], records[901] = records[901], records[900]
                elif change == "time":
                    records[900]["time_s"] += 0.0001
                elif change == "index":
                    records[900]["step_index"] += 1
                else:
                    records.pop()
                with self.assertRaises(ValueError):
                    audit_physics_steps(row, records)

    def test_each_finger_requires_an_active_force_strictly_over_threshold(self):
        for value in (None, 0.0, 0.01, -0.1):
            with self.subTest(force=value):
                row, records = evidence()
                records[900]["target_contacts"][1][4] = value
                with self.assertRaises(ValueError):
                    audit_physics_steps(row, records)
        row, records = evidence()
        records[900]["target_contacts"][1][3:] = [-1, None]
        with self.assertRaisesRegex(ValueError, "bilateral_grip"):
            audit_physics_steps(row, records)
        row, records = evidence()
        records[900]["target_contacts"][1][2] = 2
        with self.assertRaises(ValueError):
            audit_physics_steps(row, records)
        row, records = evidence()
        records[900]["target_contacts"][1][4] = 0.010000001
        result = audit_physics_steps(row, records)
        self.assertTrue(result["observed_success"])

    def test_table_support_includes_zero_force_and_inactive_contacts(self):
        for address, force in ((-1, None), (0, 0.0), (0, None)):
            with self.subTest(address=address, force=force):
                row, records = evidence()
                records[900]["target_contacts"].append([10, 13, 4, address, force])
                with self.assertRaisesRegex(ValueError, "clear_of_table"):
                    audit_physics_steps(row, records)
                # With the raw label corrected, even an inactive table contact
                # really is support. This pre-hold close frame does not retain.
                row, records = evidence()
                records[504]["target_contacts"][0] = [10, 13, 4, address, force]
                audit_physics_steps(row, records)
        row, records = evidence()
        records[504]["target_contacts"].pop(0)
        with self.assertRaisesRegex(ValueError, "clear_of_table"):
            audit_physics_steps(row, records)

    def test_reference_corruption_is_rejected_even_without_boolean_flip(self):
        for key in ("reference_z_m", "hold_relative_m"):
            for change in ("one_tick", "all_ticks", "missing", "capture"):
                with self.subTest(key=key, change=change):
                    row, records = evidence()
                    start = 502 if key == "reference_z_m" else 506
                    affected = records[start:] if change == "all_ticks" else [records[start if change == "capture" else 900]]
                    for record in affected:
                        if change == "missing":
                            record[key] = None
                        elif key == "reference_z_m":
                            record[key] += 1e-6
                        else:
                            record[key][0] += 1e-6
                    with self.assertRaises(ValueError):
                        audit_physics_steps(row, records)

    def test_positions_and_exact_stability_boundary_are_reconstructed(self):
        for key, component, delta in (("target_position_m", 2, -0.1),
                                      ("target_position_m", 0, 0.02),
                                      ("tcp_position_m", 0, 0.02)):
            with self.subTest(key=key, component=component):
                row, records = evidence()
                records[900][key][component] += delta
                with self.assertRaises(ValueError):
                    audit_physics_steps(row, records)
        row, records = evidence()
        # Exact 1cm is not stable: use a zero reference and zero TCP x so
        # subtraction introduces no rounding at the strict comparison boundary.
        for record in records[505:]:
            record["target_position_m"][0] = 0.0
            record["tcp_position_m"][0] = 0.0
        records[900]["target_position_m"][0] = 0.01
        with self.assertRaisesRegex(ValueError, "relative_pose_stable"):
            audit_physics_steps(row, records)

    def test_raw_label_timer_postconditions_pose_and_duration_corruptions(self):
        for change in ("label", "timer", "postconditions", "initial_pose", "final_pose",
                       "quaternion", "duration", "settling_duration", "phase_order", "phase_time"):
            with self.subTest(change=change):
                row, records = evidence()
                if change == "label":
                    row["success"] = False
                elif change == "timer":
                    row["verified_hold_s"] = 1.0
                elif change == "postconditions":
                    row["postconditions"]["clear_of_table"] = False
                elif change in ("initial_pose", "final_pose"):
                    row[change]["position_m"][0] += 0.01
                elif change == "quaternion":
                    row["final_pose"]["quaternion_wxyz"] = [-1.0, 0.0, 0.0, 0.0]
                elif change == "duration":
                    row["execution_sim_s"] -= 0.002
                elif change == "settling_duration":
                    row["execution_sim_s"] += 1.0
                elif change == "phase_order":
                    row["phase_order"].remove("close")
                else:
                    row["phase_first_sim_s"]["prepare"] = 0.002
                with self.assertRaises(ValueError):
                    audit_physics_steps(row, records)
        row, records = evidence(False)
        row["success"] = True
        with self.assertRaises(ValueError):
            audit_physics_steps(row, records)

    def test_internal_controller_hold_cannot_substitute_for_two_seconds(self):
        row, records = evidence()
        records = records[:1007]
        row["verified_hold_s"] = 2.0
        row["execution_sim_s"] = records[-1]["time_s"] - records[500]["time_s"]
        with self.assertRaisesRegex(ValueError, "two-second"):
            audit_physics_steps(row, records)

    def test_physical_interrupt_resets_independent_timer_despite_corrected_labels(self):
        row, records = evidence()
        record = records[900]
        record["target_contacts"][1][4] = 0.0
        record["controller_predicates"]["bilateral_grip"] = False
        with self.assertRaisesRegex(ValueError, "two-second"):
            audit_physics_steps(row, records)

    def test_failed_terminal_after_hold_resets_timer_and_window(self):
        row, records = evidence()
        records = records[:1000]
        final = records[-1]
        final["phase"] = "failed"
        final["status"] = "failed"
        final["target_contacts"][1][4] = 0.0
        final["controller_predicates"]["bilateral_grip"] = False
        row.update(success=False, verified_hold_s=0.0,
                   execution_sim_s=final["time_s"] - records[500]["time_s"],
                   postconditions=copy.deepcopy(final["controller_predicates"]))
        row["phase_order"][-1] = "failed"
        row["phase_first_sim_s"].pop("done")
        row["phase_first_sim_s"]["failed"] = final["time_s"] - records[500]["time_s"]
        result = audit_physics_steps(row, records)
        self.assertEqual(result["hold_observations"]["timer_reset_count"], 1)
        self.assertEqual(result["hold_observations"]["final_window_step_count"], 0)
        self.assertIsNone(result["hold_observations"]["minimum_lift_m"])

    def test_every_physical_boolean_and_read_only_guard_is_checked(self):
        for key in ("bilateral_grip", "clear_of_table", "target_lifted", "relative_pose_stable"):
            with self.subTest(key=key):
                row, records = evidence()
                records[900]["controller_predicates"][key] = False
                with self.assertRaisesRegex(ValueError, key):
                    audit_physics_steps(row, records)
        row, records = evidence()
        records[900]["read_only_state_unchanged"] = False
        with self.assertRaisesRegex(ValueError, "mutation guard"):
            audit_physics_steps(row, records)
        row, records = evidence()
        records[-1]["controller_predicates"]["verified_hold"] = False
        row["postconditions"]["verified_hold"] = False
        with self.assertRaises(ValueError):
            audit_physics_steps(row, records)

    def test_nonfinite_numeric_data_are_rejected(self):
        for value in (math.nan, math.inf, -math.inf):
            for field in ("time_s", "reference_z_m", "hold_relative_m", "target_position_m",
                          "tcp_position_m", "target_quaternion_wxyz", "force", "raw_duration"):
                with self.subTest(value=value, field=field):
                    row, records = evidence()
                    if field == "force":
                        records[900]["target_contacts"][0][4] = value
                    elif field == "raw_duration":
                        row["execution_sim_s"] = value
                    elif isinstance(records[900][field], list):
                        records[900][field][0] = value
                    else:
                        records[900][field] = value
                    with self.assertRaises(ValueError):
                        audit_physics_steps(row, records)

    def test_wrong_trial_body_geometry_and_schema_ids_are_rejected(self):
        for change in ("header_trial", "raw_trial", "step_trial", "finger_overlap", "target_overlap",
                       "table", "finger", "geom", "negative_geom", "fractional_body", "boolean_id",
                       "schema", "dt", "columns", "threshold"):
            with self.subTest(change=change):
                row, records = evidence()
                if change == "header_trial":
                    records[0]["trial_id"] = "another-trial"
                elif change == "raw_trial":
                    row["trial_id"] = "another-trial"
                elif change == "step_trial":
                    records[900]["trial_id"] = "another-trial"
                elif change == "finger_overlap":
                    records[0]["finger_body_ids"] = [2, 2]
                elif change == "target_overlap":
                    records[0]["target_body_id"] = 2
                elif change == "table":
                    records[0]["table_body_id"] = 9
                elif change == "finger":
                    records[0]["finger_body_ids"] = [2, 9]
                elif change == "geom":
                    records[900]["target_contacts"][0][1] = 10
                elif change == "negative_geom":
                    records[900]["target_contacts"][0][0] = -1
                elif change == "fractional_body":
                    records[900]["target_contacts"][0][2] = 2.5
                elif change == "boolean_id":
                    records[0]["target_body_id"] = True
                elif change == "schema":
                    records[0]["schema_version"] = True
                elif change == "dt":
                    records[0]["physics_step_s"] = 0.02
                elif change == "columns":
                    records[0]["contact_row_columns"].reverse()
                else:
                    records[0]["finger_normal_force_threshold_n"] = 0.0
                with self.assertRaises(ValueError):
                    audit_physics_steps(row, records)

    def test_missing_fields_shapes_unit_quaternions_and_native_booleans(self):
        for change in ("missing_header", "missing_steps", "missing_field", "position_shape",
                       "quaternion_norm", "predicate_type", "contact_shape", "inactive_force", "guard_type"):
            with self.subTest(change=change):
                row, records = evidence()
                if change == "missing_header":
                    records.pop(0)
                elif change == "missing_steps":
                    records = records[:1]
                elif change == "missing_field":
                    del records[900]["hold_relative_m"]
                elif change == "position_shape":
                    records[900]["target_position_m"].pop()
                elif change == "quaternion_norm":
                    records[900]["target_quaternion_wxyz"] = [2.0, 0.0, 0.0, 0.0]
                elif change == "predicate_type":
                    records[900]["controller_predicates"]["bilateral_grip"] = 1
                elif change == "contact_shape":
                    records[900]["target_contacts"][0].pop()
                elif change == "inactive_force":
                    records[900]["target_contacts"][-1][-1] = 0.0
                else:
                    records[900]["read_only_state_unchanged"] = 1
                with self.assertRaises(ValueError):
                    audit_physics_steps(row, records)


if __name__ == "__main__":
    unittest.main()
