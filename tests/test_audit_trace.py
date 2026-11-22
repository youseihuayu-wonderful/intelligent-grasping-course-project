"""Deterministic sampled-observer checks; no simulator or archived fixtures."""
import copy
import json
import math
import unittest

from audit_trace import audit_trace


def evidence(success=True):
    """Small 20ms trace with the real controller/observer threshold distinction."""
    samples = []
    transitions = {0: "idle", 1: "prepare", 2: "approach", 3: "descend",
                   4: "close", 5: "lift", 6: "hold", 56: "done"}
    phase = "idle"
    last_index = 106 if success else 5
    for index in range(last_index + 1):
        phase = transitions.get(index, phase)
        if not success and index == last_index:
            phase = "failed"
        status = {"idle": "idle", "done": "succeeded", "failed": "failed"}.get(phase, "running")
        physical = phase in ("hold", "done")
        samples.append({
            "time_s": (49 + index) / 50,
            "phase": phase,
            "status": status,
            "tcp_position_m": [0.5, -0.08, 1.0 if physical else 0.85],
            "target_position_m": [0.5, -0.08, 0.9 if physical else 0.75],
            "target_quaternion": [1.0, 0.0, 0.0, 0.0],
            "grasp_predicates": {
                "bilateral_grip": physical,
                "clear_of_table": physical,
                "target_lifted": physical,
                "relative_pose_stable": physical,
                "verified_hold": phase == "done",
            },
            "verified_hold_s": max(0, index - 6) / 50 if physical else 0.0,
        })
    samples.append(copy.deepcopy(samples[-1]))
    phases = list(dict.fromkeys(sample["phase"] for sample in samples))
    first = {phase: next(s["time_s"] for s in samples if s["phase"] == phase) for phase in phases}
    prepare = next(sample for sample in samples if sample["phase"] == "prepare")
    final = samples[-1]

    def pose(sample):
        return {"name": "synthetic_object", "position_m": sample["target_position_m"].copy(),
                "quaternion_wxyz": sample["target_quaternion"].copy()}

    row = {
        "trial_id": "synthetic", "trial_status": "completed", "success": success,
        "warnings": [], "verified_hold_s": final["verified_hold_s"],
        "execution_sim_s": final["time_s"] - prepare["time_s"],
        "initial_pose": pose(prepare), "final_pose": pose(final),
        "postconditions": copy.deepcopy(final["grasp_predicates"]),
        "phase_order": phases,
        "phase_first_sim_s": {phase: None if phase == "idle" else first[phase] - prepare["time_s"]
                              for phase in phases},
    }
    return row, {"trial_id": "synthetic", "samples": samples}


class AuditTraceTests(unittest.TestCase):
    def test_positive_terminal_duplicate_and_controller_threshold_are_valid(self):
        row, trace = evidence()
        result = audit_trace(row, trace)
        self.assertEqual(result["duplicate_terminal_samples"], 1)
        self.assertEqual(result["sample_count"], 108)
        self.assertEqual(result["final_controller_status"], "succeeded")
        self.assertEqual(result["final_observer_hold_s"], 2.0)
        self.assertTrue(result["positive_label_consistent"])
        self.assertEqual(result["sampled_hold_observations"]["final_window_sample_count"], 101)
        self.assertFalse(result["sampled_hold_observations"]["unsampled_continuity_independently_verified"])
        self.assertTrue(any(s["grasp_predicates"]["verified_hold"] and s["verified_hold_s"] < 2
                            for s in trace["samples"]))
        json.dumps(result, allow_nan=False)

    def test_negative_terminal_duplicate_is_valid(self):
        result = audit_trace(*evidence(False))
        self.assertFalse(result["observed_success"])
        self.assertTrue(result["positive_label_consistent"])
        self.assertEqual(result["duplicate_terminal_samples"], 1)
        self.assertEqual(result["sampled_hold_observations"]["final_window_sample_count"], 0)
        json.dumps(result, allow_nan=False)

    def test_flipped_raw_labels_are_rejected_in_both_directions(self):
        for success in (False, True):
            with self.subTest(success=success):
                row, trace = evidence(success)
                row["success"] = not success
                with self.assertRaisesRegex(ValueError, "row.success"):
                    audit_trace(row, trace)

    def test_contradictory_same_time_terminal_record_is_rejected(self):
        row, trace = evidence()
        trace["samples"][-1]["target_position_m"][0] += 0.001
        with self.assertRaisesRegex(ValueError, "contradictory same-time"):
            audit_trace(row, trace)

    def test_raw_terminal_pose_and_timer_corruption_are_rejected(self):
        for key in ("position_m", "quaternion_wxyz", "verified_hold_s"):
            with self.subTest(key=key):
                row, trace = evidence()
                if key == "verified_hold_s":
                    row[key] += 0.02
                elif key == "quaternion_wxyz":
                    row["final_pose"][key] = [math.cos(0.01), 0.0, 0.0, math.sin(0.01)]
                else:
                    row["final_pose"][key][0] += 0.001
                with self.assertRaisesRegex(ValueError, "row.(final_pose|verified_hold_s)"):
                    audit_trace(row, trace)

    def test_initial_pose_and_execution_duration_excluding_settling_are_checked(self):
        for key in ("initial_pose", "execution_sim_s"):
            with self.subTest(key=key):
                row, trace = evidence()
                if key == "initial_pose":
                    row[key]["position_m"][1] += 0.001
                else:
                    row[key] += 1.0
                with self.assertRaisesRegex(ValueError, key):
                    audit_trace(row, trace)

    def test_timer_requires_recorded_physical_eligibility(self):
        row, trace = evidence()
        trace["samples"][20]["grasp_predicates"]["bilateral_grip"] = False
        with self.assertRaisesRegex(ValueError, "positive observer timer while ineligible"):
            audit_trace(row, trace)

    def test_plausible_wrong_timer_speed_or_reset_is_rejected(self):
        for value in (0.29, 0.01):
            with self.subTest(value=value):
                row, trace = evidence()
                trace["samples"][20]["verified_hold_s"] = value
                with self.assertRaisesRegex(ValueError, "observer timer|timer evolution"):
                    audit_trace(row, trace)

    def test_fabricated_terminal_two_second_timer_is_rejected(self):
        row, trace = evidence()
        # Keep row, duplicate, and final pose consistent: only the history exposes it.
        for sample in trace["samples"][-2:]:
            sample["verified_hold_s"] = 2.2
        row["verified_hold_s"] = 2.2
        with self.assertRaisesRegex(ValueError, "observer timer"):
            audit_trace(row, trace)

    def test_positive_terminal_physical_and_controller_predicates_are_required(self):
        for key in ("target_lifted", "verified_hold"):
            with self.subTest(key=key):
                row, trace = evidence()
                for sample in trace["samples"][-2:]:
                    sample["grasp_predicates"][key] = False
                row["postconditions"][key] = False
                with self.assertRaisesRegex(ValueError, "ineligible|positive label"):
                    audit_trace(row, trace)

    def test_sampled_position_uses_a_necessary_bound_not_an_unsaved_reference(self):
        row, trace = evidence()
        trace["samples"][20]["target_quaternion"] = [0.0, 0.0, 0.0, 1.0]
        trace["samples"][20]["target_position_m"][0] += 0.011
        result = audit_trace(row, trace)
        self.assertAlmostEqual(
            result["sampled_hold_observations"]["final_window_max_relative_displacement_from_sampled_anchor_m"],
            0.011)
        trace["samples"][20]["target_position_m"][0] += 0.014
        with self.assertRaisesRegex(ValueError, "2cm necessary bound"):
            audit_trace(row, trace)

    def test_schema_nonfinite_and_time_corruption_are_rejected(self):
        changes = (
            ("time_s", float("nan"), "finite"),
            ("time_s", 0.0, "out of order"),
            ("target_quaternion", [2.0, 0.0, 0.0, 0.0], "unit norm"),
            ("target_position_m", [0.0, float("inf"), 0.0], "finite"),
            ("tcp_position_m", [0.0, 0.0], "3-element"),
            ("verified_hold_s", True, "finite number"),
        )
        for key, value, diagnosis in changes:
            with self.subTest(key=key, value=value):
                row, trace = evidence()
                trace["samples"][20][key] = value
                with self.assertRaisesRegex(ValueError, diagnosis):
                    audit_trace(row, trace)
        row, trace = evidence()
        trace["samples"][20]["grasp_predicates"]["verified_hold"] = 1
        with self.assertRaisesRegex(ValueError, "Boolean"):
            audit_trace(row, trace)

    def test_missing_and_nonterminal_evidence_are_rejected(self):
        row, trace = evidence()
        del trace["samples"][20]["target_position_m"]
        with self.assertRaisesRegex(ValueError, "missing target_position_m"):
            audit_trace(row, trace)
        row, trace = evidence()
        trace["samples"] = trace["samples"][:40]
        with self.assertRaisesRegex(ValueError, "not terminal"):
            audit_trace(row, trace)

    def test_lost_sample_and_raw_phase_metadata_are_rejected(self):
        row, trace = evidence()
        del trace["samples"][20]
        with self.assertRaisesRegex(ValueError, "sampling interval"):
            audit_trace(row, trace)
        row, trace = evidence()
        row["phase_first_sim_s"]["done"] += 0.02
        with self.assertRaisesRegex(ValueError, "phase_first_sim_s.done"):
            audit_trace(row, trace)


if __name__ == "__main__":
    unittest.main()
