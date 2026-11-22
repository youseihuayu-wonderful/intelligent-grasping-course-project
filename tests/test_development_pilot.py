"""Pilot-only execution and export protections, without additional physical trials."""
import copy
import math
from pathlib import Path
import tempfile
import unittest

from audit_development_pilot import check_variant, feature_values, make_archive
from design_independent_test import ROOT, decode_json, payload
from run_development_pilot import load_pilot, sha_file, validate_output
from test_data_quality import inputs


class DevelopmentPilotTests(unittest.TestCase):
    def test_final_role_cannot_be_executed_as_the_frozen_pilot(self):
        public = ROOT/"protocols/independent_test_v1"
        plan = decode_json((public/"pilot_plan.json").read_bytes())
        plan["role"] = "final"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path/"pilot_plan.json").write_bytes(payload(plan))
            (path/"policy.json").write_bytes((public/"policy.json").read_bytes())
            with self.assertRaises(ValueError):
                load_pilot(path, path)

    def test_export_uses_only_settled_pregrasp_pose(self):
        _, row, _ = inputs()
        row["final_pose"] = {"position_m": [9., 9., 9.]}
        before = feature_values(row)
        self.assertEqual(before["x_m"], .5)
        self.assertAlmostEqual(before["yaw_rad"], .2)
        row["final_pose"]["position_m"] = [-9., -9., -9.]
        self.assertEqual(feature_values(row), before)
        row["initial_pose"]["quaternion_wxyz"] = [0., 0., 0., 0.]
        with self.assertRaises(ValueError):
            feature_values(row)

    def test_configured_mass_and_friction_mismatch_is_rejected(self):
        for field, value in (("actual_mass_kg", .9), ("finger_collision_sliding_friction", [.2])):
            with self.subTest(field=field):
                _, row, protocol = inputs()
                row["variant_features"][field] = value
                with self.assertRaises(ValueError):
                    check_variant(row, protocol)

    def test_pilot_output_cannot_overwrite_or_enter_git_or_final_custody(self):
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory)/"existing"
            existing.write_bytes(b"keep")
            with self.assertRaises(ValueError):
                validate_output(existing, ROOT)
            self.assertEqual(existing.read_bytes(), b"keep")
        for destination in (ROOT/"never-create-evidence",
                            ROOT.parent/"intelligent-grasping-private-evaluation"/"never-create-pilot"):
            with self.subTest(destination=destination):
                with self.assertRaises(ValueError):
                    validate_output(destination, ROOT)
                self.assertFalse(destination.exists())

    def test_lossless_archive_detects_changed_member_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root/"evidence"
            evidence.mkdir()
            member = evidence/"trial.json"
            member.write_bytes(b'{"success":true}')
            inventory = {"trial.json": {"bytes": member.stat().st_size, "sha256": sha_file(member)}}
            member.write_bytes(b'{"success":false}')
            with self.assertRaises(ValueError):
                make_archive(evidence, root/"changed.tar.gz", inventory)


if __name__ == "__main__":
    unittest.main()
