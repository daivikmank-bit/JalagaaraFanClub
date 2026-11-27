"""Unit tests for compliance_map.py and tools/redteam (D9, D10, D15)."""
from __future__ import annotations

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from kavachmed import pipeline
from kavachmed.policy import load_policy
from kavachmed.security import load_keys
from tools import compliance_map
from tools.redteam.attacks import (
    eval_a6_pseudonym_inversion,
    eval_a7_cross_release,
    run_attacks,
    uniqueness_vs_attributes_curve,
)


class TestComplianceAndRedteam(unittest.TestCase):
    def test_compliance_map_reflects_policy_and_lints_clean(self) -> None:
        pol = load_policy()
        md1 = compliance_map.render_compliance_md(pol)
        self.assertEqual(compliance_map.lint_compliance_md(md1), [])
        self.assertIn("location.keep=district", md1)
        self.assertIn("security.min_secret_bytes=1", md1)
        for fr in ("FR-D2", "FR-D1", "FR-K1", "FR-R1", "FR-P1", "FR-P3", "FR-P4", "FR-F1", "FR-X2"):
            self.assertIn(fr, md1)

        pol2 = copy.deepcopy(pol)
        pol2["location"]["keep"] = "state"
        md2 = compliance_map.render_compliance_md(pol2)
        self.assertIn("location.keep=state", md2)
        self.assertNotEqual(md1, md2)

    def test_a6_and_a7_synthetic(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="km-a67-"))
        try:
            inp = tmp / "in"
            inp.mkdir()
            (inp / "visit.csv").write_text("patient_name,visit_date\nAsha Devi,2026-04-01\n", encoding="utf-8")
            pol = load_policy()
            out1 = tmp / "out1"
            out2 = tmp / "out2"
            k1 = load_keys({"RAKSHA_SECRET": "secret-alpha-01"}, pbkdf2_iterations=100)
            k2 = load_keys({"RAKSHA_SECRET": "secret-beta-02"}, pbkdf2_iterations=100)
            pipeline.run(inp, out1, k1, pol)
            pipeline.run(inp, out2, k2, pol)

            (out1 / "records.jsonl").write_text(json.dumps({"file": "sanitized/visit.csv", "loc": "row:0", "patient": "PSN-7Q2K9D2M"}) + "\n", encoding="utf-8")
            (out2 / "records.jsonl").write_text(json.dumps({"file": "sanitized/visit.csv", "loc": "row:0", "patient": "PSN-9X4M1P8R"}) + "\n", encoding="utf-8")
            kit = Path(__file__).resolve().parents[2]
            if (kit / "dev" / "attacker" / "registry.csv").is_file():
                a6 = eval_a6_pseudonym_inversion(out1 / "records.jsonl", kit / "dev" / "attacker" / "registry.csv")
                self.assertEqual(a6["A6_inversion_rate"], 0.0)
            a7 = eval_a7_cross_release(out1, out2)
            self.assertEqual(a7["A7_cross_release_risk"], 0.0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_redteam_calibration_on_kit(self) -> None:
        kit = Path(__file__).resolve().parents[2]
        if not (kit / "dev" / "attacker" / "registry.csv").is_file():
            self.skipTest("dev attacker fixtures not present")
        raw = run_attacks(
            kit / "dev" / "input",
            kit / "dev" / "attacker" / "registry.csv",
            kit / "dev" / "attacker" / "known_facts.jsonl",
            kit / "dev" / "truth" / "truth.json",
        )
        self.assertGreaterEqual(raw["A4_profile_reid"], 0.70)
        self.assertGreaterEqual(raw["A3_record_linkage"], 0.60)
        curve = uniqueness_vs_attributes_curve(kit / "dev" / "attacker" / "registry.csv")
        self.assertEqual(len(curve), 6)
        self.assertGreater(curve[-1]["unique_share"], curve[0]["unique_share"])


if __name__ == "__main__":
    unittest.main()
