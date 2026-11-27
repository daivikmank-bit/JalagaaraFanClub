"""Unit tests for Layer B independence, normalization, B1-B4 checks, and clinical preservation (D5)."""
from __future__ import annotations

import ast
import base64
import unittest
from pathlib import Path

from kavachmed import residual_b
from kavachmed.policy import load_policy


class TestResidualB(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = load_policy()

    def test_strict_import_ban(self) -> None:
        src = Path(residual_b.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        banned = ("detect", "evidence", "propagate", "residual_a", "shield", "linkage", "typing", "privacy", "urllib")
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    self.assertNotIn(top, banned)
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                top = mod.split(".")[0]
                self.assertNotIn(top, banned)
                for b in banned:
                    self.assertFalse(mod == f"kavachmed.{b}" or mod.startswith(f"kavachmed.{b}."))

    def test_b1_unseen_names_and_safe_subtraction(self) -> None:
        txt = "Patient Xyzzyfoobar arrived with अनदेखानाम and anita and कुमार."
        hits = residual_b.scan(txt, "sanitized/note.txt", [], policy=self.policy)
        checks = {h.check for h in hits}
        self.assertIn("B1", checks)
        self.assertGreaterEqual(len(hits), 4)
        for h in hits:
            self.assertFalse(hasattr(h, "text"))

    def test_b1_homoglyph_and_zero_width(self) -> None:
        disguised = "\u0410bhi\u200bshek K\u0430poor visited."
        hits = residual_b.scan(disguised, "sanitized/note.txt", [], policy=self.policy)
        self.assertTrue(any(h.check == "B1" for h in hits))

    def test_b2_grouped_and_indic_digits(self) -> None:
        txt = "Call 98 76 54 32 10 or ९८७६५४३२१०. Lab WBC 14200/cumm and INR 45000 stay safe."
        clin_start = txt.index("WBC 14200/cumm")
        clin_end = clin_start + len("WBC 14200/cumm")
        hits = residual_b.scan(txt, "sanitized/note.txt", [(clin_start, clin_end, "lab")], policy=self.policy)
        b2_hits = [h for h in hits if h.check == "B2"]
        self.assertGreaterEqual(len(b2_hits), 2)
        self.assertFalse(any(h.start >= clin_start and h.end <= clin_end for h in b2_hits))

    def test_b3_promoted_seed_and_redacted_composite(self) -> None:
        txt = 'Patient_Name: UnseenPerson\nPt_Name: [REDACTED] [REDACTED]\n{"patient_name": "[REDACTED]-[REDACTED]"}\nDistrict: Hardoi\n'
        hits = residual_b.scan(txt, "sanitized/note.txt", [], policy=self.policy, promoted_labels={"pt_name"})
        b3 = [h for h in hits if h.check == "B3"]
        self.assertEqual(len(b3), 1)

    def test_b4_typed_ids_and_base64(self) -> None:
        hidden = base64.b64encode(b"PAN ABCDE1234F phone +91 9876543210").decode("ascii")
        txt = f'{{"attachment": "{hidden}", "ref": "PSN-12345678", "note": "[NAME]"}}'
        hits = residual_b.scan(txt, "sanitized/doc.json", [], policy=self.policy)
        self.assertTrue(any(h.check == "B4" for h in hits))

    def test_clinical_and_placeholders_clean(self) -> None:
        txt = "PSN-7Q2K9D2M visited on 2026-04-15 with fever and asthma. Tab Dolo 650 mg BD x 5 days. [NAME] [PHONE]"
        hits = residual_b.scan(txt, "sanitized/clean_note.txt", [(len(txt) - 25, len(txt), "dose")], policy=self.policy)
        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
