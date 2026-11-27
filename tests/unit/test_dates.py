"""Unit tests for date finder, format-preserving shift, and date-order oracle (D1, D2)."""
from __future__ import annotations

import unittest

from kavachmed import dates
from kavachmed.contracts import Field, Record
from kavachmed.policy import load_policy


class TestDatesAndOracle(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = load_policy()
        self.ctx = {"policy": self.policy}

    def test_all_date_formats_and_roundtrip(self) -> None:
        samples = [
            ("iso", "2026-04-12", 45, "2026-05-27"),
            ("iso_dt", "2026-04-12T09:30:00+05:30", 10, "2026-04-22T09:30:00+05:30"),
            ("dmy_slash", "12/04/2026", 30, "12/05/2026"),
            ("dmy_dash", "12-04-2026", -12, "31-03-2026"),
            ("dmy_dot_2y", "12.04.26", 5, "17.04.26"),
            ("dmy_named", "12 Apr 1977", 30, "12 May 1977"),
            ("mdy_named", "April 12, 2026", 20, "May 2, 2026"),
            ("deva", "१२/०४/२०२६", 30, "१२/०५/२०२६"),
            ("yearless", "Visit: 5 September", 30, "5 October"),
            ("chat", "[30/05/26, 9:32 pm]", 5, "[04/06/26, 9:32 pm]"),
        ]
        for label, txt, delta, expected_sub in samples:
            rec = Record(key=(f"{label}.txt", "doc"), fields=[Field(0, "doc", txt)])
            refs = dates.find(rec, self.ctx, order_override="dmy")
            self.assertEqual(len(refs), 1, f"failed to find date in {label}: {txt}")
            shifted = dates.shift(refs[0], delta)
            self.assertIn(expected_sub, shifted if label != "chat" else f"[{shifted}]")
            rec2 = Record(key=(f"{label}.txt", "doc"), fields=[Field(0, "doc", shifted)])
            refs2 = dates.find(rec2, self.ctx, order_override="dmy")
            self.assertEqual(len(refs2), 1, f"shifted did not re-parse for {label}: {shifted}")
            self.assertEqual(refs2[0].fmt, refs[0].fmt)

    def test_shift_accepts_raw_string_for_edits_compat(self) -> None:
        self.assertEqual(dates.shift("2026-04-12", 45), "2026-05-27")
        self.assertEqual(dates.shift("12/04/2026", 30), "12/05/2026")
        self.assertEqual(dates.shift("not-a-date", 30), "")

    def test_clinical_numerics_not_flagged_as_dates(self) -> None:
        txt = "Tab Dolo 650 1-0-1 x 5 days; BP 111/86 mmHg; Hb 10.5 g/dL; ref 70-140; SNJ-26-053497"
        rec = Record(key=("note.txt", "doc"), fields=[Field(0, "doc", txt)])
        refs = dates.find(rec, self.ctx)
        self.assertEqual(refs, [])

    def test_order_oracle_arogya_vs_sanjeevani(self) -> None:
        rec_ass = Record(
            key=("registration/arogya_legacy_export.csv", "row:0"),
            fields=[Field(0, "DOB (MM/DD/YYYY)", "01/02/2026"), Field(1, "Registered", "02/25/2026")],
            meta={"hospital": "ASS", "header": ["DOB (MM/DD/YYYY)", "Registered"]},
        )
        rec_snj = Record(
            key=("registration/sanjeevani_registration.csv", "row:0"),
            fields=[Field(0, "DOB", "01/02/2026"), Field(1, "Reg Date", "25/02/2026")],
            meta={"hospital": "SNJ", "header": ["DOB", "Reg Date"]},
        )
        orders = dates.infer_date_orders([rec_ass, rec_snj], self.policy)
        self.assertEqual(orders["registration/arogya_legacy_export.csv"], "mdy")
        self.assertEqual(orders["registration/sanjeevani_registration.csv"], "dmy")

        refs_ass = dates.find(rec_ass, self.ctx, order_override="mdy")
        self.assertEqual((refs_ass[0].month, refs_ass[0].day), (1, 2))
        refs_snj = dates.find(rec_snj, self.ctx, order_override="dmy")
        self.assertEqual((refs_snj[0].month, refs_snj[0].day), (2, 1))

    def test_year_less_resolution(self) -> None:
        rec = Record(key=("note.txt", "doc"), fields=[Field(0, "doc", "Visit: 5 September")])
        refs = dates.find(rec, self.ctx)
        self.assertFalse(refs[0].has_year)
        yr = dates.resolve_year_less_year(refs[0], [2024, 2024, 2026], default_ref_year=2026)
        self.assertEqual(yr, 2024)
        self.assertEqual(dates.shift(refs[0], 30, resolved_year=yr), "5 October")


if __name__ == "__main__":
    unittest.main()
