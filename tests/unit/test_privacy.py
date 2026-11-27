"""Unit tests for bands, location, dx_rollup, and privacy controller (D3, D4, D13, D14)."""
from __future__ import annotations

import unittest

from kavachmed.contracts import Evidence, Field, ParsedFile, Record, Resolution
from kavachmed.policy import load_policy
from kavachmed.privacy import bands, controller, dx_rollup, location


class TestPrivacyLayer(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = load_policy()
        # Ladder-mechanics fixtures assume k_target=5; the shipped value is tuned separately.
        self.policy["controller"] = {**self.policy.get("controller", {}), "k_target": 5}

    def test_age_extraction_and_duration_exclusion(self) -> None:
        rec = Record(
            key=("note.txt", "doc"),
            fields=[
                Field(0, "doc", "Patient 48 Y / F with K/C/O DM2 x 12 yrs, H/O TB 7 yrs back, BP 70-140"),
            ],
        )
        self.assertEqual(bands.extract_ages_from_record(rec), [48])

        rec_hi = Record(
            key=("hi.txt", "doc"),
            fields=[Field(0, "doc", "आयु: 63 वर्ष, मधुमेह पिछले 9 वर्ष से")],
        )
        self.assertEqual(bands.extract_ages_from_record(rec_hi), [63])

    def test_patient_level_band_truthfulness(self) -> None:
        self.assertEqual(bands.band_for_ages([48]), "40-49")
        self.assertEqual(bands.band_for_ages([49, 51]), "40-59")
        self.assertEqual(bands.band_for_ages([87, 97]), "85+")
        self.assertEqual(bands.band_for_ages([3]), "0-4")

    def test_lgd_location_and_pin(self) -> None:
        lgd = location.get_lgd_index()
        info = lgd.lookup_district("Hardoi")
        self.assertIsNotNone(info)
        assert info is not None
        self.assertEqual(info[1], "Hardoi")
        self.assertEqual(info[2], "Uttar Pradesh")

        cleaned = location.sanitize_address_text(
            "C/O Naresh Chauhan, H.No. 364, Mohalla Sadar, Tehsil Sandila, Distt. Hardoi - 216612",
            level="district",
            pin_mode="none",
        )
        self.assertEqual(cleaned, "Hardoi")
        self.assertEqual(
            location.sanitize_address_text("Distt. Hardoi - 216612", level="state", pin_mode="none"),
            "Uttar Pradesh",
        )

    def test_dx_rollup_multilingual_concept(self) -> None:
        c_en = dx_rollup.canonicalize_dx("Bronchial Asthma")
        c_hi = dx_rollup.canonicalize_dx("दमा")
        c_icd = dx_rollup.canonicalize_dx("J45.9")
        self.assertEqual(c_en[0], c_hi[0])
        self.assertEqual(c_en[0], c_icd[0])

        roll = dx_rollup.plan_dx_rollups({"p1": ["दमा"], "p2": ["asthma"], "p3": ["asthma"], "p4": ["Kala-azar"]}, k_dx=3)
        self.assertNotIn("p1", roll)
        self.assertIn("p4", roll)

    def test_controller_escalation_and_pool_consistency(self) -> None:
        rk1 = ("a.csv", "row:0")
        rk2 = ("b.txt", "doc")
        r1 = Record(key=rk1, fields=[Field(0, "Age", "91"), Field(1, "Gender", "F"), Field(2, "District", "Hardoi")])
        r2 = Record(key=rk2, fields=[Field(0, "doc", "92 Y / F, Distt. Hardoi")])
        pf1 = ParsedFile(rel_path="a.csv", sha256="0", fmt="csv", encoding="utf-8", newline="\n", records=[r1], raw_bytes=b"")
        pf2 = ParsedFile(rel_path="b.txt", sha256="1", fmt="text", encoding="utf-8", newline="\n", records=[r2], raw_bytes=b"")
        res = Resolution(
            patient_of={rk1: "PSN-12345678", rk2: "PSN-12345678"},
            cluster_of={rk1: "C1", rk2: "C1"},
            family_of={rk1: "F1", rk2: "F1"},
            shift_days={"F1": 45},
        )
        evs = [
            Evidence(key=rk1, sex="F", age=91, dx_terms=["Kala-azar"], hospital="SNJ", los_days=12, procedure_terms=["splenectomy"]),
            Evidence(key=rk2, sex="F", age=92, dx_terms=["काला-अज़ार"], hospital="SNJ", los_days=12),
        ]
        plan = controller.build_plan(res, evs, [pf1, pf2], self.policy)
        self.assertEqual(plan.age_band["C1"], "Senior")
        self.assertEqual(plan.location_level[rk1], plan.location_level[rk2])
        self.assertGreaterEqual(plan.risk_summary["escalated_clusters"], 1)

        rk3 = ("c.csv", "row:0")
        r3 = Record(key=rk3, fields=[Field(0, "Age", "88"), Field(1, "Gender", "F"), Field(2, "District", "Lucknow")])
        pf3 = ParsedFile(rel_path="c.csv", sha256="2", fmt="csv", encoding="utf-8", newline="\n", records=[r3], raw_bytes=b"")
        res2 = Resolution(
            patient_of={rk1: "PSN-1", rk2: "PSN-1", rk3: "PSN-2"},
            cluster_of={rk1: "C1", rk2: "C1", rk3: "C2"},
            family_of={rk1: "F1", rk2: "F1", rk3: "F2"},
            shift_days={"F1": 45, "F2": 60},
        )
        evs2 = evs + [Evidence(key=rk3, sex="F", age=88, dx_terms=["Asthma"], los_days=10)]
        plan2 = controller.build_plan(res2, evs2, [pf1, pf2, pf3], self.policy)
        self.assertEqual(plan2.age_band["C1"], "85+")
        self.assertEqual(plan2.location_level[rk1], "state")
        self.assertEqual(plan2.location_level[rk1], plan2.location_level[rk2])

    def test_controller_works_with_lightweight_pass1_records(self) -> None:
        rk = ("fhir/b.json", "doc")
        light = ParsedFile(
            rel_path="fhir/b.json", sha256="x", fmt="fhir", encoding="utf-8", newline="\n",
            records=[], raw_bytes=b"",
            extra={"fhir_qi": {"sex": "female", "birth_year": 1938, "district": "Hardoi", "dx_terms": ["Kala-azar"]}},
        )
        res = Resolution(patient_of={rk: "PSN-12345678"}, cluster_of={rk: "C1"}, family_of={rk: "F1"}, shift_days={"F1": 35})
        plan = controller.build_plan(res, [], [light], self.policy)
        self.assertIn("C1", plan.age_band)
        self.assertIn(rk, plan.location_level)


if __name__ == "__main__":
    unittest.main()
