import json
import tempfile
import unittest
from pathlib import Path

from kavachmed.evidence import Evidence, Field, Record, Span, extract
from tools.evidence_eval import evaluate_records_with_oracle
from tools.oracle import load_oracle


class _StubDateProvider:
    def __init__(self, mapping: dict[tuple[str, str], list[dict]]) -> None:
        self._mapping = mapping

    def find(self, rec: Record) -> list[dict]:
        return list(self._mapping.get(rec.key, []))


class TestEvidenceExtraction(unittest.TestCase):
    def test_empty_record_and_span_validation_guards(self) -> None:
        empty_rec = Record(key=("empty.txt", "doc"), fields=[], meta={})
        ev_empty = extract(empty_rec, [])
        self.assertEqual(ev_empty, Evidence(key=("empty.txt", "doc")))

        # Nonconsecutive field indexes 2 and 9; combining marks in Devanagari.
        name_text = "मोहम्मद आरिफ़"  # contains nukta/combining mark
        rec = Record(
            key=("reg/a.csv", "row:1"),
            fields=[
                Field(idx=2, path="name", text=name_text),
                Field(idx=9, path="phone", text="+91 98765 43210"),
                Field(idx=12, path="@filename", text="9999988888_rep.csv", vtype="path"),
            ],
            meta={},
        )
        spans = [
            Span(("reg/a.csv", "row:1"), 2, 0, len(name_text), "PATIENT_NAME", "patient", 1.0, "ids"),
            # Duplicate & reordered
            Span(("reg/a.csv", "row:1"), 2, 0, len(name_text), "PATIENT_NAME", "patient", 1.0, "ids"),
            # Protected clinical lock -> skipped
            Span(("reg/a.csv", "row:1"), 9, 0, 15, "PHONE", "patient", 1.0, "shield", protected=True),
            # Filename channel -> skipped in V3
            Span(("reg/a.csv", "row:1"), 12, 0, 10, "PHONE", "patient", 1.0, "ids"),
            # Wrong record, unknown field, negative, reversed, overrun -> skipped
            Span(("other.csv", "row:1"), 2, 0, 5, "PATIENT_NAME", "patient", 1.0, "ids"),
            Span(("reg/a.csv", "row:1"), 99, 0, 5, "PATIENT_NAME", "patient", 1.0, "ids"),
            Span(("reg/a.csv", "row:1"), 2, -1, 5, "PATIENT_NAME", "patient", 1.0, "ids"),
            Span(("reg/a.csv", "row:1"), 2, 5, 2, "PATIENT_NAME", "patient", 1.0, "ids"),
            Span(("reg/a.csv", "row:1"), 2, 0, len(name_text) + 5, "PATIENT_NAME", "patient", 1.0, "ids"),
        ]
        ev = extract(rec, spans)
        self.assertEqual([t.latin for t in ev.names], ["mhmd", "arif"])
        self.assertEqual([t.skeleton for t in ev.names if t.skeleton], ["rf"])
        self.assertEqual(ev.phones, [])

        # Reordered spans produce identical Evidence
        ev_rev = extract(rec, list(reversed(spans)))
        self.assertEqual(ev, ev_rev)

    def test_names_kinship_and_sex_cascade(self) -> None:
        # Patient + relative + doctor lookalike in one note
        text = "Pt: ब्रिजेश शर्मा, S/o रमेश शर्मा, seen by Dr. Anita"
        rec = Record(
            key=("notes/n1.txt", "doc"),
            fields=[Field(idx=0, path="text", text=text)],
            meta={},
        )
        p_start = text.index("ब्रिजेश शर्मा")
        r_start = text.index("रमेश शर्मा")
        d_start = text.index("Anita")
        spans = [
            Span(rec.key, 0, p_start, p_start + len("ब्रिजेश शर्मा"), "PATIENT_NAME", "unknown", 1.0, "labels"),
            Span(rec.key, 0, r_start, r_start + len("रमेश शर्मा"), "RELATIVE_NAME", "unknown", 1.0, "labels"),
            Span(rec.key, 0, d_start, d_start + len("Anita"), "PATIENT_NAME", "doctor", 1.0, "labels"),
        ]
        ev = extract(rec, spans)
        self.assertEqual([t.skeleton for t in ev.names], ["brjsh", None])
        self.assertEqual([t.skeleton for t in ev.kin], ["rmsh", None])
        self.assertEqual(ev.kin_type, "son_of")
        self.assertEqual(ev.sex, "M")

        # Parameterize S/o, D/o, W/o
        for cue, expected_kin, expected_sex in [
            ("S/o", "son_of", "M"),
            ("D/o", "daughter_of", "F"),
            ("W/o", "wife_of", "F"),
        ]:
            t = f"Asha {cue} Kamal"
            r = Record(key=("n.txt", "doc"), fields=[Field(0, "text", t)], meta={})
            s = [
                Span(r.key, 0, 0, 4, "PATIENT_NAME", "patient", 1.0, "labels"),
                Span(r.key, 0, len(t) - 5, len(t), "RELATIVE_NAME", "relative", 1.0, "labels"),
            ]
            e = extract(r, s)
            self.assertEqual(e.kin_type, expected_kin)
            self.assertEqual(e.sex, expected_sex)

        # Explicit field precedence & conflicting explicit fields
        r_exp = Record(
            key=("csv", "row:0"),
            fields=[
                Field(0, "text", "W/o Kamal"),
                Field(1, "sex", "Male"),
            ],
            meta={},
        )
        e_exp = extract(r_exp, [Span(r_exp.key, 0, 4, 9, "RELATIVE_NAME", "relative", 1.0, "labels")])
        self.assertEqual(e_exp.sex, "M")  # explicit wins over kin

        r_conf = Record(
            key=("csv", "row:1"),
            fields=[Field(0, "sex", "Male"), Field(1, "gender", "Female")],
            meta={},
        )
        self.assertIsNone(extract(r_conf, []).sex)

        # Relative's honorific must not drive patient sex
        r_rel_hon = Record(
            key=("csv", "row:2"),
            fields=[Field(0, "text", "Patient: Kiran, C/O Shri Ramesh")],
            meta={},
        )
        s_rel = [
            Span(r_rel_hon.key, 0, 9, 14, "PATIENT_NAME", "patient", 1.0, "labels"),
            Span(r_rel_hon.key, 0, 20, 31, "RELATIVE_NAME", "relative", 1.0, "labels"),
        ]
        self.assertIsNone(extract(r_rel_hon, s_rel).sex)

        # Conflicting kinship cues produce unknown relation
        r_conf_kin = Record(
            key=("csv", "row:3"),
            fields=[Field(0, "text", "Kiran S/o Ramesh D/o Suresh")],
            meta={},
        )
        s_conf_kin = [
            Span(r_conf_kin.key, 0, 10, 16, "RELATIVE_NAME", "relative", 1.0, "labels"),
            Span(r_conf_kin.key, 0, 21, 27, "RELATIVE_NAME", "relative", 1.0, "labels"),
        ]
        self.assertIsNone(extract(r_conf_kin, s_conf_kin).kin_type)

        # Hindi gender word fallback when no explicit field, kin cue, or honorific exists
        r_hi = Record(
            key=("csv", "row:4"),
            fields=[Field(0, "note", "मरीज महिला 32 वर्ष")],
            meta={},
        )
        self.assertEqual(extract(r_hi, []).sex, "F")

    def test_ids_phones_and_admission_keys(self) -> None:
        rec = Record(
            key=("reg/hosp.csv", "row:0"),
            fields=[
                Field(0, "uhid", "TST/26/000123"),
                Field(1, "unscoped_mrn", "000123"),
                Field(2, "aadhaar", "१२३४ ५६७८ ९०१२"),  # Indic digits
                Field(3, "pan", "abcde1234f"),
                Field(4, "phone1", "+91 98765-43210"),
                Field(5, "phone2", "09876543210"),
                Field(6, "short_phone", "12345"),
                Field(7, "dose", "650mg"),
                Field(8, "adm1", "ADM-2026-01"),
            ],
            meta={"hospital": "TST"},
        )
        spans = [
            Span(rec.key, 0, 0, 13, "MRN", "patient", 1.0, "ids"),
            Span(rec.key, 2, 0, 14, "AADHAAR", "patient", 1.0, "ids"),
            Span(rec.key, 3, 0, 10, "PAN", "patient", 1.0, "ids"),
            Span(rec.key, 4, 0, 15, "PHONE", "patient", 1.0, "ids"),
            Span(rec.key, 5, 0, 11, "PHONE", "patient", 1.0, "ids"),
            Span(rec.key, 6, 0, 5, "PHONE", "patient", 1.0, "ids"),
            Span(rec.key, 7, 0, 5, "CLINICAL", "none", 1.0, "shield", protected=True),
            Span(rec.key, 8, 0, 11, "ADM_KEY", "patient", 1.0, "ids"),
        ]
        ev = extract(rec, spans)
        self.assertEqual(ev.ids["uhid:tst"], ["000123"])
        self.assertEqual(ev.ids["aadhaar"], ["123456789012"])
        self.assertEqual(ev.ids["pan"], ["ABCDE1234F"])
        self.assertEqual(ev.phones, ["9876543210"])
        self.assertEqual(ev.adm_key, "tst:adm-2026-01")

        # Missing hospital issuer leaves unscoped MRN unavailable
        rec_no_hosp = Record(
            key=("reg/x.csv", "row:0"),
            fields=[Field(0, "mrn", "000123")],
            meta={},
        )
        ev_no_hosp = extract(rec_no_hosp, [Span(rec_no_hosp.key, 0, 0, 6, "MRN", "patient", 1.0, "ids")])
        self.assertEqual(ev_no_hosp.ids, {})

        # Conflicting admission keys within one record -> None
        rec_conf_adm = Record(
            key=("reg/x.csv", "row:1"),
            fields=[Field(0, "a1", "ADM-1"), Field(1, "a2", "ADM-2")],
            meta={"hospital": "TST"},
        )
        ev_conf_adm = extract(
            rec_conf_adm,
            [
                Span(rec_conf_adm.key, 0, 0, 5, "ADM_KEY", "patient", 1.0, "ids"),
                Span(rec_conf_adm.key, 1, 0, 5, "ADM_KEY", "patient", 1.0, "ids"),
            ],
        )
        self.assertIsNone(ev_conf_adm.adm_key)

        # Distinct national IDs, issuer-scoped insurance, diagnosis code ignored, and issuer mismatch
        rec_extra = Record(
            key=("reg/y.csv", "row:0"),
            fields=[
                Field(0, "abha", "12-3456-7890-1234"),
                Field(1, "ins", "POL-9988"),
                Field(2, "dx", "E11.9"),
                Field(3, "other_uhid", "OTH/26/000999"),
                Field(4, "adm", "ADM-2026-01"),
            ],
            meta={"hospital": "SNJ"},
        )
        diag: dict[str, int] = {}
        ev_extra = extract(
            rec_extra,
            [
                Span(rec_extra.key, 0, 0, 17, "ABHA_NUMBER", "patient", 1.0, "ids"),
                Span(rec_extra.key, 1, 0, 8, "INSURANCE_ID", "patient", 1.0, "ids"),
                Span(rec_extra.key, 2, 0, 5, "ICD10", "none", 1.0, "clinical"),
                Span(rec_extra.key, 3, 0, 13, "UHID", "patient", 1.0, "ids"),
                Span(rec_extra.key, 4, 0, 11, "ADM_KEY", "patient", 1.0, "ids"),
            ],
            diagnostics=diag,
        )
        self.assertEqual(ev_extra.ids["abha"], ["12345678901234"])
        self.assertEqual(ev_extra.ids["ins:snj"], ["POL9988"])
        self.assertEqual(ev_extra.ids["uhid:oth"], ["000999"])
        self.assertNotEqual(ev_extra.adm_key, ev.adm_key)
        self.assertEqual(ev_extra.adm_key, "snj:adm-2026-01")
        self.assertEqual(diag["issuer_mismatches"], 1)

    def test_birth_years_dates_and_geography(self) -> None:
        rec = Record(
            key=("opd/v.json", "/visits/0"),
            fields=[
                Field(0, "age", "40 yrs"),
                Field(1, "district", "लखनऊ"),
                Field(2, "pin", "226001"),
            ],
            meta={"hospital": "SNJ"},
        )
        spans = [
            Span(rec.key, 0, 0, 6, "AGE", "patient", 1.0, "context"),
            Span(rec.key, 1, 0, len("लखनऊ"), "DISTRICT", "patient", 1.0, "gazetteer"),
            Span(rec.key, 2, 0, 6, "PIN", "patient", 1.0, "ids"),
        ]
        dp = _StubDateProvider(
            {
                rec.key: [
                    {"text": "2026-05-10", "role": "encounter"},
                    {"text": "2026-05-12", "role": "clinical"},
                ]
            }
        )

        def lgd(level: str, val: str, ctx: dict) -> str | None:
            if level == "district" and val in {"लखनऊ", "Lucknow"}:
                return "LGD-UP-LKO"
            return None

        ev = extract(rec, spans, date_provider=dp, lgd_lookup=lgd)
        self.assertEqual(ev.birth_years, (1985, 1986))
        self.assertEqual(ev.dates, ["2026-05-10", "2026-05-12"])
        self.assertEqual(ev.geo["district_code"], "LGD-UP-LKO")
        self.assertEqual(ev.geo["pin"], "226001")

        # Verified 1986 DOB narrows (1985, 1986) -> (1986, 1986)
        dp_dob = _StubDateProvider(
            {
                rec.key: [
                    {"text": "2026-05-10", "role": "encounter"},
                    {"text": "1986-04-01", "role": "dob"},
                ]
            }
        )
        self.assertEqual(extract(rec, spans, date_provider=dp_dob).birth_years, (1986, 1986))

        # Contradictory DOB -> None
        dp_bad = _StubDateProvider(
            {
                rec.key: [
                    {"text": "2026-05-10", "role": "encounter"},
                    {"text": "1970-04-01", "role": "dob"},
                ]
            }
        )
        self.assertIsNone(extract(rec, spans, date_provider=dp_bad).birth_years)

        # Clinical date alone (non-encounter) cannot derive age interval
        dp_clin = _StubDateProvider({rec.key: [{"text": "2026-05-10", "role": "lab"}]})
        self.assertIsNone(extract(rec, spans, date_provider=dp_clin).birth_years)

        # Unresolved date order -> None
        dp_ambig = _StubDateProvider(
            {rec.key: [{"text": "05/06/2026", "role": "encounter", "order": "AMBIGUOUS"}]}
        )
        self.assertIsNone(extract(rec, spans, date_provider=dp_ambig).birth_years)

        # Missing LGD and missing date provider
        ev_missing_providers = extract(rec, spans, date_provider=None, lgd_lookup=None)
        self.assertIsNone(ev_missing_providers.birth_years)
        self.assertNotIn("district_code", ev_missing_providers.geo)
        self.assertEqual(ev_missing_providers.geo["pin"], "226001")

        # Ambiguous district unresolved without parent state, resolved with parent state
        def lgd_parent(level: str, val: str, ctx: dict) -> str | None:
            if level == "district" and val == "Hamirpur":
                if ctx.get("state") == "Himachal Pradesh":
                    return "LGD-HP-HAM"
                if ctx.get("state") == "Uttar Pradesh":
                    return "LGD-UP-HAM"
                return None
            if level == "state" and val == "Himachal Pradesh":
                return "LGD-HP"
            return None

        r_no_parent = Record(key=("g.csv", "row:0"), fields=[Field(0, "d", "Hamirpur")], meta={})
        ev_no_parent = extract(
            r_no_parent,
            [Span(r_no_parent.key, 0, 0, 8, "DISTRICT", "patient", 1.0, "gazetteer")],
            lgd_lookup=lgd_parent,
        )
        self.assertNotIn("district_code", ev_no_parent.geo)

        r_with_parent = Record(
            key=("g.csv", "row:1"),
            fields=[Field(0, "d", "Hamirpur"), Field(1, "s", "Himachal Pradesh")],
            meta={},
        )
        ev_with_parent = extract(
            r_with_parent,
            [
                Span(r_with_parent.key, 0, 0, 8, "DISTRICT", "patient", 1.0, "gazetteer"),
                Span(r_with_parent.key, 1, 0, 16, "STATE", "patient", 1.0, "gazetteer"),
            ],
            lgd_lookup=lgd_parent,
        )
        self.assertEqual(ev_with_parent.geo["district_code"], "LGD-HP-HAM")

    def test_end_to_end_oracle_and_c5_harness(self) -> None:
        truth = {
            "files": {
                "reg/sanjeevani.csv": {
                    "units": {
                        "row:0": {
                            "patient": "PT1",
                            "link": "required",
                            "entities": [
                                {"type": "PATIENT_NAME", "surface": "Mahesh Chauhan", "channel": "content", "role": "patient"},
                                {"type": "PHONE", "surface": "+91 77319 45103", "channel": "content", "role": "patient"},
                                {"type": "MRN", "surface": "SNJ/26/019672", "channel": "content", "role": "patient"},
                                {"type": "AGE", "surface": "48", "channel": "content", "role": "patient"},
                            ],
                            "dates": ["2025-06-30"],
                            "fields": {"gender": "M", "birth_year": 1977, "hospital": "SNJ"},
                        }
                    }
                }
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            kit = Path(tmp) / "kit"
            dev_in = kit / "dev" / "input"
            dev_tr = kit / "dev" / "truth"
            dev_in.mkdir(parents=True)
            dev_tr.mkdir(parents=True)
            (dev_tr / "truth.json").write_text(json.dumps(truth), encoding="utf-8")

            oracle = load_oracle(dev_in, kit_dir=kit)
            rec = Record(
                key=("reg/sanjeevani.csv", "row:0"),
                fields=[
                    Field(0, "name", "Mahesh Chauhan"),
                    Field(1, "phone", "+91 77319 45103"),
                    Field(2, "mrn", "SNJ/26/019672"),
                    Field(3, "gender", "M"),
                    Field(4, "age", "48"),
                ],
                meta={"hospital": "SNJ"},
            )
            dp = _StubDateProvider({rec.key: [{"text": "2025-06-30", "role": "encounter"}]})
            missing_rec = Record(key=("missing.csv", "row:0"), fields=[], meta={})
            report, evidences = evaluate_records_with_oracle([rec, missing_rec], oracle, date_provider=dp)
            self.assertEqual(report.comparable_records, 1)
            self.assertEqual(report.passing_records, 1)
            self.assertEqual(report.missing_oracle_units, 1)
            self.assertEqual(report.target_status, "pass")
            self.assertEqual(evidences[0].ids["uhid:snj"], ["019672"])
            self.assertEqual(evidences[0].phones, ["7731945103"])
            self.assertEqual(evidences[0].birth_years, (1976, 1977))

            # Injected date emulation is tracked separately from measured accuracy
            rep_inj, _ = evaluate_records_with_oracle(
                [rec], oracle, date_provider=dp, injected_date_records={rec.key}
            )
            self.assertEqual(rep_inj.injected_checks, 1)
            self.assertEqual(rep_inj.comparable_records, 0)
            self.assertEqual(rep_inj.target_status, "pending")


if __name__ == "__main__":
    unittest.main()
