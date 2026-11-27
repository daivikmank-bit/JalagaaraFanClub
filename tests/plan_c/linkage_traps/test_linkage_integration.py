"""Tests for Plan C V7 integration harness, trap suite, and E6/E7 evaluation."""

from __future__ import annotations

import hashlib
import unittest

from kavachmed.evidence import Field, RecKey, Record, Span
from kavachmed.linkage import FilenameSignal
from tools.linkage_integration import (
    STAGE_MISSING_SPAN,
    VALID_FAILURE_STAGES,
    compare_detector_and_oracle,
    render_integration_summary,
    run_linkage_pipeline,
)


class _FakeKeys:
    def keyed(self, purpose: str, *fields: str) -> bytes:
        h = hashlib.sha256()
        h.update(purpose.encode("utf-8"))
        for f in fields:
            h.update(b"\x00")
            h.update(f.encode("utf-8"))
        return h.digest()


def _span(rec: Record, field_idx: int, substr: str, typ: str, role: str = "patient") -> Span:
    text = rec.fields[field_idx].text
    start = text.index(substr)
    return Span(
        key=rec.key,
        field_idx=field_idx,
        start=start,
        end=start + len(substr),
        typ=typ,
        role=role,
        score=1.0,
        src="fixture",
    )


class TestLinkageIntegrationAndTraps(unittest.TestCase):
    def test_phase3_detector_oracle_parity_drop_span_and_text_free_diagnostics(self) -> None:
        keys = _FakeKeys()
        r1 = Record(
            key=("opd_01.csv", "0"),
            fields=[
                Field(0, "name", "Arif Khan"),
                Field(1, "sex", "M"),
                Field(2, "dob", "1982-04-10"),
                Field(3, "phone", "9811122233"),
            ],
        )
        r2 = Record(
            key=("9811122233_report.txt", "0"),
            fields=[
                Field(0, "note", "Patient Arif Khan (M)"),
            ],
        )
        records = [r1, r2]
        digests = {r.key[0]: hashlib.sha256(r.key[0].encode()).hexdigest() for r in records}
        truth = {r1.key: "P1", r2.key: "P1"}
        fn_signals = [
            FilenameSignal(input_file="9811122233_report.txt", typ="PHONE", value="9811122233"),
        ]

        def oracle_spans(rec: Record) -> list[Span]:
            if rec.key == r1.key:
                return [
                    _span(rec, 0, "Arif Khan", "PATIENT_NAME"),
                    _span(rec, 1, "M", "SEX"),
                    _span(rec, 2, "1982-04-10", "DOB"),
                    _span(rec, 3, "9811122233", "PHONE"),
                ]
            return [
                _span(rec, 0, "Arif Khan", "PATIENT_NAME"),
                _span(rec, 0, "M", "SEX"),
            ]

        # 1. Equivalent detector spans -> identical assignments
        det_eq, ora_res = compare_detector_and_oracle(
            records,
            oracle_spans,
            oracle_spans,
            keys,
            filename_signals=fn_signals,
            file_digests=digests,
            truth_by_key=truth,
        )
        self.assertEqual(det_eq.resolution.patient_of, ora_res.resolution.patient_of)
        self.assertEqual(det_eq.parity.f05, 1.0)
        self.assertEqual(det_eq.coverage, ora_res.coverage)

        # 2. Drop a body span in detector fixture -> missing_or_mis_scoped_span & coverage drop
        def dropped_detector(rec: Record) -> list[Span]:
            if rec.key == r2.key:
                return [_span(rec, 0, "M", "SEX")]  # dropped PERSON span
            return oracle_spans(rec)

        det_drop, _ = compare_detector_and_oracle(
            records,
            dropped_detector,
            oracle_spans,
            keys,
            filename_signals=fn_signals,
            file_digests=digests,
            truth_by_key=truth,
        )
        self.assertLess(det_drop.coverage.with_names, ora_res.coverage.with_names)
        self.assertIsNone(det_drop.resolution.patient_of[r2.key])
        self.assertEqual(det_drop.stage_counts.get(STAGE_MISSING_SPAN, 0), 1)
        for st in det_drop.stage_counts:
            self.assertIn(st, VALID_FAILURE_STAGES)

        # Rendered diagnostics & waterfalls must never contain raw names, phones, or filenames
        summary = render_integration_summary(det_drop)
        combined_text = summary + "\n" + "\n".join(det_drop.waterfall_samples)
        self.assertNotIn("Arif", combined_text)
        self.assertNotIn("9811122233", combined_text)
        self.assertNotIn("report.txt", combined_text)

    def test_phase4_complete_trap_suite(self) -> None:
        keys = _FakeKeys()

        # Trap 1: Cross-script language (Devanagari + Latin with independent relative corroboration)
        t1_lat = Record(
            key=("t1_lat.txt", "0"),
            fields=[
                Field(0, "name", "Arif Khan"),
                Field(1, "sex", "M"),
                Field(2, "dob", "1984-01-01"),
                Field(3, "rel", "s/o Rahim Khan"),
            ],
        )
        t1_dev = Record(
            key=("t1_dev.txt", "0"),
            fields=[
                Field(0, "name", "आरिफ खान"),
                Field(1, "sex", "M"),
                Field(2, "dob", "1984-01-01"),
                Field(3, "rel", "s/o Rahim Khan"),
            ],
        )

        # Trap 2: Fragmented notes (complementary evidence without transitive conflict)
        t2_a = Record(
            key=("t2_a.txt", "0"),
            fields=[
                Field(0, "name", "Priya Nair"),
                Field(1, "sex", "F"),
                Field(2, "rel", "w/o Suresh Nair"),
                Field(3, "phone", "9822233344"),
            ],
        )
        t2_b = Record(
            key=("t2_b.txt", "0"),
            fields=[
                Field(0, "name", "Priya Nair"),
                Field(1, "sex", "F"),
                Field(2, "rel", "w/o Suresh Nair"),
            ],
        )

        # Trap 3: Homonyms (identical names, incompatible birth years -> separate)
        t3_a = Record(
            key=("t3_a.txt", "0"),
            fields=[Field(0, "name", "Sunita Sharma"), Field(1, "sex", "F"), Field(2, "dob", "1965-01-01")],
        )
        t3_b = Record(
            key=("t3_b.txt", "0"),
            fields=[Field(0, "name", "Sunita Sharma"), Field(1, "sex", "F"), Field(2, "dob", "1995-01-01")],
        )

        # Trap 4: Hindi record with no ID (single corroborated owner links; ambiguous stays null)
        t4_owner = Record(
            key=("t4_owner.txt", "0"),
            fields=[
                Field(0, "name", "Meera Joshi"),
                Field(1, "sex", "F"),
                Field(2, "rel", "w/o Gopal Joshi"),
                Field(3, "aadhaar", "555566667777"),
            ],
        )
        t4_hindi = Record(
            key=("t4_hindi.txt", "0"),
            fields=[
                Field(0, "name", "मीरा जोशी"),
                Field(1, "sex", "F"),
                Field(2, "rel", "पत्नी गोपाल जोशी"),
            ],
        )

        # Trap 5: Re-registration (two scoped IDs for same patient with corroboration link;
        # contested ID cannot force merge)
        t5_a = Record(
            key=("t5_a.txt", "0"),
            fields=[
                Field(0, "name", "Vikram Singh"),
                Field(1, "sex", "M"),
                Field(2, "uhid", "AIIMS/2023/0011"),
                Field(3, "phone", "9833344455"),
            ],
        )
        t5_b = Record(
            key=("t5_b.txt", "0"),
            fields=[
                Field(0, "name", "Vikram Singh"),
                Field(1, "sex", "M"),
                Field(2, "uhid", "AIIMS/2024/0088"),
                Field(3, "phone", "9833344455"),
            ],
        )

        records = [t1_lat, t1_dev, t2_a, t2_b, t3_a, t3_b, t4_owner, t4_hindi, t5_a, t5_b]
        truth: dict[RecKey, str | None] = {
            t1_lat.key: "P_T1",
            t1_dev.key: "P_T1",
            t2_a.key: "P_T2",
            t2_b.key: "P_T2",
            t3_a.key: "P_T3A",
            t3_b.key: "P_T3B",
            t4_owner.key: "P_T4",
            t4_hindi.key: "P_T4",
            t5_a.key: "P_T5",
            t5_b.key: "P_T5",
        }

        def spans(rec: Record) -> list[Span]:
            out: list[Span] = []
            for f in rec.fields:
                if f.path == "name":
                    out.append(_span(rec, f.idx, f.text, "PATIENT_NAME"))
                elif f.path == "sex":
                    out.append(_span(rec, f.idx, f.text, "SEX"))
                elif f.path == "age":
                    out.append(_span(rec, f.idx, f.text, "AGE"))
                elif f.path == "dob":
                    out.append(_span(rec, f.idx, f.text, "DOB"))
                elif f.path == "phone":
                    out.append(_span(rec, f.idx, f.text, "PHONE"))
                elif f.path == "aadhaar":
                    out.append(_span(rec, f.idx, f.text, "AADHAAR"))
                elif f.path == "uhid":
                    out.append(_span(rec, f.idx, f.text, "UHID"))
                elif f.path == "rel":
                    parts = f.text.split(" ", 1)
                    if f.text.startswith(("w/o ", "s/o ", "d/o ")):
                        out.append(_span(rec, f.idx, f.text[4:], "RELATIVE_NAME", role="relative"))
                    elif f.text.startswith("पत्नी "):
                        out.append(_span(rec, f.idx, f.text[len("पत्नी "):], "RELATIVE_NAME", role="relative"))
                    else:
                        out.append(_span(rec, f.idx, parts[1], "RELATIVE_NAME", role="relative"))
            return out

        digests = {r.key[0]: hashlib.sha256(r.key[0].encode()).hexdigest() for r in records}
        res = run_linkage_pipeline(records, spans, keys, file_digests=digests, truth_by_key=truth)
        p = res.resolution.patient_of
        # Trap 1
        self.assertIsNotNone(p[t1_lat.key])
        self.assertEqual(p[t1_lat.key], p[t1_dev.key])
        # Trap 2
        self.assertIsNotNone(p[t2_a.key])
        self.assertEqual(p[t2_a.key], p[t2_b.key])
        # Trap 3 (homonyms separated -> unlinked singletons)
        self.assertIsNone(p[t3_a.key])
        self.assertIsNone(p[t3_b.key])
        # Trap 4
        self.assertIsNotNone(p[t4_owner.key])
        self.assertEqual(p[t4_owner.key], p[t4_hindi.key])
        # Trap 5
        self.assertIsNotNone(p[t5_a.key])
        self.assertEqual(p[t5_a.key], p[t5_b.key])

    def test_phase5_e6_filename_on_off_and_e7_temporal_family_modes(self) -> None:
        keys = _FakeKeys()
        r_reg = Record(
            key=("reg.csv", "0"),
            fields=[
                Field(0, "name", "Deepak Verma"),
                Field(1, "sex", "M"),
                Field(2, "uhid", "KEM/2024/0101"),
                Field(3, "phone", "9844455566"),
            ],
            meta={"dates": ["2024-03-01"]},
        )
        r_visit = Record(
            key=("visit.txt", "0"),
            fields=[
                Field(0, "name", "Deepak Verma"),
                Field(1, "sex", "M"),
                Field(2, "uhid", "KEM/2024/0101"),
            ],
            meta={"dates": ["2024-03-05"]},
        )
        # Phone-named lab report with only name + sex in body, plus date
        r_lab = Record(
            key=("9844455566_cbc.txt", "0"),
            fields=[
                Field(0, "name", "Deepak Verma"),
                Field(1, "sex", "M"),
            ],
            meta={"dates": ["2024-03-10"]},
        )
        # Unlinked record with same name (different patient, no corroboration)
        r_other = Record(
            key=("other.txt", "0"),
            fields=[Field(0, "name", "Deepak Verma"), Field(1, "sex", "M"), Field(2, "dob", "1950-01-01")],
        )

        records = [r_reg, r_visit, r_lab, r_other]
        truth = {r_reg.key: "P1", r_visit.key: "P1", r_lab.key: "P1", r_other.key: "P2"}
        fn_signals = [
            FilenameSignal(input_file="9844455566_cbc.txt", typ="PHONE", value="9844455566"),
        ]

        def spans(rec: Record) -> list[Span]:
            out: list[Span] = []
            for f in rec.fields:
                if f.path == "name":
                    out.append(_span(rec, f.idx, f.text, "PATIENT_NAME"))
                elif f.path == "sex":
                    out.append(_span(rec, f.idx, f.text, "SEX"))
                elif f.path == "uhid":
                    out.append(_span(rec, f.idx, f.text, "UHID"))
                elif f.path == "phone":
                    out.append(_span(rec, f.idx, f.text, "PHONE"))
                elif f.path == "dob":
                    out.append(_span(rec, f.idx, f.text, "DOB"))
            return out

        digests = {r.key[0]: hashlib.sha256(r.key[0].encode()).hexdigest() for r in records}
        # E6: filename-off vs filename-on
        run_off = run_linkage_pipeline(records, spans, keys, file_digests=digests, truth_by_key=truth)
        run_on = run_linkage_pipeline(
            records,
            spans,
            keys,
            filename_signals=fn_signals,
            file_digests=digests,
            truth_by_key=truth,
        )
        self.assertGreater(run_on.parity.recall, run_off.parity.recall)
        self.assertEqual(run_on.parity.fp, 0)
        self.assertEqual(run_on.parity.f05, 1.0)
        self.assertIsNone(run_on.resolution.patient_of[r_other.key])

        # E7: single_graph vs temporal_family on the same V7 assignments
        run_tf = run_linkage_pipeline(
            records,
            spans,
            keys,
            policy={"family_mode": "temporal_family"},
            filename_signals=fn_signals,
            file_digests=digests,
            truth_by_key=truth,
        )
        self.assertEqual(run_tf.resolution.patient_of, run_on.resolution.patient_of)
        # P1 records share a single family and shift
        self.assertEqual(run_tf.resolution.family_of[r_reg.key], run_tf.resolution.family_of[r_lab.key])
        self.assertEqual(
            run_tf.resolution.shift_days[run_tf.resolution.family_of[r_reg.key]],
            run_tf.resolution.shift_days[run_tf.resolution.family_of[r_lab.key]],
        )


if __name__ == "__main__":
    unittest.main()
