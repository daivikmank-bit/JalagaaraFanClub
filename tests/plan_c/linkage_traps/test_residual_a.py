"""Tests for Plan C V8 Layer A residual scanner (Phases 1–3 of c-v8.md)."""

from __future__ import annotations

import base64
import unittest

from kavachmed.evidence import Evidence, Field, Record, Span
from kavachmed.indic import normalize_name
from kavachmed.linkage.filename import FilenameSignal
from kavachmed.residual_a import (
    ExemptSpan,
    Hit,
    LeakKeys,
    ResidualScanError,
    compile_keys,
    normalize_text,
    scan,
    scan_with_status,
)


class TestResidualAPhase1CompileKeys(unittest.TestCase):
    def test_compile_all_key_types_filters_and_redaction(self) -> None:
        rk1 = ("a.csv", "row:1")
        rk2 = ("b.txt", "doc")
        records = [
            Record(
                key=rk1,
                fields=[
                    Field(0, "name", "Shri Ramesh Kumar"),
                    Field(1, "dob", "14/08/1985"),
                    Field(2, "visit_date", "2026-03-11"),
                    Field(3, "email", "ramesh.k85@example.org"),
                    Field(4, "mrn", "MRN-019672"),
                    Field(5, "doc_name", "Dr. Vikram"),
                ],
            )
        ]
        spans = [
            Span(rk1, 0, 0, 17, "PATIENT_NAME", "patient", 1.0, "labels"),
            Span(rk1, 1, 0, 10, "DOB", "patient", 1.0, "labels"),
            Span(rk1, 2, 0, 10, "DATE", "none", 1.0, "labels"),
            Span(rk1, 3, 0, 22, "EMAIL", "patient", 1.0, "ids"),
            Span(rk1, 4, 0, 10, "MRN", "patient", 1.0, "ids"),
            # Non-patient role must not be harvested as patient/relative key.
            Span(rk1, 5, 0, 10, "PATIENT_NAME", "doctor", 1.0, "shield"),
            # Invalid span bounds must be safely ignored.
            Span(rk1, 0, -1, 5, "PATIENT_NAME", "patient", 1.0, "labels"),
            Span(rk1, 0, 5, 500, "PATIENT_NAME", "patient", 1.0, "labels"),
        ]
        evidences = [
            Evidence(
                key=rk2,
                names=normalize_name("Sunita Devi"),
                kin=normalize_name("Mahesh"),
                phones=["+91 98765-43210"],
                geo={"village": "Kheda"},
            )
        ]
        fn_signals = [
            FilenameSignal("c_9811223344.pdf", "PHONE", "9811223344", role="patient"),
            FilenameSignal("c_9811223344.pdf", "PATIENT_NAME", "CanaryTokenX", role="patient"),
            FilenameSignal("dr_sharma.pdf", "PATIENT_NAME", "DoctorIgnored", role="doctor"),
        ]
        diag: dict[str, int] = {}
        keys = compile_keys(
            spans,
            evidences,
            records=records,
            filename_signals=fn_signals,
            diagnostics=diag,
        )

        # Generic/honorific tokens ("shri", "kumar", "devi") are excluded.
        self.assertIn("ramesh", keys.name_tokens)
        self.assertIn("sunita", keys.name_tokens)
        self.assertIn("mahesh", keys.name_tokens)
        self.assertIn("canarytokenx", keys.name_tokens)
        self.assertNotIn("kumar", keys.name_tokens)
        self.assertNotIn("devi", keys.name_tokens)
        self.assertNotIn("shri", keys.name_tokens)
        self.assertNotIn("vikram", keys.name_tokens)
        self.assertNotIn("doctorignored", keys.name_tokens)

        # Clinical date excluded from DOBs; DOB included.
        self.assertIn("14/08/1985", keys.dobs)
        self.assertIn("14081985", keys.dobs)
        self.assertNotIn("2026-03-11", keys.dobs)

        # Digit cores and exact strings.
        self.assertIn("019672", keys.digit_cores)
        self.assertIn("9876543210", keys.digit_cores)
        self.assertIn("9811223344", keys.digit_cores)
        self.assertIn("ramesh.k85@example.org", keys.exact_strings)
        self.assertIn("kheda", keys.exact_strings)

        # Redacted representations must never leak PII or paths.
        rep = repr(keys)
        self.assertNotIn("ramesh", rep)
        self.assertNotIn("9876543210", rep)
        self.assertIn("redacted", rep)

        # Cross-file leak detection: keys from rk1/rk2 found in a third file.
        hits = scan("Patient transferred from Kheda, call 98765 43210.", "third.txt", keys)
        self.assertEqual(len(hits), 2)
        for h in hits:
            self.assertNotIn("Kheda", repr(h))
            self.assertNotIn("third.txt", repr(h))

    def test_empty_harvest_and_order_invariance(self) -> None:
        empty = compile_keys([], [])
        self.assertEqual(empty, LeakKeys())
        self.assertEqual(scan("Clean lab report: Hb 12.4 g/dL", "clean.txt", empty), [])

        ev1 = Evidence(key=("a.txt", "doc"), names=normalize_name("Ananya"), phones=["9876501234"])
        ev2 = Evidence(key=("b.txt", "doc"), names=normalize_name("Bhavesh"), phones=["9123405678"])
        k_ab = compile_keys([], [ev1, ev2])
        k_ba = compile_keys([], [ev2, ev1])
        self.assertEqual(k_ab, k_ba)


class TestResidualAPhase2SurfacesAndEncodings(unittest.TestCase):
    def setUp(self) -> None:
        self.keys = compile_keys(
            [],
            [
                Evidence(
                    key=("src.csv", "row:1"),
                    names=normalize_name("Arif आरिफ़"),
                    phones=["9876543210"],
                    ids={"pan": ["ABCDE1234F"]},
                    geo={"village": "Rampur"},
                )
            ],
        )

    def test_channels_and_unicode_variants(self) -> None:
        # 1. Cell / content with Cyrillic homoglyph + zero-width + Indic digits + split digit core
        content = "Contact \u0410rif\u200b at ९८७६५-४३२१०, PAN abcde1234f"
        hits = scan(
            content,
            "out/arif_report.txt",
            self.keys,
            structured_surfaces={
                "header": ["patient_arif_col"],
                "key": ["source_path_rampur"],
            },
        )
        channels = {h.channel for h in hits}
        self.assertEqual(channels, {"path", "content", "header", "key"})

        # Verify split digit core maps to separator-bearing range in original text.
        phone_hits = [h for h in hits if h.channel == "content" and "९" in content[h.start : h.end]]
        self.assertEqual(len(phone_hits), 1)
        self.assertEqual(content[phone_hits[0].start : phone_hits[0].end], "९८७६५-४३२१०")

        # 2. Devanagari matra retention: आरिफ़ vs आरफ
        self.assertNotEqual(normalize_text("आरिफ़"), normalize_text("आरफ"))
        dev_hits = scan("रोगी आरिफ़ ठीक है", "ok.txt", self.keys)
        self.assertEqual(len(dev_hits), 1)

    def test_fhir_div_and_encoded_wrappers(self) -> None:
        b64_payload = base64.b64encode(b"Patient Arif 9876543210").decode("ascii")
        hex_payload = b"Rampur".hex()
        u_esc = "\\u0041\\u0072\\u0069\\u0066"
        pct_esc = "%52%61%6d%70%75%72"
        html_ent = "&#65;rif"
        text = (
            f'<div xmlns="http://www.w3.org/1999/xhtml">{b64_payload}</div> '
            f"hex={hex_payload} u={u_esc} pct={pct_esc} html={html_ent}"
        )
        outcome = scan_with_status(text, "fhir.json", self.keys)
        self.assertTrue(outcome.verified)
        # All 5 encoded wrappers should yield hits mapped to wrapper coordinates.
        self.assertGreaterEqual(len(outcome.hits), 5)
        b64_start = text.index(b64_payload)
        b64_end = b64_start + len(b64_payload)
        self.assertTrue(any(h.start == b64_start and h.end == b64_end for h in outcome.hits))

    def test_malformed_and_binary_wrappers_fail_closed(self) -> None:
        # Invalid base64 / binary payload marks scan unverified and strict mode raises.
        binary_b64 = base64.b64encode(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\xff\xfe\xfd").decode("ascii")
        outcome = scan_with_status(f"attachment={binary_b64}", "att.json", self.keys)
        self.assertFalse(outcome.verified)
        self.assertEqual(outcome.reason, "unverified_wrapper")
        with self.assertRaises(ResidualScanError):
            scan(f"attachment={binary_b64}", "att.json", self.keys, strict=True)


class TestResidualAPhase3FRR7Exemption(unittest.TestCase):
    def setUp(self) -> None:
        self.keys = compile_keys(
            [],
            [
                Evidence(
                    key=("src.txt", "doc"),
                    names=normalize_name("Sunita"),
                    phones=["9876543210"],
                )
            ],
        )

    def test_clinical_and_doctor_registry_exempt_but_dr_context_caught(self) -> None:
        text = "Dr. Sunita (patient relative Sunita) prescribed Sunita-stat 9876543210."
        doc_start = text.index("Dr. Sunita")
        doc_end = doc_start + len("Dr. Sunita")
        rel_start = text.index("Sunita", doc_end)
        rel_end = rel_start + len("Sunita")

        # 1. Unverified Dr. context (exempt_kind=None) is NOT exempt!
        hits_unverified = scan(
            text,
            "visit.txt",
            self.keys,
            exempt_index=[ExemptSpan("visit.txt", "content", doc_start, doc_end, exempt_kind=None)],
        )
        # Should catch: Dr. Sunita, relative Sunita, Sunita-stat (if boundary? '-' is boundary), phone
        self.assertTrue(any(h.start == doc_start + 4 and h.end == doc_end for h in hits_unverified))
        self.assertTrue(any(h.start == rel_start and h.end == rel_end for h in hits_unverified))

        # 2. Verified doctor_registry + clinical span preserves those, still catches relative & phone.
        clin_start = text.index("Sunita-stat")
        clin_end = clin_start + len("Sunita-stat")
        hits_verified = scan(
            text,
            "visit.txt",
            self.keys,
            exempt_index=[
                ExemptSpan("visit.txt", "content", doc_start, doc_end, exempt_kind="doctor_registry"),
                ExemptSpan("visit.txt", "content", clin_start, clin_end, exempt_kind="drug"),
            ],
        )
        self.assertFalse(any(h.start == doc_start + 4 and h.end == doc_end for h in hits_verified))
        self.assertFalse(any(h.start == clin_start and h.end == clin_start + 6 for h in hits_verified))
        self.assertTrue(any(h.start == rel_start and h.end == rel_end for h in hits_verified))
        self.assertTrue(any(text[h.start : h.end] == "9876543210" for h in hits_verified))

    def test_partial_overlap_cross_span_wrong_channel_and_encoded_wrapper(self) -> None:
        text = "Sunita 9876543210"
        # Partial overlap does not exempt.
        hits_partial = scan(
            text,
            "a.txt",
            self.keys,
            exempt_index=[ExemptSpan("a.txt", "content", 0, 4, exempt_kind="drug")],
        )
        self.assertEqual(len(hits_partial), 2)

        # Wrong file or wrong channel does not exempt.
        hits_wrong_scope = scan(
            text,
            "a.txt",
            self.keys,
            exempt_index=[
                ExemptSpan("b.txt", "content", 0, 6, exempt_kind="drug"),
                ExemptSpan("a.txt", "header", 0, 6, exempt_kind="drug"),
            ],
        )
        self.assertEqual(len(hits_wrong_scope), 2)

        # Broad exemption around base64 wrapper does not exempt decoded marker without decoded coords.
        b64 = base64.b64encode(b"Patient Sunita").decode("ascii")
        hits_b64 = scan(
            b64,
            "a.txt",
            self.keys,
            exempt_index=[ExemptSpan("a.txt", "content", 0, len(b64), exempt_kind="doctor_registry")],
        )
        self.assertEqual(len(hits_b64), 1)

        # Approved name span never hides an unrelated typed identifier inside same range.
        id_in_exempt = "9876543210"
        hits_id = scan(
            id_in_exempt,
            "a.txt",
            self.keys,
            exempt_index=[ExemptSpan("a.txt", "content", 0, len(id_in_exempt), exempt_kind="lab")],
        )
        self.assertEqual(len(hits_id), 1)


if __name__ == "__main__":
    unittest.main()
