"""Tests for Plan C V7 filename evidence handoff and bounded attribution."""

from __future__ import annotations

import hashlib
import random
import unittest

from kavachmed.evidence import Evidence
from kavachmed.indic import normalize_name
from kavachmed.linkage import FilenameSignal, resolve
from kavachmed.linkage.filename import prepare_filename_evidence
from kavachmed.linkage.policy import LinkagePolicy


class _FakeKeys:
    def keyed(self, purpose: str, *fields: str) -> bytes:
        h = hashlib.sha256()
        h.update(purpose.encode("utf-8"))
        for f in fields:
            h.update(b"\x00")
            h.update(f.encode("utf-8"))
        return h.digest()


class TestFilenameEvidence(unittest.TestCase):
    def test_phase1_handoff_normalization_scope_and_shuffle_invariance(self) -> None:
        ev_single = Evidence(key=("reports/r1.txt", "0"), hospital="AIIMS")
        ev_unknown = Evidence(key=("reports/unknown.txt", "0"), hospital=None)
        ev_multi_a = Evidence(key=("reports/multi.csv", "0"), hospital="AIIMS")
        ev_multi_b = Evidence(key=("reports/multi.csv", "1"), hospital="AIIMS")
        evidences = [ev_single, ev_unknown, ev_multi_a, ev_multi_b]

        signals = [
            FilenameSignal(
                input_file="reports/r1.txt",
                typ="UHID",
                value="AIIMS/2024/0042",
                source="filename_a9",
                role="patient",
                confidence=0.95,
            ),
            # Indic-digit phone
            FilenameSignal(
                input_file="reports/r1.txt",
                typ="PHONE",
                value="९८७६५४३२१०",
                source="filename_a9",
            ),
            # Duplicate phone signal
            FilenameSignal(
                input_file="reports/r1.txt",
                typ="PHONE",
                value="+91-98765-43210",
                source="filename_a9",
            ),
            # Multi-patient file without target -> unresolved
            FilenameSignal(
                input_file="reports/multi.csv",
                typ="PHONE",
                value="9123456789",
            ),
            # Multi-patient file WITH valid target -> attached
            FilenameSignal(
                input_file="reports/multi.csv",
                typ="UHID",
                value="0099",
                target_key=("reports/multi.csv", "1"),
                hospital_hint="AIIMS",
            ),
            # Invalid target file mismatch
            FilenameSignal(
                input_file="reports/r1.txt",
                typ="PHONE",
                value="9000000001",
                target_key=("reports/other.txt", "0"),
            ),
            # Nonexistent / nonpatient target
            FilenameSignal(
                input_file="reports/r1.txt",
                typ="PHONE",
                value="9000000002",
                target_key=("reports/r1.txt", "99"),
            ),
            # Unknown issuer (no hospital_hint, no body hospital, and unscoped)
            FilenameSignal(
                input_file="reports/unknown.txt",
                typ="UHID",
                value="12345",
            ),
            # Non-patient role
            FilenameSignal(
                input_file="reports/r1.txt",
                typ="NAME",
                value="Dr Sharma",
                role="doctor",
            ),
        ]

        diag: dict[str, int] = {}
        policy = LinkagePolicy()
        attached = prepare_filename_evidence(signals, evidences, policy=policy, diagnostics=diag)

        self.assertIn(("reports/r1.txt", "0"), attached)
        r1 = attached[("reports/r1.txt", "0")]
        self.assertEqual(r1.ids, (("uhid:aiims", "0042"),))
        self.assertEqual(r1.phones, ("9876543210",))
        self.assertEqual(r1.source, "filename_a9")

        self.assertIn(("reports/multi.csv", "1"), attached)
        self.assertEqual(attached[("reports/multi.csv", "1")].ids, (("uhid:aiims", "0099"),))
        self.assertNotIn(("reports/multi.csv", "0"), attached)

        # Body Evidence objects were not mutated.
        self.assertEqual(ev_single.ids, {})
        self.assertEqual(ev_single.phones, [])

        self.assertEqual(diag.get("filename_unresolved_multi_patient"), 1)
        self.assertEqual(diag.get("filename_target_file_conflict"), 1)
        self.assertEqual(diag.get("filename_invalid_target"), 1)
        self.assertEqual(diag.get("filename_unknown_issuer"), 1)
        self.assertEqual(diag.get("filename_duplicate_signal"), 1)
        self.assertEqual(diag.get("filename_nonpatient_role_skipped"), 1)

        # Shuffle invariance
        shuffled = list(signals)
        random.Random(42).shuffle(shuffled)
        attached_shuf = prepare_filename_evidence(shuffled, evidences, policy=policy)
        self.assertEqual(attached, attached_shuf)

    def test_phase2_phone_named_report_attaches_only_when_uniquely_corroborated(self) -> None:
        keys = _FakeKeys()
        # Trusted body cluster for patient 1
        ev_p1_a = Evidence(
            key=("reg.csv", "0"),
            names=normalize_name("Ramesh Kumar"),
            ids={"aadhaar": ["111122223333"]},
            phones=["9876500001"],
            sex="M",
            birth_years=(1980, 1982),
        )
        ev_p1_b = Evidence(
            key=("visit.txt", "0"),
            names=normalize_name("Ramesh Kumar"),
            ids={"aadhaar": ["111122223333"]},
            sex="M",
            birth_years=(1980, 1982),
        )
        # Unresolved phone-named report with corroborating body name & sex
        ev_fn_ok = Evidence(
            key=("9876500001_lab.pdf", "0"),
            names=normalize_name("Ramesh Kumar"),
            sex="M",
        )
        # Phone-named report with NO corroborating body evidence (only filename name + filename phone)
        ev_fn_uncorrob = Evidence(
            key=("9876500001_uncorrob.pdf", "0"),
            sex="M",
        )

        signals = [
            FilenameSignal(input_file="9876500001_lab.pdf", typ="PHONE", value="9876500001"),
            FilenameSignal(input_file="9876500001_uncorrob.pdf", typ="PHONE", value="9876500001"),
            FilenameSignal(input_file="9876500001_uncorrob.pdf", typ="NAME", value="Ramesh Kumar"),
        ]

        digests = {e.key[0]: hashlib.sha256(e.key[0].encode()).hexdigest() for e in [ev_p1_a, ev_p1_b, ev_fn_ok, ev_fn_uncorrob]}
        res = resolve(
            [ev_p1_a, ev_p1_b, ev_fn_ok, ev_fn_uncorrob],
            keys,
            file_digests=digests,
            filename_signals=signals,
        )
        self.assertIsNotNone(res.patient_of[("reg.csv", "0")])
        self.assertEqual(res.patient_of[("9876500001_lab.pdf", "0")], res.patient_of[("reg.csv", "0")])
        self.assertIsNone(res.patient_of[("9876500001_uncorrob.pdf", "0")])

    def test_phase2_shared_phone_two_owners_and_conflicting_body_id_reject(self) -> None:
        keys = _FakeKeys()
        # Shared phone across 2 distinct patients (different sex/names/IDs)
        ev_s1 = Evidence(
            key=("a.csv", "0"),
            names=normalize_name("Sita Devi"),
            ids={"aadhaar": ["222233334444"]},
            phones=["9999911111"],
            sex="F",
        )
        ev_s2 = Evidence(
            key=("b.csv", "0"),
            names=normalize_name("Mohan Lal"),
            ids={"aadhaar": ["333344445555"]},
            phones=["9999911111"],
            sex="M",
        )
        ev_shared_fn = Evidence(
            key=("9999911111_rep.pdf", "0"),
            names=normalize_name("Sita Devi"),
            sex="F",
        )

        # Two compatible body owners sharing a non-V4-shared phone (impossible in V4 if same phone,
        # so test two compatible clusters with matching name for a filename NAME signal, or
        # a filename ID with conflicting body ID).
        ev_c1 = Evidence(
            key=("c1.csv", "0"),
            names=normalize_name("Arif Khan"),
            ids={"uhid:aiims": ["1001"]},
            sex="M",
            birth_years=(1975, 1975),
        )
        ev_c1_b = Evidence(
            key=("c1b.csv", "0"),
            names=normalize_name("Arif Khan"),
            ids={"uhid:aiims": ["1001"]},
            sex="M",
            birth_years=(1975, 1975),
        )
        # Conflicting body ID (no birth_years so V5 doesn't merge on body alone; filename UHID must reject)
        ev_conflict_id = Evidence(
            key=("AIIMS_1001_scan.pdf", "0"),
            names=normalize_name("Arif Khan"),
            ids={"uhid:aiims": ["9999"]},
            sex="M",
        )

        # Filename-only ID cannot seed a V4 exact bridge between two unresolved records
        # (only name + sex in body -> +2.10 < 4.0; must remain unlinked)
        ev_fn_only_1 = Evidence(
            key=("fn_id_1.pdf", "0"),
            names=normalize_name("Kiran Rao"),
            sex="F",
        )
        ev_fn_only_2 = Evidence(
            key=("fn_id_2.pdf", "0"),
            names=normalize_name("Kiran Rao"),
            sex="F",
        )

        signals = [
            FilenameSignal(input_file="9999911111_rep.pdf", typ="PHONE", value="9999911111"),
            FilenameSignal(input_file="AIIMS_1001_scan.pdf", typ="UHID", value="1001", hospital_hint="AIIMS"),
            FilenameSignal(input_file="fn_id_1.pdf", typ="AADHAAR", value="444455556666"),
            FilenameSignal(input_file="fn_id_2.pdf", typ="AADHAAR", value="444455556666"),
        ]
        evs = [ev_s1, ev_s2, ev_shared_fn, ev_c1, ev_c1_b, ev_conflict_id, ev_fn_only_1, ev_fn_only_2]
        digests = {e.key[0]: hashlib.sha256(e.key[0].encode()).hexdigest() for e in evs}
        diag: dict[str, int] = {}
        res = resolve(
            evs,
            keys,
            file_digests=digests,
            filename_signals=signals,
            diagnostics=diag,
        )

        # Shared phone rejected
        self.assertIsNone(res.patient_of[("9999911111_rep.pdf", "0")])
        # Conflicting body ID leaves record separate from ev_c1 (never merged)
        self.assertNotEqual(res.patient_of[("AIIMS_1001_scan.pdf", "0")], res.patient_of[("c1.csv", "0")])
        # Filename-only ID did not bridge two records with no body IDs
        self.assertIsNone(res.patient_of[("fn_id_1.pdf", "0")])
        self.assertIsNone(res.patient_of[("fn_id_2.pdf", "0")])

        # Diagnostics keys are strictly text-free codes
        for k in diag:
            self.assertNotIn("9999911111", k)
            self.assertNotIn("arif", k.lower())
            self.assertNotIn("scan.pdf", k.lower())


if __name__ == "__main__":
    unittest.main()
