"""Tests for Plan C V10 (C15) A6 pseudonym inversion and A7 key rotation verification."""

from __future__ import annotations

import base64
import hashlib
import tempfile
import unittest
from pathlib import Path

from tools.redteam.release_privacy import (
    A6ControlVector,
    FrozenReleaseConfig,
    RedteamFixtureError,
    ReleaseRunArtifacts,
    ReleaseRunRecord,
    SyntheticCohort,
    SyntheticEvent,
    SyntheticPatient,
    SyntheticRecordSpec,
    build_default_synthetic_cohort,
    build_release_runner_double,
    evaluate_a6_controls,
    evaluate_a6_release,
    evaluate_a7_rotation,
    render_v10_summary,
    run_v10_verification,
    verify_same_secret_determinism,
)


def _write_release_files(
    out_dir: Path,
    files: dict[str, bytes],
    records_jsonl: bytes = b'{"loc":"row:1"}\n',
) -> None:
    sanitized = out_dir / "sanitized"
    sanitized.mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        p = sanitized / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
    (out_dir / "records.jsonl").write_bytes(records_jsonl)


class TestV10Phase1FixtureAndAlignment(unittest.TestCase):
    def test_duplicate_keys_rejected(self) -> None:
        p1 = SyntheticPatient("p1", uhids=("UH1",))
        with self.assertRaises(RedteamFixtureError):
            SyntheticCohort(patients=(p1, p1), records=())

        r1 = SyntheticRecordSpec("r1", "p1", "a.csv", "row:1", "csv", {})
        r2 = SyntheticRecordSpec("r2", "p1", "a.csv", "row:1", "csv", {})
        with self.assertRaises(RedteamFixtureError):
            SyntheticCohort(patients=(p1,), records=(r1, r2))

        ev = SyntheticEvent("ev1", "p1", "a.csv", "row:1", "d", "2025-01-01")
        r_dup_ev = SyntheticRecordSpec("r1", "p1", "a.csv", "row:1", "csv", {}, events=(ev, ev))
        with self.assertRaises(RedteamFixtureError):
            SyntheticCohort(patients=(p1,), records=(r_dup_ev,))

    def test_rename_alignment_null_denominator_and_missing_ambiguous_coverage(self) -> None:
        patients = (SyntheticPatient("p1", uhids=("SNJ/26/001122",)),)
        records = (
            SyntheticRecordSpec(
                "r1",
                "p1",
                "raw/Ramesh_Sharma.csv",
                "row:1",
                "csv",
                {},
                events=(
                    SyntheticEvent("ev1", "p1", "raw/Ramesh_Sharma.csv", "row:1", "visit_date", "2025-01-10"),
                ),
            ),
            SyntheticRecordSpec("r2_null", None, "raw/orphan.txt", "doc:0", "txt", {}),
            SyntheticRecordSpec("r3_ambig", "p1", "raw/dup.txt", "doc:0", "txt", {}),
        )
        cohort = SyntheticCohort(patients=patients, records=records)

        with tempfile.TemporaryDirectory() as tmp:
            d_a = Path(tmp) / "a"
            d_b = Path(tmp) / "b"
            _write_release_files(d_a, {"san/file_001.csv": b"ok\n", "san/orphan.txt": b"o\n"})
            _write_release_files(d_b, {"san/file_001.csv": b"ok\n", "san/orphan.txt": b"o\n"})

            art_a = ReleaseRunArtifacts(
                output_dir=d_a,
                records=(
                    ReleaseRunRecord("san/file_001.csv", "row:1", "PSN-ABCD2345", {"visit_date": "2025-03-01"}),
                    ReleaseRunRecord("san/orphan.txt", "doc:0", None),
                    ReleaseRunRecord("san/dup.txt", "doc:0", "PSN-ABCD2345"),
                    ReleaseRunRecord("san/dup.txt", "doc:0", "PSN-ABCD2345"),
                ),
                source_to_output={
                    ("raw/Ramesh_Sharma.csv", "row:1"): ("san/file_001.csv", "row:1"),
                    ("raw/orphan.txt", "doc:0"): ("san/orphan.txt", "doc:0"),
                    ("raw/dup.txt", "doc:0"): ("san/dup.txt", "doc:0"),
                },
            )
            art_b = ReleaseRunArtifacts(
                output_dir=d_b,
                records=(
                    ReleaseRunRecord("san/file_001.csv", "row:1", "PSN-WXYZ6789", {"visit_date": "2025-05-12"}),
                    ReleaseRunRecord("san/orphan.txt", "doc:0", None),
                ),
                renamed_files={
                    "raw/Ramesh_Sharma.csv": "san/file_001.csv",
                    "raw/orphan.txt": "san/orphan.txt",
                },
            )

            a6_res = evaluate_a6_release(cohort, art_a)
            self.assertEqual(a6_res.patients_evaluated, 1)
            self.assertEqual(a6_res.patients_flagged, 0)
            self.assertEqual(a6_res.unlinked_records_excluded, 1)
            self.assertEqual(a6_res.missing_or_ambiguous_locators, 1)

            a7_res = evaluate_a7_rotation(cohort, art_a, art_b)
            self.assertFalse(a7_res.coverage.adequate)
            self.assertGreaterEqual(a7_res.dropped_or_unresolved, 2)
            self.assertEqual(a7_res.psn_comparable, 1)
            self.assertEqual(a7_res.psn_reused, 0)
            self.assertEqual(a7_res.date_comparable, 1)
            self.assertEqual(a7_res.date_reused, 0)

    def test_summary_omits_secrets_identifiers_paths_and_patient_rows(self) -> None:
        cohort = build_default_synthetic_cohort()
        runner = build_release_runner_double()
        secret_a = "SUPER-SECRET-ALPHA-999"
        secret_b = "SUPER-SECRET-BRAVO-888"
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_v10_verification(
                cohort,
                runner,
                Path(tmp),
                secret_a=secret_a,
                secret_b=secret_b,
            )
            text = render_v10_summary(summary)
            for forbidden in (
                secret_a,
                secret_b,
                "Ramesh Sharma",
                "Sunita Devi",
                "SNJ/26/019672",
                "9811100001",
                "fx_p1",
                "fx_p2",
                "reg_01.csv",
            ):
                self.assertNotIn(forbidden, text)


class TestV10Phase2A6ControlsAndRelease(unittest.TestCase):
    def test_unsafe_positive_controls_and_safe_negative_controls(self) -> None:
        p1 = SyntheticPatient(
            "p1",
            uhids=("SNJ/26/019672",),
            aadhaar=("1234 5678 9012",),
            phones=("9811100001",),
            abhas=("12-3456-7890-1234",),
            latin_names=("Ramesh Sharma",),
        )
        p2 = SyntheticPatient("p2", uhids=("SNJ/26/099999",), phones=("9822200002",))
        cohort = SyntheticCohort(patients=(p1, p2), records=())

        md5_p1 = hashlib.md5(b"SNJ/26/019672").hexdigest()[:10]
        sha1_p1 = hashlib.sha1(b"9811100001").hexdigest()[:8]
        sha256_p1 = hashlib.sha256(b"Ramesh Sharma").hexdigest()[:12]
        sha512_p1 = hashlib.sha512(b"123456789012").hexdigest()[:8]
        b64_p1 = base64.b64encode(b"SNJ/26/019672").decode("ascii")
        rev_p1 = "Ramesh Sharma"[::-1]
        md5_p2 = hashlib.md5(b"SNJ/26/099999").hexdigest()[:10]

        vectors = (
            A6ControlVector("raw", "p1", "SNJ/26/019672", True, "raw"),
            A6ControlVector("uhid_core", "p1", "PSN-019672XY", True, "uhid_core"),
            A6ControlVector("phone", "p1", "PSN-9811100001", True, "phone"),
            A6ControlVector("md5", "p1", f"PSN-{md5_p1}", True, "md5"),
            A6ControlVector("sha1", "p1", sha1_p1, True, "sha1"),
            A6ControlVector("sha256", "p1", f"PSN-{sha256_p1}", True, "sha256"),
            A6ControlVector("sha512", "p1", sha512_p1, True, "sha512"),
            A6ControlVector("b64", "p1", f"PSN-{b64_p1}", True, "b64"),
            A6ControlVector("rev", "p1", rev_p1, True, "rev"),
            A6ControlVector("safe_keyed", "p1", "PSN-37K9M2P4", False, "safe_keyed"),
            A6ControlVector("safe_ext", "p1", "PSN-37K9M2P4C", False, "safe_ext"),
            # Another patient's hash must NOT count as recovery for p1.
            A6ControlVector("cross_patient", "p1", f"PSN-{md5_p2}", False, "cross_patient"),
        )
        ctrl = evaluate_a6_controls(cohort, vectors)
        self.assertTrue(ctrl.all_passed)
        self.assertEqual(ctrl.positive_total, 9)
        self.assertEqual(ctrl.negative_total, 3)

    def test_repeated_records_count_once_and_empty_denominator_unavailable(self) -> None:
        p1 = SyntheticPatient("p1", uhids=("SNJ/26/019672",))
        r1 = SyntheticRecordSpec("r1", "p1", "f1.csv", "row:1", "csv", {})
        r2 = SyntheticRecordSpec("r2", "p1", "f2.csv", "row:1", "csv", {})
        cohort = SyntheticCohort(patients=(p1,), records=(r1, r2))

        with tempfile.TemporaryDirectory() as tmp:
            art = ReleaseRunArtifacts(
                output_dir=Path(tmp),
                records=(
                    ReleaseRunRecord("f1.csv", "row:1", "PSN-019672ZZ"),
                    ReleaseRunRecord("f2.csv", "row:1", "PSN-019672ZZ"),
                ),
            )
            res = evaluate_a6_release(cohort, art)
            self.assertEqual(res.patients_evaluated, 1)
            self.assertEqual(res.patients_flagged, 1)
            self.assertEqual(res.rate, 1.0)

            # Empty pseudonym denominator (only null records) -> unavailable (None).
            art_null = ReleaseRunArtifacts(
                output_dir=Path(tmp),
                records=(
                    ReleaseRunRecord("f1.csv", "row:1", None),
                    ReleaseRunRecord("f2.csv", "row:1", None),
                ),
            )
            res_null = evaluate_a6_release(cohort, art_null)
            self.assertEqual(res_null.patients_evaluated, 0)
            self.assertIsNone(res_null.rate)
            self.assertFalse(res_null.available)


class TestV10Phase3DeterminismAndA7(unittest.TestCase):
    def test_byte_change_and_missing_file_fail_determinism(self) -> None:
        cohort = build_default_synthetic_cohort()
        with tempfile.TemporaryDirectory() as tmp:
            d1 = Path(tmp) / "d1"
            d2 = Path(tmp) / "d2"
            d3 = Path(tmp) / "d3"
            _write_release_files(d1, {"a.txt": b"hello"}, b"r\n")
            _write_release_files(d2, {"a.txt": b"world"}, b"r\n")
            _write_release_files(d3, {}, b"r\n")

            art1 = ReleaseRunArtifacts(d1, ())
            art2 = ReleaseRunArtifacts(d2, ())
            art3 = ReleaseRunArtifacts(d3, ())

            det_byte = verify_same_secret_determinism(cohort, art1, art2)
            self.assertFalse(det_byte.passed)
            self.assertIn("sanitized_file_bytes_mismatch", det_byte.failure_reasons)

            det_miss = verify_same_secret_determinism(cohort, art1, art3)
            self.assertFalse(det_miss.passed)
            self.assertIn("sanitized_inventory_mismatch", det_miss.failure_reasons)

    def test_a7_max_rate_formatting_and_coincidence(self) -> None:
        # 4 comparable patients (1 reused PSN -> 1/4 = 0.25)
        # 10 comparable events (1 coinciding date -> 1/10 = 0.10) -> A7 = 0.25
        patients = tuple(SyntheticPatient(f"p{i}") for i in range(4))
        records: list[SyntheticRecordSpec] = []
        recs_a: list[ReleaseRunRecord] = []
        recs_b: list[ReleaseRunRecord] = []

        # Distribute 10 events across the 4 patients (3, 3, 2, 2).
        event_counts = [3, 3, 2, 2]
        ev_idx = 0
        for p_idx, count in enumerate(event_counts):
            pid = f"p{p_idx}"
            for v in range(count):
                rec_id = f"r_{p_idx}_{v}"
                in_file = f"f_{p_idx}_{v}.csv"
                loc = "row:1"
                ev = SyntheticEvent(
                    event_id=f"ev_{ev_idx}",
                    patient_id=pid,
                    input_file=in_file,
                    loc=loc,
                    date_field="d",
                    iso_date="2025-01-01",
                )
                records.append(
                    SyntheticRecordSpec(rec_id, pid, in_file, loc, "csv", {}, events=(ev,))
                )
                psn_a = f"PSN-AAAA000{p_idx}"
                psn_b = psn_a if p_idx == 0 else f"PSN-BBBB000{p_idx}"
                # Same DD-MM-YYYY formatting in both runs, but actual date differs except ev_idx == 0.
                date_a = "15-03-2025"
                date_b = "15-03-2025" if ev_idx == 0 else f"{16 + ev_idx:02d}-04-2025"
                recs_a.append(ReleaseRunRecord(in_file, loc, psn_a, {"d": date_a}))
                recs_b.append(ReleaseRunRecord(in_file, loc, psn_b, {"d": date_b}))
                ev_idx += 1

        cohort = SyntheticCohort(patients=patients, records=tuple(records))
        with tempfile.TemporaryDirectory() as tmp:
            art_a = ReleaseRunArtifacts(Path(tmp) / "a", tuple(recs_a))
            art_b = ReleaseRunArtifacts(Path(tmp) / "b", tuple(recs_b))
            res = evaluate_a7_rotation(cohort, art_a, art_b)
            self.assertEqual(res.psn_reused, 1)
            self.assertEqual(res.psn_comparable, 4)
            self.assertAlmostEqual(res.psn_rate or 0.0, 0.25)
            self.assertEqual(res.date_reused, 1)
            self.assertEqual(res.date_comparable, 10)
            self.assertAlmostEqual(res.date_rate or 0.0, 0.10)
            self.assertAlmostEqual(res.a7_rate or 0.0, 0.25)
            self.assertFalse(res.systematic_date_reuse)


class TestV10Phase4EndToEndRunnerDouble(unittest.TestCase):
    def test_end_to_end_three_runs_with_linkage_resolve(self) -> None:
        cohort = build_default_synthetic_cohort()
        runner = build_release_runner_double()
        cfg = FrozenReleaseConfig()
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_v10_verification(cohort, runner, Path(tmp), config=cfg)
            self.assertTrue(summary.determinism.passed, summary.determinism.failure_reasons)
            self.assertGreaterEqual(summary.determinism.intervals_checked, 1)
            self.assertTrue(summary.a6_controls.all_passed)
            self.assertEqual(summary.a6_release.patients_evaluated, 2)
            self.assertEqual(summary.a6_release.patients_flagged, 0)
            self.assertEqual(summary.a6_release.rate, 0.0)
            self.assertEqual(summary.a6_release.unlinked_records_excluded, 1)
            self.assertEqual(summary.a7_rotation.psn_comparable, 2)
            self.assertEqual(summary.a7_rotation.psn_reused, 0)
            self.assertTrue(summary.a7_rotation.coverage.adequate)
            self.assertFalse(summary.real_runner_wired)
            self.assertIn("real_keys_a3", summary.pending_checks)


if __name__ == "__main__":
    unittest.main()
