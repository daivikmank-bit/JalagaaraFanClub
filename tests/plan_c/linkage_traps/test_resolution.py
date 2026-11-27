import hashlib
import hmac
import unittest
from datetime import date, timedelta

from kavachmed.evidence import Field, Record, Span, extract
from kavachmed.linkage import LinkageError, Resolution, resolve
from kavachmed.linkage.pseudo import derive_shift_days
from tools.link_eval import compute_parity_metrics


class _SpyKeys:
    def __init__(self, secret: bytes = b"res-secret", fixed: dict[tuple[str, ...], bytes] | None = None) -> None:
        self._secret = secret
        self._fixed = fixed or {}
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def keyed(self, purpose: str, *fields: str) -> bytes:
        self.calls.append((purpose, fields))
        key = (purpose, *fields)
        if key in self._fixed:
            return self._fixed[key]
        msg = b"".join(f"{len(f)}:{f}".encode("utf-8") for f in fields)
        return hmac.new(self._secret, purpose.encode("utf-8") + b":" + msg, hashlib.sha256).digest()


class TestResolutionAndBaseline(unittest.TestCase):
    def test_empty_missing_digest_duplicate_bytes_and_shift_bounds(self) -> None:
        keys = _SpyKeys()
        empty = resolve([], keys)
        self.assertEqual(empty, Resolution({}, {}, {}, {}))

        # Shift boundary values: 0 -> -365, 335 -> -30, 336 -> 30, 671 -> 365
        for idx, expected_offset in [(0, -365), (335, -30), (336, 30), (671, 365)]:
            k = _SpyKeys(fixed={("shift", "x"): idx.to_bytes(8, "big")})
            self.assertEqual(derive_shift_days(k, "x"), expected_offset)

        # Missing digest for orphan raises safe error.
        orphan_a = extract(Record(("dup/a.txt", "doc"), [Field(0, "t", "hello")]), [])
        orphan_b = extract(Record(("dup/b.txt", "doc"), [Field(0, "t", "hello")]), [])
        with self.assertRaises(LinkageError) as ctx:
            resolve([orphan_a], keys, file_digests={})
        self.assertEqual(ctx.exception.code, "missing_file_digest")

        # Duplicate-byte files at distinct paths keep separate family_of, same shift_days.
        digests = {"dup/a.txt": "sha256_same", "dup/b.txt": "sha256_same"}
        res = resolve([orphan_b, orphan_a], keys, file_digests=digests)
        self.assertIsNone(res.patient_of[orphan_a.key])
        self.assertIsNone(res.patient_of[orphan_b.key])
        self.assertNotEqual(res.family_of[orphan_a.key], res.family_of[orphan_b.key])
        self.assertEqual(
            res.shift_days[res.family_of[orphan_a.key]],
            res.shift_days[res.family_of[orphan_b.key]],
        )
        # Digest and locator reach Keys.keyed("shift", "orphan", digest, loc) as separate args.
        self.assertIn(("shift", ("orphan", "sha256_same", "doc")), keys.calls)

    def test_end_to_end_v3_to_v4_baseline_and_date_interval(self) -> None:
        # 1) Linked registration CSV + discharge TXT (9-day stay)
        r_csv = Record(
            key=("reg/r1.csv", "row:0"),
            fields=[
                Field(0, "name", "Ramesh Kumar"),
                Field(1, "uhid", "SNJ/26/019672"),
                Field(2, "adm", "2026-03-01"),
            ],
            meta={"hospital": "SNJ"},
        )
        r_txt = Record(
            key=("notes/d1.txt", "doc"),
            fields=[Field(0, "text", "Pt रमेश कुमार UHID SNJ/26/019672 discharged 2026-03-10")],
            meta={"hospital": "SNJ"},
        )
        # 2) Unlinked shared-phone record (true pair missed in V4 -> recall 1/2, P=1, F0.5=5/6)
        r_shared_1 = Record(
            key=("reg/r2.csv", "row:0"),
            fields=[Field(0, "name", "Asha Devi"), Field(1, "phone", "9876543210")],
        )
        r_shared_2 = Record(
            key=("notes/d2.txt", "doc"),
            fields=[Field(0, "text", "Asha"), Field(1, "phone", "9876543210")],
        )
        # 3) Contested ID record pair (not merged)
        r_con_1 = Record(
            key=("reg/c1.csv", "row:0"),
            fields=[Field(0, "sex", "Male"), Field(1, "aadhaar", "123456789012")],
        )
        r_con_2 = Record(
            key=("reg/c2.csv", "row:0"),
            fields=[Field(0, "sex", "Female"), Field(1, "aadhaar", "123456789012")],
        )

        evidences = [
            extract(
                r_csv,
                [
                    Span(r_csv.key, 0, 0, 12, "PATIENT_NAME", "patient", 1.0, "labels"),
                    Span(r_csv.key, 1, 0, 13, "UHID", "patient", 1.0, "ids"),
                ],
            ),
            extract(
                r_txt,
                [
                    Span(r_txt.key, 0, 3, 13, "PATIENT_NAME", "patient", 1.0, "labels"),
                    Span(r_txt.key, 0, 19, 32, "UHID", "patient", 1.0, "ids"),
                ],
            ),
            extract(
                r_shared_1,
                [
                    Span(r_shared_1.key, 0, 0, 9, "PATIENT_NAME", "patient", 1.0, "labels"),
                    Span(r_shared_1.key, 1, 0, 10, "PHONE", "patient", 1.0, "ids"),
                ],
            ),
            extract(
                r_shared_2,
                [
                    Span(r_shared_2.key, 0, 0, 4, "PATIENT_NAME", "patient", 1.0, "labels"),
                    Span(r_shared_2.key, 1, 0, 10, "PHONE", "patient", 1.0, "ids"),
                ],
            ),
            extract(r_con_1, [Span(r_con_1.key, 1, 0, 12, "AADHAAR", "patient", 1.0, "ids")]),
            extract(r_con_2, [Span(r_con_2.key, 1, 0, 12, "AADHAAR", "patient", 1.0, "ids")]),
        ]

        digests = {
            "reg/r2.csv": "d_r2",
            "notes/d2.txt": "d_d2",
            "reg/c1.csv": "d_c1",
            "reg/c2.csv": "d_c2",
        }
        keys = _SpyKeys(secret=b"baseline-secret")
        res = resolve(
            evidences,
            keys,
            policy={"mode": "exact_only"},
            file_digests=digests,
        )
        res_rev = resolve(
            list(reversed(evidences)),
            _SpyKeys(secret=b"baseline-secret"),
            policy={"mode": "exact_only"},
            file_digests=digests,
        )
        self.assertEqual(res, res_rev)

        # Linked CSV and TXT share PSN and nonzero bounded shift, preserving 9-day stay.
        self.assertIsNotNone(res.patient_of[r_csv.key])
        self.assertEqual(res.patient_of[r_csv.key], res.patient_of[r_txt.key])
        self.assertEqual(res.family_of[r_csv.key], res.family_of[r_txt.key])
        delta = res.shift_days[res.family_of[r_csv.key]]
        self.assertTrue(30 <= abs(delta) <= 365)
        d1 = date(2026, 3, 1) + timedelta(days=delta)
        d2 = date(2026, 3, 10) + timedelta(days=delta)
        self.assertEqual((d2 - d1).days, 9)

        # Contested IDs stay unlinked.
        self.assertIsNone(res.patient_of[r_con_1.key])
        self.assertIsNone(res.patient_of[r_con_2.key])

        # V2 parity evaluation: 1 TP (r1), 0 FP, 1 FN (r2) -> P=1.0, R=0.5, F0.5=5/6.
        truth = {
            r_csv.key: "PAT-1",
            r_txt.key: "PAT-1",
            r_shared_1.key: "PAT-2",
            r_shared_2.key: "PAT-2",
            r_con_1.key: "PAT-3",
            r_con_2.key: "PAT-4",
        }
        units = [(truth[k], res.patient_of[k]) for k in sorted(truth)]
        parity = compute_parity_metrics(units)
        self.assertAlmostEqual(parity.precision, 1.0)
        self.assertAlmostEqual(parity.recall, 0.5)
        self.assertAlmostEqual(parity.f05, 5.0 / 6.0)
        self.assertEqual(parity.merged_clusters, 0)

        # Key rotation changes shift offsets.
        res_rot = resolve(evidences, _SpyKeys(secret=b"rotated-secret"), file_digests=digests)
        self.assertNotEqual(res.shift_days, res_rot.shift_days)


if __name__ == "__main__":
    unittest.main()
