import hashlib
import hmac
import unittest
from datetime import date, timedelta

from kavachmed.evidence import Field, Record, Span, extract
from kavachmed.linkage import resolve


class _SpyKeys:
    def __init__(self, secret: bytes = b"e7-interval-secret") -> None:
        self._secret = secret

    def keyed(self, purpose: str, *fields: str) -> bytes:
        msg = b"".join(f"{len(f)}:{f}".encode("utf-8") for f in fields)
        return hmac.new(self._secret, purpose.encode("utf-8") + b":" + msg, hashlib.sha256).digest()


class _StaticDateProvider:
    def __init__(self, by_key: dict[tuple[str, str], list[dict[str, str]]]) -> None:
        self._by_key = by_key

    def find(self, rec: Record) -> list[dict[str, str]]:
        return self._by_key.get(rec.key, [])


class TestTemporalIntervals(unittest.TestCase):
    def test_cross_format_nine_day_stay_and_year_boundary(self) -> None:
        # CSV registration + FHIR encounter linked by UHID; discharge note has no UHID
        # (left patient=None in V5) but shares name + overlapping birth year.
        # Unrelated homonym in another file must stay in a separate family.
        r_csv = Record(
            key=("reg/adm.csv", "row:0"),
            fields=[
                Field(0, "name", "Ramesh Kumar"),
                Field(1, "uhid", "SNJ/26/019672"),
                Field(2, "dob", "1980-04-12"),
                Field(3, "sex", "Male"),
                Field(4, "adm_date", "2025-12-28"),
            ],
            meta={"hospital": "SNJ"},
        )
        r_fhir = Record(
            key=("fhir/enc.json", "/entry/0"),
            fields=[
                Field(0, "name", "Ramesh Kumar"),
                Field(1, "uhid", "SNJ/26/019672"),
                Field(2, "dob", "1980-04-12"),
                Field(3, "sex", "Male"),
                Field(4, "obs_date", "2025-12-31"),
            ],
            meta={"hospital": "SNJ"},
        )
        r_discharge = Record(
            key=("notes/discharge.txt", "doc"),
            fields=[
                Field(0, "text", "Pt श्री रमेश कुमार, Age 45 yrs, discharged 2026-01-06"),
            ],
            meta={"hospital": "SNJ", "doc_year": 2025},
        )
        r_homonym = Record(
            key=("notes/other.txt", "doc"),
            fields=[
                Field(0, "text", "Pt Ramesh Kumar, Age 22 yrs, seen 2026-01-06"),
            ],
            meta={"hospital": "APO", "doc_year": 2026},
        )

        ev_csv = extract(
            r_csv,
            [
                Span(r_csv.key, 0, 0, 12, "PATIENT_NAME", "patient", 1.0, "labels"),
                Span(r_csv.key, 1, 0, 13, "UHID", "patient", 1.0, "ids"),
                Span(r_csv.key, 2, 0, 10, "DOB", "patient", 1.0, "dates"),
            ],
        )
        ev_fhir = extract(
            r_fhir,
            [
                Span(r_fhir.key, 0, 0, 12, "PATIENT_NAME", "patient", 1.0, "labels"),
                Span(r_fhir.key, 1, 0, 13, "UHID", "patient", 1.0, "ids"),
                Span(r_fhir.key, 2, 0, 10, "DOB", "patient", 1.0, "dates"),
            ],
        )
        dp = _StaticDateProvider(
            {
                r_discharge.key: [{"iso": "2026-01-06", "role": "discharge"}],
                r_homonym.key: [{"iso": "2026-01-06", "role": "encounter"}],
            }
        )
        ev_dis = extract(
            r_discharge,
            [
                Span(r_discharge.key, 0, 3, 18, "PATIENT_NAME", "patient", 1.0, "labels"),
                Span(r_discharge.key, 0, 24, 26, "AGE", "patient", 1.0, "rules"),
            ],
            date_provider=dp,
        )
        ev_hom = extract(
            r_homonym,
            [
                Span(r_homonym.key, 0, 3, 15, "PATIENT_NAME", "patient", 1.0, "labels"),
                Span(r_homonym.key, 0, 21, 23, "AGE", "patient", 1.0, "rules"),
            ],
            date_provider=dp,
        )

        evidences = [ev_csv, ev_fhir, ev_dis, ev_hom]
        digests = {
            "notes/discharge.txt": "digest_dis",
            "notes/other.txt": "digest_other",
        }

        res_single = resolve(
            evidences,
            _SpyKeys(),
            policy={"mode": "scored", "family_mode": "single_graph", "tau_high_bits": 6.0, "tau_low_bits": 2.0},
            file_digests=digests,
        )
        res_family = resolve(
            evidences,
            _SpyKeys(),
            policy={"mode": "scored", "family_mode": "temporal_family", "tau_high_bits": 6.0, "tau_low_bits": 2.0},
            file_digests=digests,
        )

        # Identical identity assignments across modes; unlinked discharge stays patient=None.
        self.assertEqual(res_family.patient_of, res_single.patient_of)
        self.assertEqual(res_family.cluster_of, res_single.cluster_of)
        self.assertIsNotNone(res_family.patient_of[r_csv.key])
        self.assertEqual(res_family.patient_of[r_csv.key], res_family.patient_of[r_fhir.key])
        self.assertIsNone(res_family.patient_of[r_discharge.key])

        adm_date = date(2025, 12, 28)
        obs_date = date(2025, 12, 31)
        dis_date = date(2026, 1, 6)
        self.assertEqual((dis_date - adm_date).days, 9)

        # Single-graph gives different offsets to the strict cluster and the unlinked discharge note.
        s_adm = res_single.shift_days[res_single.family_of[r_csv.key]]
        s_dis = res_single.shift_days[res_single.family_of[r_discharge.key]]
        self.assertNotEqual(s_adm, s_dis)
        self.assertNotEqual(
            ((dis_date + timedelta(days=s_dis)) - (adm_date + timedelta(days=s_adm))).days,
            9,
        )

        # Temporal-family places discharge into the strict cluster's family, preserving the 9-day stay.
        self.assertEqual(res_family.family_of[r_discharge.key], res_family.family_of[r_csv.key])
        self.assertNotEqual(res_family.family_of[r_homonym.key], res_family.family_of[r_csv.key])

        f_adm = res_family.shift_days[res_family.family_of[r_csv.key]]
        f_obs = res_family.shift_days[res_family.family_of[r_fhir.key]]
        f_dis = res_family.shift_days[res_family.family_of[r_discharge.key]]
        self.assertEqual(
            ((dis_date + timedelta(days=f_dis)) - (adm_date + timedelta(days=f_adm))).days,
            9,
        )
        # Within-record / intra-cluster gap (adm -> obs = 3 days) preserved in both modes.
        self.assertEqual(
            ((obs_date + timedelta(days=f_obs)) - (adm_date + timedelta(days=f_adm))).days,
            3,
        )
