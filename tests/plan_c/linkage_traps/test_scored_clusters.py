"""Unit tests for complete-linkage scored clustering and E6 sweep (Plan C V5 Phases 4-5)."""

from __future__ import annotations

import hashlib
import random
import unittest

from kavachmed.evidence import Evidence
from kavachmed.indic import normalize_name
from kavachmed.linkage import resolve
from tools.link_eval import compute_b3_metrics, compute_parity_metrics


class _StubKeys:
    def keyed(self, purpose: str, *fields: str) -> bytes:
        return hashlib.sha256(f"{purpose}|{'|'.join(fields)}".encode("utf-8")).digest()


def _ev(
    rec_id: str,
    *,
    name: str = "",
    kin: str = "",
    sex: str | None = None,
    birth_years: tuple[int, int] | None = None,
    ids: dict[str, list[str]] | None = None,
    phones: list[str] | None = None,
    district: str | None = None,
) -> Evidence:
    geo = {"district_code": district} if district else {}
    return Evidence(
        key=(f"{rec_id}.json", "/0"),
        names=normalize_name(name) if name else [],
        kin=normalize_name(kin) if kin else [],
        sex=sex,
        birth_years=birth_years,
        ids=ids or {},
        phones=phones or [],
        geo=geo,
    )


class TestScoredClusteringAndE6(unittest.TestCase):
    def test_cross_script_corroborated_merge_and_no_corrob_abstention(self) -> None:
        # Cross-script Arif with corroborating birth_years + relative merges
        a = _ev("a", name="Arif Khan", kin="Salim", sex="M", birth_years=(1980, 1980), ids={"aadhaar": ["111122223333"]})
        b = _ev("b", name="Arif Khan", kin="Salim", sex="M", birth_years=(1980, 1980), ids={"aadhaar": ["111122223333"]})
        c = _ev("c", name="आरिफ", kin="सलीम", sex="M", birth_years=(1980, 1981))

        # Two records with name-only + sex (no independent corroboration beyond name+sex? Wait: sex is not enough if threshold not met or corroboration checked)
        n1 = _ev("n1", name="Mohan", sex="M")
        n2 = _ev("n2", name="Mohan", sex="M")

        digests = {f"{x}.json": f"{i:032x}" for i, x in enumerate(["a", "b", "c", "n1", "n2"], 1)}
        res = resolve([a, b, c, n1, n2], _StubKeys(), file_digests=digests)
        self.assertIsNotNone(res.patient_of[a.key])
        self.assertEqual(res.patient_of[a.key], res.patient_of[b.key])
        self.assertEqual(res.patient_of[a.key], res.patient_of[c.key])
        self.assertIsNone(res.patient_of[n1.key])
        self.assertIsNone(res.patient_of[n2.key])

    def test_ambiguous_relative_note_and_vetoed_chain_abstain(self) -> None:
        # Ambiguous note compatible with two distinct patients stays null
        p1 = _ev("p1", name="Ramesh", kin="Suresh", sex="M", birth_years=(1975, 1975), district="D1", ids={"aadhaar": ["100000000001"]})
        p2 = _ev("p2", name="Ramesh", kin="Suresh", sex="M", birth_years=(1976, 1976), district="D1", ids={"aadhaar": ["200000000002"]})
        note = _ev("note", name="Ramesh", kin="Suresh", sex="M", birth_years=(1975, 1976), district="D1")

        # Chain A-B and B-C pass, but A-C has birth-year veto (>3 years) -> cannot become one cluster
        ca = _ev("ca", name="Kavita", kin="Raju", sex="F", birth_years=(1980, 1980), district="D2")
        cb = _ev("cb", name="Kavita", kin="Raju", sex="F", birth_years=(1983, 1983), district="D2")
        cc = _ev("cc", name="Kavita", kin="Raju", sex="F", birth_years=(1986, 1986), district="D2")

        digests = {f"{x}.json": f"{i:032x}" for i, x in enumerate(["p1", "p2", "note", "ca", "cb", "cc"], 1)}
        diag: dict[str, int] = {}
        res = resolve([p1, p2, note, ca, cb, cc], _StubKeys(), file_digests=digests, diagnostics=diag)
        self.assertIsNone(res.patient_of[note.key])
        # A, B, C cannot all be in one cluster
        chain_psns = {res.patient_of[ca.key], res.patient_of[cb.key], res.patient_of[cc.key]} - {None}
        self.assertFalse(
            res.patient_of[ca.key] is not None
            and res.patient_of[ca.key] == res.patient_of[cb.key] == res.patient_of[cc.key]
        )

    def test_scored_only_cluster_fallback_collision_and_shuffle_invariance(self) -> None:
        # Scored-only cluster
        s1 = _ev("s1", name="Deepak", kin="Mahesh", sex="M", birth_years=(1990, 1990), district="D5")
        s2 = _ev("s2", name="दीपक", kin="महेश", sex="M", birth_years=(1990, 1991), district="D5")
        digests = {"s1.json": "aa" * 16, "s2.json": "bb" * 16}

        res1 = resolve([s1, s2], _StubKeys(), file_digests=digests)
        res2 = resolve([s2, s1], _StubKeys(), file_digests=digests)
        self.assertIsNotNone(res1.patient_of[s1.key])
        self.assertEqual(res1.patient_of[s1.key], res1.patient_of[s2.key])
        self.assertEqual(res1.patient_of, res2.patient_of)
        self.assertEqual(res1.shift_days, res2.shift_days)

        # Fallback-key collision: two distinct clusters with identical (name_skel, rel_skel, birth_years)
        c1a = _ev("c1a", name="Sunil", kin="Raj", sex="M", birth_years=(1985, 1985), phones=["9911001100"])
        c1b = _ev("c1b", name="Sunil", kin="Raj", sex="M", birth_years=(1985, 1985), phones=["9911001100"])
        c2a = _ev("c2a", name="Sunil", kin="Raj", sex="F", birth_years=(1985, 1985), phones=["9922002200"])
        c2b = _ev("c2b", name="Sunil", kin="Raj", sex="F", birth_years=(1985, 1985), phones=["9922002200"])
        d_col = {f"{k}.json": f"{i:032x}" for i, k in enumerate(["c1a", "c1b", "c2a", "c2b"], 1)}
        diag: dict[str, int] = {}
        res_col = resolve([c1a, c1b, c2a, c2b], _StubKeys(), file_digests=d_col, diagnostics=diag)
        self.assertGreaterEqual(diag.get("fallback_key_collisions", 0), 1)
        self.assertIsNone(res_col.patient_of[c1a.key])
        self.assertIsNone(res_col.patient_of[c2a.key])

    def test_e6_threshold_sweep_and_v4_comparison(self) -> None:
        # Multi-patient trap fixture: exact IDs + cross-script no-ID + homonyms
        records = [
            ("P1", _ev("r1", name="Arif Khan", kin="Salim", sex="M", birth_years=(1980, 1980), ids={"aadhaar": ["100000000011"]}, district="D1")),
            ("P1", _ev("r2", name="आरिफ", kin="सलीम", sex="M", birth_years=(1980, 1981), district="D1")),
            ("P2", _ev("r3", name="Meera Devi", kin="Gopal", sex="F", birth_years=(1972, 1972), phones=["9811100011"], district="D2")),
            ("P2", _ev("r4", name="मीरा", kin="गोपाल", sex="F", birth_years=(1972, 1972), phones=["9811100011"], district="D2")),
            ("P3", _ev("r5", name="Ramesh", kin="Hari", sex="M", birth_years=(1965, 1965), district="D3")),
            ("P4", _ev("r6", name="Ramesh", kin="Kishan", sex="M", birth_years=(1965, 1965), district="D3")),
        ]
        evs = [ev for _, ev in records]
        digests = {ev.key[0]: f"{i:032x}" for i, ev in enumerate(evs, 1)}

        v4_res = resolve(evs, _StubKeys(), policy={"mode": "exact_only"}, file_digests=digests)
        v4_units = [(pt, v4_res.patient_of[ev.key]) for pt, ev in records]
        v4_metrics = compute_parity_metrics(v4_units)

        sweep_results = []
        for tau_high in (2.0, 3.0, 4.0, 5.0, 6.0, 8.0):
            res = resolve(
                evs,
                _StubKeys(),
                policy={"tau_high_bits": tau_high, "tau_low_bits": 1.0},
                file_digests=digests,
            )
            units = [(pt, res.patient_of[ev.key]) for pt, ev in records]
            pm = compute_parity_metrics(units)
            b3 = compute_b3_metrics(units)
            sweep_results.append((tau_high, pm, b3))

        # V5 at default tau_high=4.0 beats V4 exact-only recall while maintaining precision=1.0
        best_4 = next(pm for th, pm, _ in sweep_results if th == 4.0)
        self.assertEqual(best_4.precision, 1.0)
        self.assertGreater(best_4.recall, v4_metrics.recall)
        self.assertGreater(best_4.f05, v4_metrics.f05)


if __name__ == "__main__":
    unittest.main()
