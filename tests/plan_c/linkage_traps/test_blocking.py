"""Unit tests for bounded candidate blocking (Plan C V5 Phase 2)."""

from __future__ import annotations

import random
import unittest

from kavachmed.evidence import Evidence
from kavachmed.indic import normalize_name
from kavachmed.linkage.blocking import generate_candidates


def _ev(
    rec_id: str,
    *,
    name: str = "",
    kin: str = "",
    sex: str | None = None,
    birth_years: tuple[int, int] | None = None,
    ids: dict[str, list[str]] | None = None,
    phones: list[str] | None = None,
) -> Evidence:
    return Evidence(
        key=(f"{rec_id}.json", "/0"),
        names=normalize_name(name) if name else [],
        kin=normalize_name(kin) if kin else [],
        sex=sex,
        birth_years=birth_years,
        ids=ids or {},
        phones=phones or [],
    )


class TestBlocking(unittest.TestCase):
    def test_four_block_types_decade_boundary_dedup_and_issuer_separation(self) -> None:
        # 1. strong_id block + issuer separation
        a1 = _ev("a1", ids={"uhid:h1": ["1001"]})
        a2 = _ev("a2", ids={"uhid:h1": ["1001"]}, phones=["+919876543210"])
        a3 = _ev("a3", ids={"uhid:h2": ["1001"]})  # different issuer!

        # 2. phone last 8 digits (also overlaps with a2 -> deduplication)
        p1 = _ev("p1", phones=["09876543210"])

        # 3. name_sex block (blank or generic produces no name block)
        n1 = _ev("n1", name="Ramesh Kumar", sex="M")
        n2 = _ev("n2", name="रमेश", sex="M")
        g1 = _ev("g1", name="Kumar", sex="M")  # generic only
        g2 = _ev("g2", name="Singh", sex="M")  # generic only

        # 4. rel_decade across decade boundary (1978, 1982) -> intersects 1970 and 1980
        r1 = _ev("r1", kin="Suresh", birth_years=(1978, 1982))
        r2 = _ev("r2", kin="सुरेश", birth_years=(1975, 1976))
        r3 = _ev("r3", kin="Suresh", birth_years=(1984, 1985))

        res = generate_candidates(
            [a1, a2, a3, p1, n1, n2, g1, g2, r1, r2, r3],
            contested_bridges=[(0, "uhid:h2", "1001")],
        )
        pairs = set(res.candidates)
        self.assertIn((a1.key, a2.key), pairs)
        self.assertNotIn((a1.key, a3.key), pairs)
        self.assertIn((a2.key, p1.key), pairs)
        self.assertIn((n1.key, n2.key), pairs)
        self.assertNotIn((g1.key, g2.key), pairs)
        self.assertIn((r1.key, r2.key), pairs)
        self.assertIn((r1.key, r3.key), pairs)
        # Deduplication: (a1, a2) counted once
        self.assertEqual(len(res.candidates), len(pairs))

    def test_oversized_shared_phone_reblocks_and_abstains_cleanly(self) -> None:
        # 12 records sharing one clinic phone, max_bucket_size=4
        shared = "9811122233"
        evs = []
        # 2 records have secondary birth_decade (1980) + name_sex
        evs.append(_ev("r00", name="Arif Khan", sex="M", birth_years=(1981, 1982), phones=[shared]))
        evs.append(_ev("r01", name="आरिफ", sex="M", birth_years=(1983, 1984), phones=[shared]))
        # 10 records with NO secondary evidence -> unresolved oversized block
        for i in range(2, 12):
            evs.append(_ev(f"r{i:02d}", phones=[shared]))

        res = generate_candidates(evs, policy={"max_bucket_size": 4})
        pairs = set(res.candidates)
        # Useful secondary block + name_sex still produces (r00, r01)
        self.assertIn((evs[0].key, evs[1].key), pairs)
        # No unbounded Cartesian expansion (12*11/2 = 66)
        self.assertLess(len(pairs), 10)
        self.assertGreaterEqual(res.oversized_blocks_abstained, 1)
        self.assertGreater(res.skipped_by_cap, 0)

    def test_shuffling_determinism_and_pair_budget(self) -> None:
        evs = [
            _ev(f"p{i}", name="Anil", sex="M", ids={"aadhaar": [f"9000{i:04d}"] if i < 2 else []})
            for i in range(6)
        ]
        res1 = generate_candidates(evs, policy={"pair_budget": 4})
        shuffled = list(evs)
        random.Random(42).shuffle(shuffled)
        res2 = generate_candidates(shuffled, policy={"pair_budget": 4})

        self.assertEqual(res1.candidates, res2.candidates)
        self.assertEqual(res1.skipped_by_cap, res2.skipped_by_cap)
        self.assertEqual(len(res1.candidates), 4)
        # Trusted strong_id candidate is emitted first
        self.assertEqual(res1.candidates[0], (evs[0].key, evs[1].key))


if __name__ == "__main__":
    unittest.main()
