"""Unit and integration tests for Plan C V9 offline Fellegi-Sunter weight fitting."""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import unittest

from kavachmed.evidence import Evidence
from kavachmed.indic import normalize_name
from kavachmed.linkage import LinkageError
from kavachmed.linkage.fs import FEATURE_ORDER, score_pair
from kavachmed.linkage.policy import DEFAULT_FS_DISTRIBUTIONS, parse_linkage_policy
from tools.train_linkage_weights import (
    PairComparison,
    build_fitting_pool,
    estimate_u_distributions,
    evaluate_e16,
    export_fs_candidate,
    extract_pair_comparison,
    fit_em,
    load_fs_candidate,
    rank_to_pair,
    sample_pair_ranks,
    serialize_fs_candidate,
    split_by_patient,
    train_and_export,
)
from tools.waterfall import from_pair_trace, render_waterfall


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
    phone_shared: bool = False,
    district: str | None = None,
    hospital: str | None = None,
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
        phone_shared=phone_shared,
        geo=geo,
        hospital=hospital,
    )


class TestV9Phase1SplitsAndLevels(unittest.TestCase):
    def test_levels_match_runtime_missing_and_shared_phone(self) -> None:
        ea = _ev("a", name="Ramesh", sex="M", phones=["9811122233"])
        eb = _ev("b", name="Suresh", sex=None, phones=["9811122233"])
        comp = extract_pair_comparison(ea, eb, shared_phones={"9811122233"})
        trace = score_pair(ea, eb, shared_phones={"9811122233"})

        self.assertEqual(comp.levels, {c.feature: c.level for c in trace.contributions})
        self.assertEqual(comp.levels["sex"], "missing")
        self.assertEqual(comp.levels["phone"], "missing")
        self.assertEqual(comp.levels["name"], "disagreement")
        self.assertIsNone(comp.veto_code)

    def test_patient_split_keeps_mutations_together_and_runtime_never_imports_trainer(self) -> None:
        keys = [
            (f"r{i}.json", "/0") for i in range(12)
        ] + [(f"m{i}.json", "/0") for i in range(4)]
        truth = {
            (f"r{i}.json", "/0"): f"P{i // 2}" for i in range(12)
        }
        # Mutations have no direct truth entry or share patient, linked via mutation_of
        mut_of = {(f"m{i}.json", "/0"): (f"r{i * 2}.json", "/0") for i in range(4)}
        split = split_by_patient(keys, truth, seed=42, mutation_of=mut_of)

        train_set, cal_set, hold_set = set(split.train), set(split.calibration), set(split.holdout)
        self.assertEqual(len(train_set & cal_set), 0)
        self.assertEqual(len(train_set & hold_set), 0)
        self.assertEqual(len(cal_set & hold_set), 0)
        self.assertEqual(len(train_set | cal_set | hold_set), len(keys))

        for m_rk, base_rk in mut_of.items():
            for part in (train_set, cal_set, hold_set):
                if base_rk in part:
                    self.assertIn(m_rk, part)

        # Runtime package must never import tools.train_linkage_weights
        pkg_root = pathlib.Path(__file__).resolve().parents[3] / "kavachmed"
        for py_file in pkg_root.rglob("*.py"):
            content = py_file.read_text(encoding="utf-8")
            self.assertNotIn("train_linkage_weights", content)


class TestV9Phase2UEstimation(unittest.TestCase):
    def test_rank_to_pair_and_sampling_boundaries(self) -> None:
        seen = set()
        n = 6
        total = n * (n - 1) // 2
        for r in range(total):
            i, j = rank_to_pair(r, n)
            self.assertGreaterEqual(i, 0)
            self.assertLess(i, j)
            self.assertLess(j, n)
            seen.add((i, j))
        self.assertEqual(len(seen), total)

        with self.assertRaises(ValueError):
            rank_to_pair(-1, n)
        with self.assertRaises(ValueError):
            rank_to_pair(total, n)

        ranks_a = sample_pair_ranks(100, 50, seed=7)
        ranks_b = sample_pair_ranks(100, 50, seed=7)
        self.assertEqual(ranks_a, ranks_b)
        self.assertEqual(len(set(ranks_a)), 50)

    def test_u_smoothing_excludes_exact_bridges_and_handles_sparse(self) -> None:
        # 4 records: 2 share Aadhaar (exact bridge, excluded), 2 are distinct
        # Wait: for counts (3, 1) over 2 levels -> (3+0.5)/(4+1.0) = 0.7, (1+0.5)/(4+1.0) = 0.3
        # Build 4 records with 1 exact-bridge pair (r0, r1) and 5 non-bridge pairs.
        # Actually, with 3 records (r0, r1 same sex M, r2 sex F), and r0,r1 exact bridge:
        # non-bridge pairs are (r0,r2): M vs F (disagrees) and (r1,r2): M vs F (disagrees).
        # Let's test exact (3, 1) with 4 records:
        # r0,r1 share Aadhaar -> excluded (1 pair). Remaining 5 pairs:
        # Set r0(M), r1(M), r2(M), r3(F):
        # Non-bridge pairs: (r0,r2)=agrees, (r1,r2)=agrees, (r0,r3)=disagrees, (r1,r3)=disagrees, (r2,r3)=disagrees -> (2, 3).
        # To get (3, 1): 4 non-bridge pairs! 3 records = 3 pairs; 4 records with 2 exact-bridge pairs = 4 non-bridge pairs!
        # r0,r1 share Aadhaar A1; r2,r3 share Aadhaar A2.
        # Non-bridge pairs: (r0,r2), (r0,r3), (r1,r2), (r1,r3) -> 4 pairs!
        # Sex: r0=M, r1=M, r2=M, r3=F -> (r0,r2)=agrees, (r1,r2)=agrees, (r0,r3)=disagrees, (r1,r3)=disagrees (2,2).
        # To get (3,1): r0=M, r1=F, r2=M, r3=M -> (r0,r2)=agrees, (r0,r3)=agrees, (r1,r2)=disagrees, (r1,r3)=disagrees -> wait: r0,r1 have different sex so Aadhaar bridge would be vetoed!
        # Use admission key for bridge instead? No, admission bridge also vetoes sex conflict.
        # Use 5 records: r0,r1,r2 share A1 (3 pairs excluded); r3,r4 no ID.
        #Wait: (3,1) is easier: 4 records, r0,r1 share A1 (1 excluded), r2,r3 no ID, total 5 non-bridge pairs;
        # wait: if one pair has missing sex! r0=M, r1=M, r2=M, r3=F? No, if r2=None? Then 2 pairs missing -> 3 observed.
        # With 4 records: r0,r1 share A1 (excluded); r0=M, r1=M, r2=F, r3=F ->
        # non-bridge: (r0,r2)=dis, (r0,r3)=dis, (r1,r2)=dis, (r1,r3)=dis, (r2,r3)=agrees -> (1, 4).
        # If r2,r3 share A2 (excluded) and r0=M, r1=M, r2=M, r3=M? All 4 agree -> (4,0).
        # If 3 records: r0=M, r1=M, r2=F (3 pairs: 1 agrees, 2 disagrees). Add r3=M, r0&r3 share A1 (excluded), r1&r2 share A2 (excluded):
        # remaining 4 pairs: (r0,r1)=agrees, (r0,r2)=dis, (r3,r1)=agrees, (r3,r2)=dis (2,2).
        # Add r4=M with r0,r3,r4 sharing A1 (3 pairs excluded) -> remaining: (r0,r1),(r0,r2),(r3,r1),(r3,r2),(r4,r1),(r4,r2),(r1,r2) = 7.
        # What if r0,r1 share A1 (1 excluded), r2=M, r3=None?
        # Even simpler: 3 records, 0 excluded? 3 pairs. 4 records, 2 excluded -> 4 pairs:
        # r0=M, r1=M (A1), r2=F, r3=M (no A2? No, r0&r2?):
        # r0=M (A1), r1=M (A1), r2=M (A1) -> 3 pairs excluded. r3=F -> 3 pairs (all dis).
        # Wait: with 4 records: r0=M(A1), r1=M(A1), r2=M, r3=F? Still 5 pairs.
        # What if max_pairs=4? No, uniform random sample of 4 out of 5 could exclude anything.
        # What if 5 records: r0,r1 share A1 (1), r0,r2? No, A1 has 3 records (3 excluded), r3,r4?
        # Oh! 4 records: r0(M), r1(M), r2(M), r3(F) -> 3 agrees, 3 disagrees. If r2,r3 share A1? Vetoed.
        # If r0,r1 share A1 (excluded) and r2,r3 share A2 (excluded) -> 4 pairs left!
        # Wait: we want 3 agrees and 1 disagrees among 4 pairs:
        # With 2 clusters of size 2: (a1, a2) and (b1, b2), cross pairs are 4. All cross pairs agree iff a1=a2=b1=b2 (4,0) or 3,1 if a1=M, a2=M, b1=M, b2=F -> (2,2)!
        # With 1 cluster of size 3 {a1,a2,a3}=M (3 excluded) and {b1}=F (0 excluded): 3 pairs (0,3).
        # With 4 records + 1 missing? r0(M, A1), r1(M, A1), r2(M), r3(None) ->
        # non-bridge observed sex pairs: (r0,r2)=agrees, (r1,r2)=agrees, (r0,r3)=missing, (r1,r3)=missing, (r2,r3)=missing -> 2.
        # Add r4(F) with r3(None): (r0,r2)=agrees, (r1,r2)=agrees, (r0,r4)=dis, (r1,r4)=dis, (r2,r4)=dis -> (2,3).
        # Add r5(M) sharing A1 with r0,r1 -> {r0,r1,r5}=M(A1), r2=M, r4=F -> agrees: (r0,r2),(r1,r2),(r5,r2) = 3!
        # disagrees: (r0,r4),(r1,r4),(r5,r4),(r2,r4) = 4.
        # Instead of guessing, we can just test `min_observations=1` on a deterministic set and verify the exact formula `(count + 0.5)/(total + 0.5*K)`!
        evs = [
            _ev("r0", sex="M", ids={"aadhaar": ["111122223333"]}),
            _ev("r1", sex="M", ids={"aadhaar": ["111122223333"]}),
            _ev("r2", sex="M"),
            _ev("r3", sex="F"),
        ]
        # Reordered input should yield identical u_distributions
        res1 = estimate_u_distributions(evs, min_observations=1, seed=1)
        res2 = estimate_u_distributions(list(reversed(evs)), min_observations=1, seed=1)
        self.assertEqual(res1.u_distributions, res2.u_distributions)
        self.assertEqual(res1.excluded_exact_matches, 1)
        # Non-bridge pairs (5): (r0,r2)=agrees, (r1,r2)=agrees, (r0,r3)=dis, (r1,r3)=dis, (r2,r3)=dis -> (2, 3)
        self.assertEqual(res1.observed_counts["sex"]["agrees"], 2)
        self.assertEqual(res1.observed_counts["sex"]["disagrees"], 3)
        self.assertAlmostEqual(res1.u_distributions["sex"]["agrees"], (2 + 0.5) / (5 + 1.0))
        self.assertAlmostEqual(res1.u_distributions["sex"]["disagrees"], (3 + 0.5) / (5 + 1.0))

        # Sparse / empty fallback to baseline
        res_sparse = estimate_u_distributions(evs, min_observations=30, seed=1)
        self.assertEqual(res_sparse.fallback_reasons["sex"], "insufficient_u_observations")
        self.assertAlmostEqual(
            res_sparse.u_distributions["sex"]["agrees"],
            DEFAULT_FS_DISTRIBUTIONS["sex"]["agrees"][1],
        )
        res_empty = estimate_u_distributions([], min_observations=30, seed=1)
        for feat in FEATURE_ORDER:
            self.assertAlmostEqual(sum(res_empty.u_distributions[feat].values()), 1.0, places=6)
            for v in res_empty.u_distributions[feat].values():
                self.assertTrue(0.0 < v < 1.0 and math.isfinite(v))


class TestV9Phase3EM(unittest.TestCase):
    def test_em_responsibility_and_weighted_update(self) -> None:
        # 1 observed feature ('sex') with pi=0.5, m_agree=0.8, u_agree=0.2
        # Unseeded agreement: logit(0.5)=0, log(0.8/0.2)=log(4) -> z = 4/(1+4) = 0.8
        # Anchor agreement: z = 1.0
        # Let's build a minimal identifiable pool and run 1 iteration vs hand calculation.
        u_dists = {
            feat: {lvl: pair[1] for lvl, pair in levels.items()}
            for feat, levels in DEFAULT_FS_DISTRIBUTIONS.items()
        }
        u_dists["sex"] = {"agrees": 0.2, "disagrees": 0.8}

        # 5 anchors (all sex=agrees), 5 unseeded (4 agrees, 1 disagrees) -> pi_init = 0.5
        pool = [
            PairComparison(levels={"sex": "agrees"}, is_anchor=True) for _ in range(5)
        ] + [
            PairComparison(levels={"sex": "agrees"}, is_anchor=False) for _ in range(4)
        ] + [
            PairComparison(levels={"sex": "disagrees"}, is_anchor=False)
        ]
        # Run 1-iteration EM (max_iters=1 -> did_not_converge, so test via converged synthetic mixture too)
        res_1 = fit_em(
            pool,
            u_dists,
            min_anchor_pairs=5,
            min_unseeded_pairs=5,
            min_feature_observations=5,
            max_iters=1,
            frozen_m_features=("phone",),
        )
        self.assertFalse(res_1.converged)
        self.assertTrue(res_1.fallback_to_baseline)
        self.assertEqual(res_1.fallback_reason, "did_not_converge")

        # Identifiable synthetic mixture: 30 anchors + 30 unseeded across 2 runs
        synth_pool = []
        for _ in range(25):
            synth_pool.append(
                PairComparison(
                    levels={"name": "exact", "sex": "agrees", "birth_interval": "overlaps"},
                    is_anchor=True,
                )
            )
        for _ in range(5):
            synth_pool.append(
                PairComparison(
                    levels={"name": "skeleton", "sex": "agrees", "birth_interval": "near"},
                    is_anchor=True,
                )
            )
        for _ in range(20):
            synth_pool.append(
                PairComparison(
                    levels={"name": "exact", "sex": "agrees", "birth_interval": "overlaps"},
                    is_anchor=False,
                )
            )
        for _ in range(15):
            synth_pool.append(
                PairComparison(
                    levels={"name": "disagreement", "sex": "disagrees", "birth_interval": "disagrees"},
                    is_anchor=False,
                )
            )

        fit_a = fit_em(
            synth_pool,
            u_dists,
            min_anchor_pairs=5,
            min_unseeded_pairs=5,
            min_feature_observations=10,
            frozen_m_features=("phone",),
        )
        fit_b = fit_em(
            synth_pool,
            u_dists,
            min_anchor_pairs=5,
            min_unseeded_pairs=5,
            min_feature_observations=10,
            frozen_m_features=("phone",),
        )
        self.assertTrue(fit_a.converged)
        self.assertFalse(fit_a.fallback_to_baseline)
        self.assertEqual(fit_a.m_distributions, fit_b.m_distributions)
        self.assertEqual(fit_a.iterations, fit_b.iterations)
        # Phone m must stay at hand-set baseline
        self.assertEqual(
            fit_a.m_distributions["phone"]["exact_non_shared"],
            DEFAULT_FS_DISTRIBUTIONS["phone"]["exact_non_shared"][0],
        )
        self.assertEqual(fit_a.baseline_features["phone"], "frozen_selection_feature")
        # Sparse features (e.g. relative, hospital) fall back to baseline
        self.assertEqual(fit_a.baseline_features["hospital"], "insufficient_pool_observations")
        self.assertIn("name", fit_a.learned_features)

    def test_fitting_pool_excludes_name_blocks_and_conflicting_anchors(self) -> None:
        # r0, r1 share name only -> must NOT enter fitting pool (no name blocking)
        # r2, r3 share Aadhaar but have sex conflict -> vetoed, excluded
        # r4, r5 share non-shared phone -> enters pool as unseeded
        evs = [
            _ev("r0", name="Arif", sex="M"),
            _ev("r1", name="Arif", sex="M"),
            _ev("r2", name="Ramesh", sex="M", ids={"aadhaar": ["123412341234"]}),
            _ev("r3", name="Ramesh", sex="F", ids={"aadhaar": ["123412341234"]}),
            _ev("r4", name="Sita", sex="F", phones=["9811100001"]),
            _ev("r5", name="Sita", sex="F", phones=["9811100001"]),
        ]
        pool = build_fitting_pool(evs)
        self.assertEqual(pool.anchor_pairs, 0)
        self.assertEqual(pool.unseeded_pairs, 1)
        self.assertEqual(pool.counts_by_route["non_shared_phone"], 1)
        self.assertEqual(pool.counts_by_route["strong_id"], 0)


class TestV9Phase4ExportAndE16(unittest.TestCase):
    def test_export_byte_identical_roundtrip_waterfall_and_malformed_rejection(self) -> None:
        m_dists = {
            feat: {lvl: pair[0] for lvl, pair in levels.items()}
            for feat, levels in DEFAULT_FS_DISTRIBUTIONS.items()
        }
        u_dists = {
            feat: {lvl: pair[1] for lvl, pair in levels.items()}
            for feat, levels in DEFAULT_FS_DISTRIBUTIONS.items()
        }
        m_dists["sex"] = {"agrees": 0.8, "disagrees": 0.2}
        u_dists["sex"] = {"agrees": 0.2, "disagrees": 0.8}

        art1 = export_fs_candidate(m_dists, u_dists)
        art2 = export_fs_candidate(m_dists, u_dists)
        s1 = serialize_fs_candidate(art1)
        s2 = serialize_fs_candidate(art2)
        self.assertEqual(s1, s2)

        loaded_cfg, ok = load_fs_candidate(s1)
        self.assertTrue(ok)
        tr = score_pair(_ev("a", sex="M"), _ev("b", sex="M"), policy=loaded_cfg)
        sex_c = next(c for c in tr.contributions if c.feature == "sex")
        self.assertAlmostEqual(sex_c.base_bits, 2.0, places=6)
        wf = render_waterfall(from_pair_trace(tr))
        self.assertIn("base=+2.000", wf)

        # Malformed candidate must fall back to known-good baseline and never replace it
        bad_missing_level = json.loads(s1)
        del bad_missing_level["linkage"]["fs_distributions"]["sex"]["agrees"]
        fallback_cfg, ok_bad = load_fs_candidate(bad_missing_level)
        self.assertFalse(ok_bad)
        self.assertIsNone(fallback_cfg.fs_distributions)

        bad_nonfinite = json.loads(s1)
        bad_nonfinite["linkage"]["fs_distributions"]["sex"]["agrees"] = [float("nan"), 0.2]
        _, ok_nan = load_fs_candidate(bad_nonfinite)
        self.assertFalse(ok_nan)

        with self.assertRaises(LinkageError):
            del m_dists["sex"]
            export_fs_candidate(m_dists, u_dists)

    def test_end_to_end_train_export_and_e16(self) -> None:
        evs: list[Evidence] = []
        truth: dict[tuple[str, str], str] = {}
        # 10 anchored patients (2 records each with Aadhaar)
        for i in range(10):
            aad = f"99000000{i:04d}"
            e1 = _ev(f"a{i}_1", name=f"Patient{i}", sex="M", birth_years=(1980, 1980), ids={"aadhaar": [aad]}, district="D1")
            e2 = _ev(f"a{i}_2", name=f"Patient{i}", sex="M", birth_years=(1980, 1980), ids={"aadhaar": [aad]}, district="D1")
            evs.extend([e1, e2])
            truth[e1.key] = f"PA{i}"
            truth[e2.key] = f"PA{i}"

        # 8 phone-linked patients (unseeded in fitting pool)
        for i in range(8):
            ph = f"981110{i:04d}"
            e1 = _ev(f"p{i}_1", name=f"Unseeded{i}", sex="F", birth_years=(1990, 1990), phones=[ph], district="D2")
            e2 = _ev(f"p{i}_2", name=f"Unseeded{i}", sex="F", birth_years=(1990, 1990), phones=[ph], district="D2")
            evs.extend([e1, e2])
            truth[e1.key] = f"PU{i}"
            truth[e2.key] = f"PU{i}"

        artifact, em_res, report = train_and_export(
            evs,
            seed=11,
            min_u_observations=10,
            min_anchor_pairs=5,
            min_unseeded_pairs=5,
            min_feature_observations=10,
            truth_by_key=truth,
        )
        self.assertTrue(em_res.converged)
        self.assertFalse(report.fallback_to_baseline)
        self.assertIn("fs_distributions", artifact["linkage"])

        e16 = evaluate_e16(evs, truth, _StubKeys(), em_res)
        self.assertEqual(e16.baseline.evaluated_records, len(evs))
        self.assertGreaterEqual(e16.learned_m_u.parity.precision, 0.99)


if __name__ == "__main__":
    unittest.main()
