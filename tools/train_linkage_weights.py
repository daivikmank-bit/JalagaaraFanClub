"""Offline Fellegi-Sunter weight fitting (Plan C V9: C14 and E16).

Dev-only module for estimating ``u`` from random record pairs, fitting ``m``
via Expectation-Maximization on bounded non-name blocks with V4 strong-ID
anchors, exporting runtime-compatible ``fs_distributions`` artifacts, and
running E16 baseline vs learned-``u`` vs learned-``m/u`` comparisons.

Runtime modules in ``kavachmed/`` must never import this file.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from kavachmed.evidence import Evidence, RecKey
from kavachmed.linkage.blocking import _phone_last8
from kavachmed.linkage.fs import FEATURE_ORDER, score_pair
from kavachmed.linkage import LinkageError
from kavachmed.linkage.policy import (
    DEFAULT_FS_DISTRIBUTIONS,
    LinkagePolicy,
    parse_linkage_policy,
)
from kavachmed.linkage.strict import (
    StrictCluster,
    _is_eligible_id_namespace,
    build_strict_clusters,
)
from tools.link_eval import B3Metrics, ParityMetrics, compute_b3_metrics, compute_parity_metrics

_DEFAULT_U_MAX_PAIRS = 100_000
_DEFAULT_MIN_U_OBS = 30
_DEFAULT_MIN_M_OBS = 30
_DEFAULT_MIN_ANCHOR_PAIRS = 5
_DEFAULT_MIN_UNSEEDED_PAIRS = 5
_DEFAULT_MAX_CLUSTER_PAIR_CONTRIB = 50
_DEFAULT_MAX_EM_ITERS = 100
_DEFAULT_EM_TOL = 1e-5
_DEFAULT_CONSECUTIVE_TOL = 3
_PI_MIN = 0.01
_PI_MAX = 0.99
_EXPORT_ROUND_DIGITS = 9


@dataclass(frozen=True)
class PairComparison:
    """Categorical feature levels and veto status for one record pair (no raw PII)."""

    levels: dict[str, str]
    veto_code: str | None = None
    is_anchor: bool = False
    block_route: str = "random"
    cluster_id: str | None = None


@dataclass(frozen=True)
class PatientSplit:
    """Patient-grouped train/calibration/holdout record partitions."""

    train: tuple[RecKey, ...]
    calibration: tuple[RecKey, ...]
    holdout: tuple[RecKey, ...]
    patient_counts: dict[str, int]


@dataclass(frozen=True)
class UEstimateResult:
    """Result of random-pair ``u`` estimation."""

    u_distributions: dict[str, dict[str, float]]
    observed_counts: dict[str, dict[str, int]]
    observed_totals: dict[str, int]
    sampled_pairs: int
    excluded_exact_matches: int
    unknown_match_contamination: int | None
    fallback_reasons: dict[str, str]


@dataclass(frozen=True)
class FittingPool:
    """Bounded non-name fitting pool with V4 strong-ID anchors."""

    pairs: tuple[PairComparison, ...]
    counts_by_route: dict[str, int]
    counts_by_status: dict[str, int]
    anchor_pairs: int
    unseeded_pairs: int
    vetoed_excluded: int
    capped_pairs: int


@dataclass(frozen=True)
class EMFitResult:
    """Result of Expectation-Maximization ``m`` fitting."""

    m_distributions: dict[str, dict[str, float]]
    u_distributions: dict[str, dict[str, float]]
    pi_pool: float
    iterations: int
    converged: bool
    max_delta: float
    learned_features: tuple[str, ...]
    baseline_features: dict[str, str]
    fallback_to_baseline: bool
    fallback_reason: str | None = None


@dataclass(frozen=True)
class FitReport:
    """Text-free aggregate provenance and convergence report for V9."""

    seed: int
    train_records: int
    calibration_records: int
    holdout_records: int
    u_sampled_pairs: int
    u_excluded_exact_matches: int
    u_unknown_contamination: int | None
    u_fallback_reasons: dict[str, str]
    pool_anchor_pairs: int
    pool_unseeded_pairs: int
    pool_vetoed_excluded: int
    pool_capped_pairs: int
    pool_by_route: dict[str, int]
    em_iterations: int
    em_converged: bool
    em_pi_pool: float
    learned_features: tuple[str, ...]
    baseline_features: dict[str, str]
    fallback_to_baseline: bool
    fallback_reason: str | None = None


@dataclass(frozen=True)
class E16ArmResult:
    """Evaluation metrics for one E16 arm (baseline, learned_u, learned_mu)."""

    arm: str
    parity: ParityMetrics
    b3: B3Metrics
    false_merges: int
    abstentions: int
    evaluated_records: int


@dataclass(frozen=True)
class E16Comparison:
    """Side-by-side E16 evaluation and proposed promotion-gate outcome."""

    baseline: E16ArmResult
    learned_u_only: E16ArmResult
    learned_m_u: E16ArmResult
    promote_learned_m_u: bool
    gate_reasons: tuple[str, ...] = ()


def _seeded_rng(purpose: str, seed: int) -> random.Random:
    digest = hashlib.sha256(f"kavachmed:v9:{purpose}:{seed}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:16], "big"))


def _stable_sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def extract_pair_comparison(
    ev_a: Evidence,
    ev_b: Evidence,
    *,
    shared_phones: set[str] | frozenset[str] = frozenset(),
    policy: Mapping[str, Any] | LinkagePolicy | Any = None,
    is_anchor: bool = False,
    block_route: str = "random",
    cluster_id: str | None = None,
) -> PairComparison:
    """Extract V5 categorical comparison levels and veto code via ``score_pair()``."""
    cfg = parse_linkage_policy(policy)
    # Force TF off during level extraction; training estimates base (m, u).
    no_tf_cfg = LinkagePolicy(
        mode=cfg.mode,
        family_mode=cfg.family_mode,
        prior_bits=cfg.prior_bits,
        tau_high_bits=cfg.tau_high_bits,
        tau_low_bits=cfg.tau_low_bits,
        max_component_size=cfg.max_component_size,
        max_pair_checks=cfg.max_pair_checks,
        max_bucket_size=cfg.max_bucket_size,
        pair_budget_factor=cfg.pair_budget_factor,
        pair_budget=cfg.pair_budget,
        tf_enabled=False,
        tf_total_rows=None,
        tf_frequencies=None,
        disabled_id_namespaces=cfg.disabled_id_namespaces,
        allowed_id_namespaces=cfg.allowed_id_namespaces,
        year_reused_uhid_issuers=cfg.year_reused_uhid_issuers,
        fs_distributions=DEFAULT_FS_DISTRIBUTIONS,
    )
    trace = score_pair(ev_a, ev_b, policy=no_tf_cfg, shared_phones=shared_phones)
    levels = {c.feature: c.level for c in trace.contributions}
    return PairComparison(
        levels=levels,
        veto_code=trace.veto_code,
        is_anchor=is_anchor,
        block_route=block_route,
        cluster_id=cluster_id,
    )


def split_by_patient(
    record_keys: Iterable[RecKey],
    truth_by_key: Mapping[RecKey, str | None],
    *,
    seed: int = 0,
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
    mutation_of: Mapping[RecKey, RecKey] | None = None,
) -> PatientSplit:
    """Split records by patient group so no patient or mutation crosses partitions."""
    if len(ratios) != 3 or any(r < 0.0 for r in ratios) or sum(ratios) <= 0.0:
        raise ValueError("ratios must be 3 non-negative numbers with positive sum")

    sorted_keys = sorted(set(record_keys))
    mut_map = mutation_of or {}

    def _group_id(rk: RecKey) -> str:
        base = mut_map.get(rk, rk)
        lbl = truth_by_key.get(base)
        if lbl is None:
            lbl = truth_by_key.get(rk)
        if lbl is not None:
            return f"p:{lbl}"
        return f"u:{base[0]}:{base[1]}"

    groups: dict[str, list[RecKey]] = {}
    for rk in sorted_keys:
        groups.setdefault(_group_id(rk), []).append(rk)

    group_ids = sorted(groups)
    rng = _seeded_rng("patient_split", seed)
    rng.shuffle(group_ids)

    total_g = len(group_ids)
    r_sum = sum(ratios)
    n_train = int(round(total_g * (ratios[0] / r_sum)))
    n_cal = int(round(total_g * (ratios[1] / r_sum)))
    if n_train + n_cal > total_g:
        n_cal = max(0, total_g - n_train)

    train_g = group_ids[:n_train]
    cal_g = group_ids[n_train : n_train + n_cal]
    hold_g = group_ids[n_train + n_cal :]

    train_keys = tuple(sorted(rk for g in train_g for rk in groups[g]))
    cal_keys = tuple(sorted(rk for g in cal_g for rk in groups[g]))
    hold_keys = tuple(sorted(rk for g in hold_g for rk in groups[g]))

    return PatientSplit(
        train=train_keys,
        calibration=cal_keys,
        holdout=hold_keys,
        patient_counts={
            "train": len(train_g),
            "calibration": len(cal_g),
            "holdout": len(hold_g),
        },
    )


def rank_to_pair(rank: int, n: int) -> tuple[int, int]:
    """Convert unordered pair rank ``0 .. n*(n-1)//2 - 1`` to ``(i, j)`` with ``0 <= i < j < n``."""
    if n < 2:
        raise ValueError("n must be >= 2")
    total = n * (n - 1) // 2
    if rank < 0 or rank >= total:
        raise ValueError(f"rank {rank} out of range [0, {total})")
    j = (1 + math.isqrt(1 + 8 * rank)) // 2
    i = rank - j * (j - 1) // 2
    return i, j


def sample_pair_ranks(n: int, max_pairs: int, *, seed: int = 0) -> list[int]:
    """Uniformly sample up to ``max_pairs`` distinct pair ranks without replacement."""
    if n < 2 or max_pairs <= 0:
        return []
    total = n * (n - 1) // 2
    k = min(max_pairs, total)
    rng = _seeded_rng("u_pair_sample", seed)
    if k == total:
        ranks = list(range(total))
        rng.shuffle(ranks)
        return ranks
    return rng.sample(range(total), k)


def _exact_bridge_cluster_map(
    clusters: Sequence[StrictCluster],
    *,
    strong_id_only: bool = False,
) -> dict[RecKey, str]:
    mapping: dict[RecKey, str] = {}
    for idx, cl in enumerate(clusters):
        if strong_id_only and not any(bt[0] == 0 for bt in cl.bridge_keys):
            continue
        cid = f"sc:{idx}"
        for rk in cl.members:
            mapping[rk] = cid
    return mapping


def estimate_u_distributions(
    evidences: Sequence[Evidence],
    *,
    policy: Mapping[str, Any] | LinkagePolicy | Any = None,
    max_pairs: int = _DEFAULT_U_MAX_PAIRS,
    min_observations: int = _DEFAULT_MIN_U_OBS,
    seed: int = 0,
    truth_by_key: Mapping[RecKey, str | None] | None = None,
) -> UEstimateResult:
    """Estimate ``u`` per feature from random non-bridge record pairs."""
    cfg = parse_linkage_policy(policy)
    sorted_evs = sorted(evidences, key=lambda e: e.key)
    n = len(sorted_evs)

    strict_res = build_strict_clusters(sorted_evs, policy=cfg)
    shared_phones = set(strict_res.shared_phones)
    bridge_cluster = _exact_bridge_cluster_map(strict_res.clusters, strong_id_only=False)

    counts: dict[str, dict[str, int]] = {
        feat: {lvl: 0 for lvl in DEFAULT_FS_DISTRIBUTIONS[feat]}
        for feat in FEATURE_ORDER
    }
    totals: dict[str, int] = {feat: 0 for feat in FEATURE_ORDER}

    ranks = sample_pair_ranks(n, max_pairs, seed=seed)
    excluded_exact = 0
    contamination = 0 if truth_by_key is not None else None

    for r in ranks:
        i, j = rank_to_pair(r, n)
        ea, eb = sorted_evs[i], sorted_evs[j]
        ca, cb = bridge_cluster.get(ea.key), bridge_cluster.get(eb.key)
        if ca is not None and ca == cb:
            excluded_exact += 1
            continue

        if contamination is not None and truth_by_key is not None:
            ta, tb = truth_by_key.get(ea.key), truth_by_key.get(eb.key)
            if ta is not None and ta == tb:
                contamination += 1

        comp = extract_pair_comparison(ea, eb, shared_phones=shared_phones, policy=cfg)
        for feat in FEATURE_ORDER:
            lvl = comp.levels[feat]
            if lvl == "missing":
                continue
            counts[feat][lvl] += 1
            totals[feat] += 1

    u_dists: dict[str, dict[str, float]] = {}
    fallback_reasons: dict[str, str] = {}
    for feat in FEATURE_ORDER:
        levels = DEFAULT_FS_DISTRIBUTIONS[feat]
        obs = totals[feat]
        if obs < min_observations:
            u_dists[feat] = {lvl: pair[1] for lvl, pair in levels.items()}
            fallback_reasons[feat] = "insufficient_u_observations"
            continue
        k = len(levels)
        denom = obs + 0.5 * k
        u_dists[feat] = {
            lvl: (counts[feat][lvl] + 0.5) / denom for lvl in levels
        }

    return UEstimateResult(
        u_distributions=u_dists,
        observed_counts=counts,
        observed_totals=totals,
        sampled_pairs=len(ranks),
        excluded_exact_matches=excluded_exact,
        unknown_match_contamination=contamination,
        fallback_reasons=fallback_reasons,
    )


def build_fitting_pool(
    evidences: Sequence[Evidence],
    *,
    policy: Mapping[str, Any] | LinkagePolicy | Any = None,
    max_pairs_per_cluster: int = _DEFAULT_MAX_CLUSTER_PAIR_CONTRIB,
) -> FittingPool:
    """Build bounded fitting pool from scoped-ID and non-shared-phone blocks only."""
    cfg = parse_linkage_policy(policy)
    sorted_evs = sorted(evidences, key=lambda e: e.key)
    by_key: dict[RecKey, Evidence] = {e.key: e for e in sorted_evs}

    strict_res = build_strict_clusters(sorted_evs, policy=cfg)
    shared_phones = set(strict_res.shared_phones)
    contested_set = set(strict_res.contested_bridges)
    strong_anchor_cluster = _exact_bridge_cluster_map(
        strict_res.clusters, strong_id_only=True
    )

    id_buckets: dict[str, list[RecKey]] = {}
    phone_buckets: dict[str, list[RecKey]] = {}

    for ev in sorted_evs:
        rk = ev.key
        for ns in sorted(ev.ids):
            if not _is_eligible_id_namespace(ns, cfg):
                continue
            for core in sorted(ev.ids[ns]):
                if core and (0, ns, core) not in contested_set:
                    id_buckets.setdefault(f"{ns}:{core}", []).append(rk)

        if not ev.phone_shared:
            seen_p8: set[str] = set()
            for ph in sorted(ev.phones):
                if ph in shared_phones:
                    continue
                p8 = _phone_last8(ph)
                if p8 and p8 not in seen_p8:
                    seen_p8.add(p8)
                    phone_buckets.setdefault(p8, []).append(rk)

    seen_pairs: set[tuple[RecKey, RecKey]] = set()
    cluster_pair_counts: dict[str, int] = {}
    pool_pairs: list[PairComparison] = []
    counts_by_route: dict[str, int] = {"strong_id": 0, "non_shared_phone": 0}
    counts_by_status: dict[str, int] = {"anchored": 0, "unseeded": 0}
    anchor_pairs = 0
    unseeded_pairs = 0
    vetoed_excluded = 0
    capped_pairs = 0

    def _process_bucket(members: list[RecKey], route: str) -> None:
        nonlocal anchor_pairs, unseeded_pairs, vetoed_excluded, capped_pairs
        if len(members) < 2 or len(members) > cfg.max_bucket_size:
            return
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                pair = (a, b) if a < b else (b, a)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                ca = strong_anchor_cluster.get(pair[0])
                cb = strong_anchor_cluster.get(pair[1])
                is_anchor = ca is not None and ca == cb
                cid = ca if is_anchor else None

                if cid is not None:
                    used = cluster_pair_counts.get(cid, 0)
                    if used >= max_pairs_per_cluster:
                        capped_pairs += 1
                        continue
                    cluster_pair_counts[cid] = used + 1

                comp = extract_pair_comparison(
                    by_key[pair[0]],
                    by_key[pair[1]],
                    shared_phones=shared_phones,
                    policy=cfg,
                    is_anchor=is_anchor,
                    block_route=route,
                    cluster_id=cid,
                )
                if comp.veto_code is not None:
                    vetoed_excluded += 1
                    continue

                pool_pairs.append(comp)
                counts_by_route[route] += 1
                if is_anchor:
                    anchor_pairs += 1
                    counts_by_status["anchored"] += 1
                else:
                    unseeded_pairs += 1
                    counts_by_status["unseeded"] += 1

    for bkey in sorted(id_buckets):
        _process_bucket(id_buckets[bkey], "strong_id")
    for bkey in sorted(phone_buckets):
        _process_bucket(phone_buckets[bkey], "non_shared_phone")

    return FittingPool(
        pairs=tuple(pool_pairs),
        counts_by_route=counts_by_route,
        counts_by_status=counts_by_status,
        anchor_pairs=anchor_pairs,
        unseeded_pairs=unseeded_pairs,
        vetoed_excluded=vetoed_excluded,
        capped_pairs=capped_pairs,
    )


def fit_em(
    pool: Sequence[PairComparison],
    u_distributions: Mapping[str, Mapping[str, float]],
    *,
    min_anchor_pairs: int = _DEFAULT_MIN_ANCHOR_PAIRS,
    min_unseeded_pairs: int = _DEFAULT_MIN_UNSEEDED_PAIRS,
    min_feature_observations: int = _DEFAULT_MIN_M_OBS,
    max_iters: int = _DEFAULT_MAX_EM_ITERS,
    tol: float = _DEFAULT_EM_TOL,
    consecutive_tol_iters: int = _DEFAULT_CONSECUTIVE_TOL,
    frozen_m_features: Iterable[str] = ("phone",),
) -> EMFitResult:
    """Fit ``m`` distributions via EM on ``PairComparison`` records."""
    baseline_m = {
        feat: {lvl: pair[0] for lvl, pair in levels.items()}
        for feat, levels in DEFAULT_FS_DISTRIBUTIONS.items()
    }
    u_copy = {
        feat: {lvl: float(u_distributions[feat][lvl]) for lvl in levels}
        for feat, levels in DEFAULT_FS_DISTRIBUTIONS.items()
    }

    usable = [p for p in pool if p.veto_code is None]
    anchors = [p for p in usable if p.is_anchor]
    unseeded = [p for p in usable if not p.is_anchor]

    if len(anchors) < min_anchor_pairs or len(unseeded) < min_unseeded_pairs:
        return EMFitResult(
            m_distributions=baseline_m,
            u_distributions=u_copy,
            pi_pool=0.5,
            iterations=0,
            converged=False,
            max_delta=0.0,
            learned_features=(),
            baseline_features={f: "insufficient_pool_support" for f in FEATURE_ORDER},
            fallback_to_baseline=True,
            fallback_reason="insufficient_pool_support",
        )

    frozen_set = set(frozen_m_features)
    baseline_features: dict[str, str] = {}
    learned_list: list[str] = []

    # Check pool support per feature and initialize m from smoothed anchor counts.
    m_curr: dict[str, dict[str, float]] = {}
    for feat in FEATURE_ORDER:
        levels = DEFAULT_FS_DISTRIBUTIONS[feat]
        if feat in frozen_set:
            m_curr[feat] = dict(baseline_m[feat])
            baseline_features[feat] = "frozen_selection_feature"
            continue

        pool_obs = sum(1 for p in usable if p.levels.get(feat, "missing") != "missing")
        if pool_obs < min_feature_observations:
            m_curr[feat] = dict(baseline_m[feat])
            baseline_features[feat] = "insufficient_pool_observations"
            continue

        learned_list.append(feat)
        anchor_counts = {lvl: 0 for lvl in levels}
        anchor_obs = 0
        for p in anchors:
            lvl = p.levels.get(feat, "missing")
            if lvl != "missing":
                anchor_counts[lvl] += 1
                anchor_obs += 1
        if anchor_obs == 0:
            m_curr[feat] = dict(baseline_m[feat])
        else:
            k = len(levels)
            denom = anchor_obs + 0.5 * k
            m_curr[feat] = {
                lvl: (anchor_counts[lvl] + 0.5) / denom for lvl in levels
            }

    if not learned_list:
        return EMFitResult(
            m_distributions=baseline_m,
            u_distributions=u_copy,
            pi_pool=0.5,
            iterations=0,
            converged=False,
            max_delta=0.0,
            learned_features=(),
            baseline_features=baseline_features,
            fallback_to_baseline=True,
            fallback_reason="no_learnable_features",
        )

    pi = max(_PI_MIN, min(_PI_MAX, len(anchors) / len(usable)))
    stable_count = 0
    converged = False
    max_delta = float("inf")
    iters_run = 0

    for it in range(1, max_iters + 1):
        iters_run = it
        logit_pi = math.log(pi / (1.0 - pi))

        # E-step
        responsibilities: list[float] = []
        for p in usable:
            if p.is_anchor:
                responsibilities.append(1.0)
                continue
            ll_sum = 0.0
            for feat in FEATURE_ORDER:
                lvl = p.levels.get(feat, "missing")
                if lvl == "missing":
                    continue
                ll_sum += math.log(m_curr[feat][lvl] / u_copy[feat][lvl])
            responsibilities.append(_stable_sigmoid(logit_pi + ll_sum))

        # M-step
        m_next: dict[str, dict[str, float]] = {}
        for feat in FEATURE_ORDER:
            if feat not in learned_list:
                m_next[feat] = dict(m_curr[feat])
                continue
            levels = DEFAULT_FS_DISTRIBUTIONS[feat]
            k = len(levels)
            w_counts = {lvl: 0.0 for lvl in levels}
            w_total = 0.0
            for p, z in zip(usable, responsibilities):
                lvl = p.levels.get(feat, "missing")
                if lvl == "missing":
                    continue
                w_counts[lvl] += z
                w_total += z
            denom = w_total + 0.5 * k
            m_next[feat] = {
                lvl: (w_counts[lvl] + 0.5) / denom for lvl in levels
            }

        pi_next = max(_PI_MIN, min(_PI_MAX, sum(responsibilities) / len(usable)))

        delta = abs(pi_next - pi)
        for feat in learned_list:
            for lvl in DEFAULT_FS_DISTRIBUTIONS[feat]:
                delta = max(delta, abs(m_next[feat][lvl] - m_curr[feat][lvl]))

        m_curr = m_next
        pi = pi_next
        max_delta = delta

        if not math.isfinite(max_delta):
            return EMFitResult(
                m_distributions=baseline_m,
                u_distributions=u_copy,
                pi_pool=0.5,
                iterations=iters_run,
                converged=False,
                max_delta=float("nan"),
                learned_features=(),
                baseline_features=baseline_features,
                fallback_to_baseline=True,
                fallback_reason="non_finite_estimate",
            )

        if max_delta < tol:
            stable_count += 1
            if stable_count >= consecutive_tol_iters:
                converged = True
                break
        else:
            stable_count = 0

    if not converged:
        return EMFitResult(
            m_distributions=baseline_m,
            u_distributions=u_copy,
            pi_pool=pi,
            iterations=iters_run,
            converged=False,
            max_delta=max_delta,
            learned_features=tuple(learned_list),
            baseline_features=baseline_features,
            fallback_to_baseline=True,
            fallback_reason="did_not_converge",
        )

    return EMFitResult(
        m_distributions=m_curr,
        u_distributions=u_copy,
        pi_pool=pi,
        iterations=iters_run,
        converged=True,
        max_delta=max_delta,
        learned_features=tuple(learned_list),
        baseline_features=baseline_features,
        fallback_to_baseline=False,
        fallback_reason=None,
    )


def _round_distribution_pair(
    m_map: Mapping[str, float],
    u_map: Mapping[str, float],
    levels: Sequence[str],
    digits: int = _EXPORT_ROUND_DIGITS,
) -> dict[str, tuple[float, float]]:
    def _round_probs(probs: Mapping[str, float]) -> dict[str, float]:
        rounded = {lvl: round(float(probs[lvl]), digits) for lvl in levels}
        diff = round(1.0 - sum(rounded.values()), digits)
        if diff != 0.0:
            largest = max(levels, key=lambda l: rounded[l])
            rounded[largest] = round(rounded[largest] + diff, digits)
        return rounded

    rm = _round_probs(m_map)
    ru = _round_probs(u_map)
    return {lvl: (rm[lvl], ru[lvl]) for lvl in levels}


def export_fs_candidate(
    m_distributions: Mapping[str, Mapping[str, float]],
    u_distributions: Mapping[str, Mapping[str, float]],
    *,
    base_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Export ordered ``linkage.fs_distributions`` artifact and validate via ``parse_linkage_policy``."""
    fs_dists: dict[str, dict[str, tuple[float, float]]] = {}
    for feat in FEATURE_ORDER:
        expected_levels = list(DEFAULT_FS_DISTRIBUTIONS[feat].keys())
        if feat not in m_distributions or feat not in u_distributions:
            raise LinkageError("invalid_linkage_policy", f"missing feature '{feat}' in candidate")
        for lvl in expected_levels:
            if lvl not in m_distributions[feat] or lvl not in u_distributions[feat]:
                raise LinkageError(
                    "invalid_linkage_policy",
                    f"missing level '{feat}.{lvl}' in candidate",
                )
        fs_dists[feat] = _round_distribution_pair(
            m_distributions[feat], u_distributions[feat], expected_levels
        )

    linkage_section: dict[str, Any] = {}
    if base_policy and isinstance(base_policy.get("linkage"), Mapping):
        linkage_section.update(base_policy["linkage"])
    linkage_section["fs_distributions"] = fs_dists

    artifact: dict[str, Any] = {"linkage": linkage_section}
    parse_linkage_policy(artifact)
    return artifact


def serialize_fs_candidate(artifact: Mapping[str, Any]) -> str:
    """Deterministically serialize a candidate artifact to JSON."""
    return json.dumps(artifact, indent=2, sort_keys=True) + "\n"


def load_fs_candidate(
    raw_or_json: str | Mapping[str, Any],
    *,
    fallback_policy: Mapping[str, Any] | LinkagePolicy | Any = None,
) -> tuple[LinkagePolicy, bool]:
    """Load an optional candidate artifact; fall back to baseline if malformed."""
    baseline = parse_linkage_policy(fallback_policy)
    try:
        data = json.loads(raw_or_json) if isinstance(raw_or_json, str) else raw_or_json
        if not isinstance(data, Mapping):
            return baseline, False
        linkage = data.get("linkage", data)
        if not isinstance(linkage, Mapping) or "fs_distributions" not in linkage:
            return baseline, False
        fs_raw = linkage["fs_distributions"]
        if not isinstance(fs_raw, Mapping):
            return baseline, False
        for feat in FEATURE_ORDER:
            if feat not in fs_raw or not isinstance(fs_raw[feat], Mapping):
                return baseline, False
            for lvl in DEFAULT_FS_DISTRIBUTIONS[feat]:
                if lvl not in fs_raw[feat]:
                    return baseline, False
        parsed = parse_linkage_policy(data)
        return parsed, True
    except (LinkageError, ValueError, TypeError):
        return baseline, False


def train_and_export(
    evidences: Sequence[Evidence],
    *,
    policy: Mapping[str, Any] | LinkagePolicy | Any = None,
    seed: int = 0,
    max_u_pairs: int = _DEFAULT_U_MAX_PAIRS,
    min_u_observations: int = _DEFAULT_MIN_U_OBS,
    min_anchor_pairs: int = _DEFAULT_MIN_ANCHOR_PAIRS,
    min_unseeded_pairs: int = _DEFAULT_MIN_UNSEEDED_PAIRS,
    min_feature_observations: int = _DEFAULT_MIN_M_OBS,
    max_pairs_per_cluster: int = _DEFAULT_MAX_CLUSTER_PAIR_CONTRIB,
    truth_by_key: Mapping[RecKey, str | None] | None = None,
    split: PatientSplit | None = None,
) -> tuple[dict[str, Any], EMFitResult, FitReport]:
    """Run Phases 2-4 on training evidences and return validated artifact + report."""
    u_res = estimate_u_distributions(
        evidences,
        policy=policy,
        max_pairs=max_u_pairs,
        min_observations=min_u_observations,
        seed=seed,
        truth_by_key=truth_by_key,
    )
    pool = build_fitting_pool(
        evidences,
        policy=policy,
        max_pairs_per_cluster=max_pairs_per_cluster,
    )
    em_res = fit_em(
        pool.pairs,
        u_res.u_distributions,
        min_anchor_pairs=min_anchor_pairs,
        min_unseeded_pairs=min_unseeded_pairs,
        min_feature_observations=min_feature_observations,
    )
    artifact = export_fs_candidate(em_res.m_distributions, em_res.u_distributions)
    report = FitReport(
        seed=seed,
        train_records=len(split.train) if split else len(evidences),
        calibration_records=len(split.calibration) if split else 0,
        holdout_records=len(split.holdout) if split else 0,
        u_sampled_pairs=u_res.sampled_pairs,
        u_excluded_exact_matches=u_res.excluded_exact_matches,
        u_unknown_contamination=u_res.unknown_match_contamination,
        u_fallback_reasons=u_res.fallback_reasons,
        pool_anchor_pairs=pool.anchor_pairs,
        pool_unseeded_pairs=pool.unseeded_pairs,
        pool_vetoed_excluded=pool.vetoed_excluded,
        pool_capped_pairs=pool.capped_pairs,
        pool_by_route=pool.counts_by_route,
        em_iterations=em_res.iterations,
        em_converged=em_res.converged,
        em_pi_pool=em_res.pi_pool,
        learned_features=em_res.learned_features,
        baseline_features=em_res.baseline_features,
        fallback_to_baseline=em_res.fallback_to_baseline,
        fallback_reason=em_res.fallback_reason,
    )
    return artifact, em_res, report


def _evaluate_arm(
    arm: str,
    evidences: Sequence[Evidence],
    truth_by_key: Mapping[RecKey, str | None],
    keys: Any,
    policy: LinkagePolicy,
    file_digests: Mapping[str, str] | None = None,
) -> E16ArmResult:
    from kavachmed.linkage import resolve

    digests = (
        dict(file_digests)
        if file_digests is not None
        else {
            ev.key[0]: hashlib.sha256(ev.key[0].encode("utf-8")).hexdigest()[:32]
            for ev in evidences
        }
    )
    diag: dict[str, int] = {}
    res = resolve(list(evidences), keys, policy=policy, file_digests=digests, diagnostics=diag)
    units: list[tuple[str, str | None]] = []
    for rk in sorted(truth_by_key):
        lbl = truth_by_key[rk]
        if lbl is not None:
            units.append((lbl, res.patient_of.get(rk)))
    parity = compute_parity_metrics(units)
    b3 = compute_b3_metrics(units)
    abstentions = sum(
        diag.get(k, 0)
        for k in (
            "abstentions",
            "over_budget_abstentions",
            "cross_bucket_veto_abstentions",
            "oversized_blocks_abstained",
        )
    )
    return E16ArmResult(
        arm=arm,
        parity=parity,
        b3=b3,
        false_merges=parity.fp,
        abstentions=abstentions,
        evaluated_records=len(units),
    )


def evaluate_e16(
    holdout_evidences: Sequence[Evidence],
    truth_by_key: Mapping[RecKey, str | None],
    keys: Any,
    em_result: EMFitResult,
    *,
    base_policy: Mapping[str, Any] | LinkagePolicy | Any = None,
    file_digests: Mapping[str, str] | None = None,
    precision_floor: float = 0.99,
) -> E16Comparison:
    """Evaluate baseline, learned-u-only, and learned-m/u on identical frozen evidence."""
    base_cfg = parse_linkage_policy(base_policy)

    def _with_dists(dists: Mapping[str, Mapping[str, tuple[float, float]]]) -> LinkagePolicy:
        return LinkagePolicy(
            mode=base_cfg.mode,
            family_mode=base_cfg.family_mode,
            prior_bits=base_cfg.prior_bits,
            tau_high_bits=base_cfg.tau_high_bits,
            tau_low_bits=base_cfg.tau_low_bits,
            max_component_size=base_cfg.max_component_size,
            max_pair_checks=base_cfg.max_pair_checks,
            max_bucket_size=base_cfg.max_bucket_size,
            pair_budget_factor=base_cfg.pair_budget_factor,
            pair_budget=base_cfg.pair_budget,
            tf_enabled=base_cfg.tf_enabled,
            tf_total_rows=base_cfg.tf_total_rows,
            tf_frequencies=base_cfg.tf_frequencies,
            disabled_id_namespaces=base_cfg.disabled_id_namespaces,
            allowed_id_namespaces=base_cfg.allowed_id_namespaces,
            year_reused_uhid_issuers=base_cfg.year_reused_uhid_issuers,
            fs_distributions=dists,
        )

    u_only_dists = {
        feat: {
            lvl: (DEFAULT_FS_DISTRIBUTIONS[feat][lvl][0], em_result.u_distributions[feat][lvl])
            for lvl in DEFAULT_FS_DISTRIBUTIONS[feat]
        }
        for feat in FEATURE_ORDER
    }
    mu_dists = {
        feat: {
            lvl: (em_result.m_distributions[feat][lvl], em_result.u_distributions[feat][lvl])
            for lvl in DEFAULT_FS_DISTRIBUTIONS[feat]
        }
        for feat in FEATURE_ORDER
    }

    arm_base = _evaluate_arm(
        "baseline", holdout_evidences, truth_by_key, keys, _with_dists(DEFAULT_FS_DISTRIBUTIONS), file_digests
    )
    arm_u = _evaluate_arm(
        "learned_u_only", holdout_evidences, truth_by_key, keys, _with_dists(u_only_dists), file_digests
    )
    arm_mu = _evaluate_arm(
        "learned_m_u", holdout_evidences, truth_by_key, keys, _with_dists(mu_dists), file_digests
    )

    reasons: list[str] = []
    if em_result.fallback_to_baseline or not em_result.converged:
        reasons.append("em_fallback_or_nonconverged")
    if arm_mu.parity.precision < precision_floor:
        reasons.append("precision_below_floor")
    if arm_mu.parity.f05 <= arm_base.parity.f05:
        reasons.append("f05_not_improved")
    if arm_mu.false_merges > arm_base.false_merges:
        reasons.append("false_merges_regressed")

    return E16Comparison(
        baseline=arm_base,
        learned_u_only=arm_u,
        learned_m_u=arm_mu,
        promote_learned_m_u=len(reasons) == 0,
        gate_reasons=tuple(reasons),
    )
