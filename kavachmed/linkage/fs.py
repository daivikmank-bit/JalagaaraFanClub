"""Pure Fellegi-Sunter pair scorer and neutral trace (Plan C V5)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from kavachmed.evidence import Evidence
from kavachmed.indic import NameToken
from kavachmed.linkage.policy import (
    DEFAULT_FS_DISTRIBUTIONS,
    LinkagePolicy,
    parse_linkage_policy,
)
from kavachmed.linkage.strict import _birth_year_gap, _pair_has_veto, _tokens_compatible

FEATURE_ORDER: tuple[str, ...] = (
    "name",
    "relative",
    "birth_interval",
    "sex",
    "geography",
    "phone",
    "hospital",
)

_JARO_WINKLER_THRESHOLD = 0.90


@dataclass(frozen=True)
class FeatureScore:
    """Single feature's comparison level and log2 likelihood-ratio weights."""

    feature: str
    level: str
    base_bits: float
    tf_adjusted_bits: float


@dataclass(frozen=True)
class PairTrace:
    """Neutral pair-scoring trace (no raw tokens, identifiers, or paths)."""

    pair_label: str
    prior_bits: float
    contributions: tuple[FeatureScore, ...]
    bayes_factor_bits: float
    total_bits: float
    tau_high_bits: float
    tau_low_bits: float
    tf_enabled: bool
    veto_code: str | None = None
    abstention_code: str | None = None
    reason_codes: tuple[str, ...] = ()

    @property
    def passes_high(self) -> bool:
        return self.veto_code is None and self.total_bits >= self.tau_high_bits

    @property
    def passes_low(self) -> bool:
        return self.veto_code is None and self.total_bits >= self.tau_low_bits


def jaro_winkler(s1: str, s2: str) -> float:
    """Compute standard Jaro-Winkler similarity (max prefix 4, p=0.1)."""
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    match_dist = max(len1, len2) // 2 - 1
    if match_dist < 0:
        match_dist = 0

    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = 0
    for i in range(len1):
        start = max(0, i - match_dist)
        end = min(i + match_dist + 1, len2)
        for j in range(start, end):
            if s2_matches[j]:
                continue
            if s1[i] == s2[j]:
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break

    if matches == 0:
        return 0.0

    t = 0
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            t += 1
        k += 1

    jaro = (
        (matches / len1) + (matches / len2) + ((matches - t / 2.0) / matches)
    ) / 3.0

    prefix = 0
    for i in range(min(4, len1, len2)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break

    return jaro + prefix * 0.1 * (1.0 - jaro)


def _token_surface_key(tok: NameToken) -> str | None:
    if tok.latin:
        return tok.latin
    if tok.surface:
        return tok.surface.casefold()
    return None


def _compare_name_tokens(
    tokens_a: list[NameToken],
    tokens_b: list[NameToken],
) -> tuple[str, list[NameToken]]:
    """Compare name token lists and return ``(level, supporting_tokens)``.

    Generic and honorific tokens never create agreement.
    """
    id_a = [t for t in tokens_a if t.kind in {"identifying", "initial"}]
    id_b = [t for t in tokens_b if t.kind in {"identifying", "initial"}]
    if not id_a or not id_b:
        return "missing", []

    ident_a = [t for t in id_a if t.kind == "identifying"]
    ident_b = [t for t in id_b if t.kind == "identifying"]

    # 1. Exact normalized match on identifying tokens.
    exact_matches: list[NameToken] = []
    b_exact = {_token_surface_key(t) for t in ident_b if _token_surface_key(t)}
    for ta in ident_a:
        ka = _token_surface_key(ta)
        if ka and ka in b_exact:
            exact_matches.append(ta)
    if exact_matches:
        return "exact", exact_matches

    # 2. Identifying skeleton match.
    skel_matches: list[NameToken] = []
    b_skels = {t.skeleton for t in ident_b if t.skeleton}
    for ta in ident_a:
        if ta.skeleton and ta.skeleton in b_skels:
            skel_matches.append(ta)
    if skel_matches:
        return "skeleton", skel_matches

    # 3. Jaro-Winkler >= 0.90 on identifying tokens.
    for ta in ident_a:
        ka = _token_surface_key(ta)
        if not ka or len(ka) <= 1:
            continue
        for tb in ident_b:
            kb = _token_surface_key(tb)
            if not kb or len(kb) <= 1:
                continue
            if jaro_winkler(ka, kb) >= _JARO_WINKLER_THRESHOLD:
                return "jaro_winkler", []

    # 4. Compatible initial.
    for ta in id_a:
        for tb in id_b:
            if _tokens_compatible(ta, tb):
                return "initial", []

    return "disagreement", []


def _tf_correction(
    level: str,
    supporting_tokens: list[NameToken],
    *,
    tf_active: bool,
    tf_total_rows: int | None,
    tf_frequencies: Mapping[str, int] | None,
) -> float:
    if not tf_active or not tf_total_rows or not tf_frequencies or not supporting_tokens:
        return 0.0

    alpha = 1.0 if level == "exact" else (0.5 if level == "skeleton" else 0.0)
    if alpha == 0.0:
        return 0.0

    # Use the least rare shared supporting token when several qualify.
    max_count: int | None = None
    for tok in supporting_tokens:
        if tok.kind != "identifying":
            continue
        cnt = None
        if tok.latin and tok.latin in tf_frequencies:
            cnt = tf_frequencies[tok.latin]
        elif tok.skeleton and tok.skeleton in tf_frequencies:
            cnt = tf_frequencies[tok.skeleton]
        if cnt is not None and cnt > 0:
            if max_count is None or cnt > max_count:
                max_count = cnt

    if max_count is None:
        return 0.0

    p_token = max_count / tf_total_rows
    if p_token <= 0.0:
        return 0.0
    raw = alpha * math.log2(0.01 / p_token)
    return max(-2.0, min(2.0, raw))


def _district_key(ev: Evidence) -> str | None:
    if not ev.geo:
        return None
    val = ev.geo.get("district_code") or ev.geo.get("district")
    if isinstance(val, str) and val.strip():
        return val.strip().casefold()
    return None


def _non_shared_phones(ev: Evidence, shared_set: set[str]) -> set[str]:
    if not ev.phones:
        return set()
    if shared_set:
        return {p for p in ev.phones if p and p not in shared_set}
    if ev.phone_shared:
        return set()
    return {p for p in ev.phones if p}


def score_pair(
    ev_a: Evidence,
    ev_b: Evidence,
    *,
    policy: Mapping[str, Any] | LinkagePolicy | Any = None,
    shared_phones: set[str] | frozenset[str] = frozenset(),
    pair_label: str = "pair",
) -> PairTrace:
    """Compute Fellegi-Sunter log2 Bayes-factor score and neutral trace."""
    cfg = parse_linkage_policy(policy)
    dists = cfg.fs_distributions or DEFAULT_FS_DISTRIBUTIONS
    tf_active = bool(
        cfg.tf_enabled
        and cfg.tf_total_rows is not None
        and cfg.tf_total_rows > 0
        and cfg.tf_frequencies
    )

    vetoed, veto_code = _pair_has_veto(ev_a, ev_b)
    shared_set = set(shared_phones)

    # 1. Name
    name_level, name_support = _compare_name_tokens(ev_a.names, ev_b.names)

    # 2. Relative
    rel_raw_level, rel_support = _compare_name_tokens(ev_a.kin, ev_b.kin)
    if rel_raw_level == "missing":
        rel_level = "missing"
        rel_tf_level = "missing"
    elif rel_raw_level in {"exact", "skeleton", "jaro_winkler", "initial"}:
        rel_level = "agrees"
        rel_tf_level = rel_raw_level if rel_raw_level in {"exact", "skeleton"} else "none"
    else:
        rel_level = "disagrees"
        rel_tf_level = "none"

    # 3. Birth interval
    if ev_a.birth_years is not None and ev_b.birth_years is not None:
        gap = _birth_year_gap(ev_a.birth_years, ev_b.birth_years)
        if gap == 0:
            birth_level = "overlaps"
        elif gap <= 3:
            birth_level = "near"
        else:
            birth_level = "disagrees"
    else:
        birth_level = "missing"

    # 4. Sex
    if ev_a.sex is not None and ev_b.sex is not None:
        sex_level = "agrees" if ev_a.sex == ev_b.sex else "disagrees"
    else:
        sex_level = "missing"

    # 5. Geography
    da, db = _district_key(ev_a), _district_key(ev_b)
    if da is not None and db is not None:
        geo_level = "same_district" if da == db else "different_district"
    else:
        geo_level = "missing"

    # 6. Phone (non-shared only)
    pa, pb = _non_shared_phones(ev_a, shared_set), _non_shared_phones(ev_b, shared_set)
    if pa and pb:
        phone_level = "exact_non_shared" if (pa & pb) else "different"
    else:
        phone_level = "missing"

    # 7. Hospital
    if ev_a.hospital and ev_b.hospital:
        hosp_level = (
            "agrees"
            if ev_a.hospital.strip().casefold() == ev_b.hospital.strip().casefold()
            else "disagrees"
        )
    else:
        hosp_level = "missing"

    levels = {
        "name": name_level,
        "relative": rel_level,
        "birth_interval": birth_level,
        "sex": sex_level,
        "geography": geo_level,
        "phone": phone_level,
        "hospital": hosp_level,
    }

    contributions: list[FeatureScore] = []
    bayes_factor = 0.0
    reason_codes: list[str] = []

    for feat in FEATURE_ORDER:
        lvl = levels[feat]
        if lvl == "missing":
            contributions.append(FeatureScore(feat, "missing", 0.0, 0.0))
            continue

        m_val, u_val = dists[feat][lvl]
        base_bits = math.log2(m_val / u_val)
        corr = 0.0
        if feat == "name":
            corr = _tf_correction(
                lvl,
                name_support,
                tf_active=tf_active,
                tf_total_rows=cfg.tf_total_rows,
                tf_frequencies=cfg.tf_frequencies,
            )
        elif feat == "relative" and lvl == "agrees":
            corr = _tf_correction(
                rel_tf_level,
                rel_support,
                tf_active=tf_active,
                tf_total_rows=cfg.tf_total_rows,
                tf_frequencies=cfg.tf_frequencies,
            )
        adj_bits = base_bits + corr
        bayes_factor += adj_bits
        contributions.append(FeatureScore(feat, lvl, base_bits, adj_bits))
        reason_codes.append(f"{feat}:{lvl}")

    if not tf_active:
        reason_codes.append("tf:disabled")
    if vetoed and veto_code:
        reason_codes.append(f"veto:{veto_code}")

    total_bits = round(cfg.prior_bits + bayes_factor, 6)

    return PairTrace(
        pair_label=pair_label,
        prior_bits=cfg.prior_bits,
        contributions=tuple(contributions),
        bayes_factor_bits=round(bayes_factor, 6),
        total_bits=total_bits,
        tau_high_bits=cfg.tau_high_bits,
        tau_low_bits=cfg.tau_low_bits,
        tf_enabled=tf_active,
        veto_code=veto_code if vetoed else None,
        abstention_code=None,
        reason_codes=tuple(reason_codes),
    )
