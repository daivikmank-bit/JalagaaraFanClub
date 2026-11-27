"""Bounded candidate blocking for Fellegi-Sunter linkage (Plan C V5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from kavachmed.evidence import Evidence, RecKey
from kavachmed.linkage.policy import LinkagePolicy, parse_linkage_policy
from kavachmed.linkage.strict import _is_eligible_id_namespace

_MAX_USABLE_BIRTH_SPAN = 20

# Priority order: trusted-ID candidates first, then phone, name+sex, relative+decade.
_BLOCK_PRIORITIES: tuple[str, ...] = (
    "strong_id",
    "phone",
    "name_sex",
    "rel_decade",
)


@dataclass(frozen=True)
class BlockingResult:
    """Deterministic candidate pairs and bounded-blocking diagnostics."""

    candidates: tuple[tuple[RecKey, RecKey], ...]
    uncapped_candidate_count: int
    skipped_by_cap: int
    oversized_blocks_abstained: int
    by_priority_counts: dict[str, int]


def _usable_decades(birth_years: tuple[int, int] | None) -> tuple[int, ...]:
    if birth_years is None:
        return ()
    lo, hi = birth_years
    if lo > hi or (hi - lo) > _MAX_USABLE_BIRTH_SPAN:
        return ()
    start_dec = (lo // 10) * 10
    end_dec = (hi // 10) * 10
    return tuple(range(start_dec, end_dec + 1, 10))


def _identifying_skeletons(tokens: list[Any]) -> tuple[str, ...]:
    skels = {
        tok.skeleton
        for tok in tokens
        if tok.kind == "identifying" and tok.skeleton
    }
    return tuple(sorted(skels))


def _phone_last8(phone: str) -> str | None:
    digits = "".join(ch for ch in phone if ch.isdigit())
    if len(digits) < 8:
        return None
    return digits[-8:]


def _secondary_reblock_keys(
    ev: Evidence,
    *,
    policy: LinkagePolicy,
    contested_set: set[tuple[int, str, str]],
) -> tuple[str, ...]:
    """Build independent secondary keys to reblock an oversized bucket."""
    keys: list[str] = []
    for ns in sorted(ev.ids):
        if not _is_eligible_id_namespace(ns, policy):
            continue
        for core in sorted(ev.ids[ns]):
            if core and (0, ns, core) not in contested_set:
                keys.append(f"id:{ns}:{core}")
    for dec in _usable_decades(ev.birth_years):
        keys.append(f"dec:{dec}")
    for skel in _identifying_skeletons(ev.names):
        if ev.sex in {"M", "F"}:
            keys.append(f"ns:{ev.sex}:{skel}")
        else:
            keys.append(f"n:{skel}")
    return tuple(dict.fromkeys(keys))


def generate_candidates(
    evidences: list[Evidence],
    *,
    policy: Mapping[str, Any] | LinkagePolicy | Any = None,
    contested_bridges: Iterable[tuple[int, str, str]] = (),
    diagnostics: dict[str, int] | None = None,
) -> BlockingResult:
    """Generate bounded, deduplicated candidate pairs in fixed priority order."""
    cfg = parse_linkage_policy(policy)
    by_key: dict[RecKey, Evidence] = {ev.key: ev for ev in evidences}
    sorted_keys = sorted(by_key)
    n_records = len(sorted_keys)
    pair_budget = cfg.pair_budget if cfg.pair_budget is not None else cfg.pair_budget_factor * n_records
    contested_set = set(contested_bridges)

    buckets: dict[str, dict[str, list[RecKey]]] = {
        btype: {} for btype in _BLOCK_PRIORITIES
    }

    for rk in sorted_keys:
        ev = by_key[rk]

        # 1. Eligible scoped strong IDs (excluding contested bridges).
        for ns in sorted(ev.ids):
            if not _is_eligible_id_namespace(ns, cfg):
                continue
            for core in sorted(ev.ids[ns]):
                if core and (0, ns, core) not in contested_set:
                    buckets["strong_id"].setdefault(f"{ns}:{core}", []).append(rk)

        # 2. Phone last 8 digits (includes shared phones within bucket-budget rules).
        seen_p8: set[str] = set()
        for ph in ev.phones:
            p8 = _phone_last8(ph)
            if p8 and p8 not in seen_p8:
                seen_p8.add(p8)
                buckets["phone"].setdefault(p8, []).append(rk)

        # 3. Identifying name skeleton + known sex (generic/blank produces no block).
        if ev.sex in {"M", "F"}:
            for skel in _identifying_skeletons(ev.names):
                buckets["name_sex"].setdefault(f"{ev.sex}:{skel}", []).append(rk)

        # 4. Identifying relative skeleton + intersected birth decade(s).
        rel_skels = _identifying_skeletons(ev.kin)
        if rel_skels:
            decades = _usable_decades(ev.birth_years)
            for dec in decades:
                for r_skel in rel_skels:
                    buckets["rel_decade"].setdefault(f"{r_skel}:{dec}", []).append(rk)

    seen_pairs: set[tuple[RecKey, RecKey]] = set()
    ordered_candidates: list[tuple[RecKey, RecKey]] = []
    by_priority_counts: dict[str, int] = {btype: 0 for btype in _BLOCK_PRIORITIES}
    uncapped_pairs: set[tuple[RecKey, RecKey]] = set()
    skipped_by_cap = 0
    oversized_abstained = 0
    budget_exhausted = False

    def _emit_from_group(members: list[RecKey], btype: str) -> None:
        nonlocal skipped_by_cap, budget_exhausted
        m_len = len(members)
        for i in range(m_len):
            a = members[i]
            for j in range(i + 1, m_len):
                b = members[j]
                pair = (a, b) if a < b else (b, a)
                if pair in seen_pairs:
                    continue
                uncapped_pairs.add(pair)
                if len(ordered_candidates) < pair_budget and not budget_exhausted:
                    seen_pairs.add(pair)
                    ordered_candidates.append(pair)
                    by_priority_counts[btype] += 1
                else:
                    budget_exhausted = True
                    skipped_by_cap += 1

    for btype in _BLOCK_PRIORITIES:
        bmap = buckets[btype]
        for bkey in sorted(bmap):
            members = bmap[bkey]
            if len(members) < 2:
                continue
            if len(members) <= cfg.max_bucket_size:
                _emit_from_group(members, btype)
                continue

            # Oversized bucket: reblock using independent secondary evidence before expanding.
            sub_buckets: dict[str, list[RecKey]] = {}
            for rk in members:
                for skey in _secondary_reblock_keys(
                    by_key[rk], policy=cfg, contested_set=contested_set
                ):
                    sub_buckets.setdefault(skey, []).append(rk)

            # Track pairs that cannot be resolved into bounded sub-buckets.
            resolved_in_sub: set[tuple[RecKey, RecKey]] = set()
            has_unresolved_sub = False
            for skey in sorted(sub_buckets):
                sub_members = sub_buckets[skey]
                if len(sub_members) < 2:
                    continue
                if len(sub_members) <= cfg.max_bucket_size:
                    _emit_from_group(sub_members, btype)
                    for i in range(len(sub_members)):
                        for j in range(i + 1, len(sub_members)):
                            a, b = sub_members[i], sub_members[j]
                            resolved_in_sub.add((a, b) if a < b else (b, a))
                else:
                    has_unresolved_sub = True

            total_bucket_pairs = len(members) * (len(members) - 1) // 2
            unresolved_count = max(0, total_bucket_pairs - len(resolved_in_sub))
            if unresolved_count > 0 or has_unresolved_sub:
                oversized_abstained += 1
                skipped_by_cap += unresolved_count

    if diagnostics is not None:
        diagnostics["candidates_generated"] = len(ordered_candidates)
        diagnostics["candidates_skipped_by_cap"] = skipped_by_cap
        diagnostics["oversized_blocks_abstained"] = oversized_abstained

    return BlockingResult(
        candidates=tuple(ordered_candidates),
        uncapped_candidate_count=len(uncapped_pairs) + (
            0 if oversized_abstained == 0 else 0
        ),
        skipped_by_cap=skipped_by_cap,
        oversized_blocks_abstained=oversized_abstained,
        by_priority_counts=by_priority_counts,
    )
