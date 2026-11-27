"""Complete-linkage scored clustering with abstention (Plan C V5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from kavachmed.evidence import Evidence, RecKey
from kavachmed.linkage.blocking import generate_candidates
from kavachmed.linkage.fs import PairTrace, score_pair
from kavachmed.linkage.policy import LinkagePolicy, parse_linkage_policy
from kavachmed.linkage.strict import StrictCluster, StrictResult, _UnionFind


@dataclass(frozen=True)
class ScoredCluster:
    """An accepted cluster after V4 seeding and V5 complete-linkage expansion."""

    canonical_key: tuple[int, str, str]  # (0, ns, core), (1, "adm", adm), or (2, "scored", fallback)
    members: tuple[RecKey, ...]
    from_v4: bool = False


@dataclass(frozen=True)
class ScoredResult:
    """Result of V5 complete-linkage clustering."""

    record_keys: tuple[RecKey, ...]
    clusters: tuple[ScoredCluster, ...]
    low_threshold_candidates: int = 0
    abstained_components: int = 0


def _has_corroboration_beyond_name(trace: PairTrace) -> bool:
    """Require independent corroborating evidence beyond the patient name."""
    for contrib in trace.contributions:
        if contrib.feature == "relative" and contrib.level == "agrees":
            return True
        if contrib.feature == "birth_interval" and contrib.level in {"overlaps", "near"}:
            return True
        if contrib.feature == "phone" and contrib.level == "exact_non_shared":
            return True
        if contrib.feature == "geography" and contrib.level == "same_district":
            return True
    return False


def _scored_fallback_key(
    members: tuple[RecKey, ...],
    by_key: dict[RecKey, Evidence],
) -> tuple[int, str, str]:
    name_skels = sorted(
        {
            tok.skeleton
            for rk in members
            for tok in by_key[rk].names
            if tok.kind == "identifying" and tok.skeleton
        }
    )
    rel_skels = sorted(
        {
            tok.skeleton
            for rk in members
            for tok in by_key[rk].kin
            if tok.kind == "identifying" and tok.skeleton
        }
    )
    los = [by_key[rk].birth_years[0] for rk in members if by_key[rk].birth_years is not None]
    his = [by_key[rk].birth_years[1] for rk in members if by_key[rk].birth_years is not None]
    by_str = f"{min(los)}-{max(his)}" if los and his else "none"
    core = f"n:{','.join(name_skels)}|r:{','.join(rel_skels)}|b:{by_str}"
    return (2, "scored", core)


def _strict_reblock_key(ev: Evidence) -> str:
    skels = sorted(
        {tok.skeleton for tok in ev.names if tok.kind == "identifying" and tok.skeleton}
    )
    name_part = skels[0] if skels else "_"
    sex_part = ev.sex or "_"
    dec_part = str((ev.birth_years[0] // 10) * 10) if ev.birth_years is not None else "_"
    return f"{sex_part}:{name_part}:{dec_part}"


def build_scored_clusters(
    evidences: list[Evidence],
    strict_result: StrictResult,
    *,
    policy: Mapping[str, Any] | LinkagePolicy | Any = None,
    diagnostics: dict[str, int] | None = None,
) -> ScoredResult:
    """Expand V4 strict clusters with bounded complete-linkage Fellegi-Sunter merges."""
    cfg = parse_linkage_policy(policy)
    by_key: dict[RecKey, Evidence] = {ev.key: ev for ev in evidences}
    sorted_keys = list(strict_result.record_keys)
    shared_set = set(strict_result.shared_phones)

    # Map each record to its V4 seed cluster (if any).
    v4_by_rk: dict[RecKey, StrictCluster] = {}
    for sc in strict_result.clusters:
        for rk in sc.members:
            v4_by_rk[rk] = sc

    blocking = generate_candidates(
        evidences,
        policy=cfg,
        contested_bridges=strict_result.contested_bridges,
        diagnostics=diagnostics,
    )

    pair_cache: dict[tuple[RecKey, RecKey], PairTrace] = {}

    def _score(rk_a: RecKey, rk_b: RecKey) -> PairTrace:
        pair = (rk_a, rk_b) if rk_a < rk_b else (rk_b, rk_a)
        cached = pair_cache.get(pair)
        if cached is not None:
            return cached
        tr = score_pair(
            by_key[pair[0]],
            by_key[pair[1]],
            policy=cfg,
            shared_phones=shared_set,
        )
        pair_cache[pair] = tr
        return tr

    # Score candidate pairs not already in the same V4 cluster.
    uf = _UnionFind(sorted_keys)
    for sc in strict_result.clusters:
        anchor = sc.members[0]
        for rk in sc.members[1:]:
            uf.union(anchor, rk)

    low_threshold_candidates = 0
    for rk_a, rk_b in blocking.candidates:
        if v4_by_rk.get(rk_a) is not None and v4_by_rk.get(rk_a) is v4_by_rk.get(rk_b):
            continue
        tr = _score(rk_a, rk_b)
        if tr.veto_code is None and cfg.tau_low_bits <= tr.total_bits < cfg.tau_high_bits:
            low_threshold_candidates += 1
        if tr.passes_high:
            uf.union(rk_a, rk_b)

    raw_components: dict[RecKey, list[RecKey]] = {}
    for rk in sorted_keys:
        raw_components.setdefault(uf.find(rk), []).append(rk)

    # Split oversized components with stricter evidence; abstain on any still > cap.
    bounded_components: list[list[RecKey]] = []
    abstained_components = 0
    for root in sorted(raw_components):
        comp = sorted(raw_components[root])
        if len(comp) <= cfg.max_component_size:
            bounded_components.append(comp)
            continue

        # Reblock using stricter key while keeping V4 seed clusters intact.
        sub_groups: dict[str, list[RecKey]] = {}
        for rk in comp:
            sc = v4_by_rk.get(rk)
            skey = f"v4:{sc.canonical_key}" if sc is not None else _strict_reblock_key(by_key[rk])
            sub_groups.setdefault(skey, []).append(rk)

        for skey in sorted(sub_groups):
            sub = sorted(sub_groups[skey])
            if len(sub) <= cfg.max_component_size:
                bounded_components.append(sub)
            else:
                abstained_components += 1
                # Preserve V4 clusters inside the abstained sub-component; singletons stay singletons.
                seen_v4: set[tuple[int, str, str]] = set()
                for rk in sub:
                    sc = v4_by_rk.get(rk)
                    if sc is not None and sc.canonical_key not in seen_v4:
                        seen_v4.add(sc.canonical_key)
                        bounded_components.append(list(sc.members))
                    elif sc is None:
                        bounded_components.append([rk])

    # Run complete-linkage rounds inside each bounded component.
    provisional_clusters: list[tuple[tuple[RecKey, ...], list[StrictCluster]]] = []

    for comp in bounded_components:
        if len(comp) < 2:
            continue

        # Build initial working clusters: indivisible V4 seeds + singletons.
        working: dict[RecKey, list[RecKey]] = {}
        seed_v4s: dict[RecKey, list[StrictCluster]] = {}
        assigned_v4: set[tuple[int, str, str]] = set()
        for rk in comp:
            sc = v4_by_rk.get(rk)
            if sc is not None:
                if sc.canonical_key not in assigned_v4:
                    assigned_v4.add(sc.canonical_key)
                    w_id = sc.members[0]
                    working[w_id] = sorted(sc.members)
                    seed_v4s[w_id] = [sc]
            else:
                working[rk] = [rk]
                seed_v4s[rk] = []

        if len(working) < 2:
            for w_id, members in working.items():
                if len(members) >= 2:
                    provisional_clusters.append((tuple(members), seed_v4s[w_id]))
            continue

        comp_checks = 0
        budget_hit = False

        def _clusters_compatible(id_a: RecKey, id_b: RecKey) -> bool:
            nonlocal comp_checks, budget_hit
            m_a = working[id_a]
            m_b = working[id_b]
            has_corrob = False
            for ra in m_a:
                for rb in m_b:
                    pair = (ra, rb) if ra < rb else (rb, ra)
                    if pair not in pair_cache:
                        if comp_checks >= cfg.max_pair_checks:
                            budget_hit = True
                            return False
                        comp_checks += 1
                    tr = _score(ra, rb)
                    if not tr.passes_high:
                        return False
                    if _has_corroboration_beyond_name(tr):
                        has_corrob = True
            return has_corrob

        while len(working) >= 2 and not budget_hit:
            w_ids = sorted(working)
            compat: dict[RecKey, list[RecKey]] = {wid: [] for wid in w_ids}
            for i in range(len(w_ids)):
                for j in range(i + 1, len(w_ids)):
                    wa, wb = w_ids[i], w_ids[j]
                    if _clusters_compatible(wa, wb):
                        compat[wa].append(wb)
                        compat[wb].append(wa)

            merges: list[tuple[RecKey, RecKey]] = []
            used: set[RecKey] = set()
            for wa in w_ids:
                if wa in used:
                    continue
                partners = compat[wa]
                if len(partners) == 1:
                    wb = partners[0]
                    if wb not in used and compat[wb] == [wa]:
                        pair = (wa, wb) if wa < wb else (wb, wa)
                        merges.append(pair)
                        used.add(wa)
                        used.add(wb)
                elif len(partners) > 1 and diagnostics is not None:
                    diagnostics["ambiguous_partner_abstentions"] = (
                        diagnostics.get("ambiguous_partner_abstentions", 0) + 1
                    )

            if not merges:
                break

            for wa, wb in sorted(merges):
                merged_members = sorted(working[wa] + working[wb])
                merged_v4s = seed_v4s[wa] + seed_v4s[wb]
                del working[wa]
                del working[wb]
                del seed_v4s[wa]
                del seed_v4s[wb]
                new_id = merged_members[0]
                working[new_id] = merged_members
                seed_v4s[new_id] = merged_v4s

        if budget_hit:
            abstained_components += 1
            if diagnostics is not None:
                diagnostics["over_budget_abstentions"] = (
                    diagnostics.get("over_budget_abstentions", 0) + 1
                )

        for w_id in sorted(working):
            members = working[w_id]
            if len(members) >= 2:
                provisional_clusters.append((tuple(members), seed_v4s[w_id]))

    # Assign canonical keys and guard against fallback-key collisions.
    keyed_clusters: list[tuple[tuple[int, str, str], tuple[RecKey, ...], list[StrictCluster]]] = []
    key_counts: dict[tuple[int, str, str], int] = {}

    for members, v4_list in provisional_clusters:
        strong = sorted({sc.canonical_key for sc in v4_list if sc.canonical_key[0] == 0})
        adm = sorted({sc.canonical_key for sc in v4_list if sc.canonical_key[0] == 1})
        if strong:
            ckey = strong[0]
        elif adm:
            ckey = adm[0]
        else:
            ckey = _scored_fallback_key(members, by_key)
        keyed_clusters.append((ckey, members, v4_list))
        key_counts[ckey] = key_counts.get(ckey, 0) + 1

    final_clusters: list[ScoredCluster] = []
    for ckey, members, v4_list in keyed_clusters:
        if key_counts[ckey] > 1:
            # Collision: abstain on scored expansion and retain last valid V4 state.
            if diagnostics is not None:
                diagnostics["fallback_key_collisions"] = (
                    diagnostics.get("fallback_key_collisions", 0) + 1
                )
            for sc in v4_list:
                final_clusters.append(
                    ScoredCluster(
                        canonical_key=sc.canonical_key,
                        members=sc.members,
                        from_v4=True,
                    )
                )
        else:
            final_clusters.append(
                ScoredCluster(
                    canonical_key=ckey,
                    members=members,
                    from_v4=(len(v4_list) == 1 and len(members) == len(v4_list[0].members)),
                )
            )

    final_clusters.sort(key=lambda c: c.canonical_key)

    if diagnostics is not None:
        diagnostics["low_threshold_candidates"] = low_threshold_candidates
        diagnostics["scored_clusters"] = len(final_clusters)
        diagnostics["abstained_components"] = abstained_components

    return ScoredResult(
        record_keys=tuple(sorted_keys),
        clusters=tuple(final_clusters),
        low_threshold_candidates=low_threshold_candidates,
        abstained_components=abstained_components,
    )
