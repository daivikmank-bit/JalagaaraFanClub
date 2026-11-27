"""Temporal-family date-shift graph after V5 patient linkage (Plan C V6)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from kavachmed.evidence import Evidence, RecKey
from kavachmed.linkage.blocking import _identifying_skeletons, generate_candidates
from kavachmed.linkage.fs import PairTrace, score_pair
from kavachmed.linkage.policy import LinkagePolicy, parse_linkage_policy
from kavachmed.linkage.pseudo import derive_shift_days
from kavachmed.linkage.strict import StrictCluster, StrictResult

NodeKey = tuple[int, Any]  # (0, canonical_key) for V5 cluster | (1, RecKey) for orphan


class KeysProtocol(Protocol):
    """Protocol for A's ``security.Keys``."""

    def keyed(self, purpose: str, *fields: str) -> bytes: ...


@dataclass(frozen=True)
class FamilyResult:
    """Result of temporal-family grouping and date-shift derivation."""

    family_of: dict[RecKey, str]
    shift_days: dict[str, int]
    candidates_accepted: int = 0
    candidates_rejected: int = 0
    candidates_capped: int = 0
    newly_grouped_unlinked: int = 0
    joins_skipped_veto: int = 0
    joins_skipped_cap: int = 0
    joins_skipped_budget: int = 0


def format_cluster_id(canonical_key: tuple[int, str, str]) -> str:
    kind, ns, core = canonical_key
    if kind == 0:
        return f"id:{ns}:{core}"
    if kind == 1:
        return f"adm:{core}"
    return f"scored:{core}"


def _cluster_shift_fields(canonical_key: tuple[int, str, str]) -> tuple[str, ...]:
    kind, ns, core = canonical_key
    if kind == 0:
        return ("cluster", "id", ns, core)
    if kind == 1:
        return ("cluster", "adm", core)
    return ("cluster", "scored", core)


def _cluster_anchor_fields(canonical_key: tuple[int, str, str]) -> tuple[str, ...]:
    kind, ns, core = canonical_key
    if kind == 0:
        return ("id", ns, core)
    if kind == 1:
        return ("adm", core)
    return ("scored", core)


def _shared_encounter_context(ev_a: Evidence, ev_b: Evidence) -> bool:
    ctx_a = getattr(ev_a, "encounter_context", None) or ev_a.adm_key
    ctx_b = getattr(ev_b, "encounter_context", None) or ev_b.adm_key
    return bool(ctx_a and ctx_b and ctx_a == ctx_b)


def _has_temporal_corroboration(
    ev_a: Evidence,
    ev_b: Evidence,
    trace: PairTrace,
    rk_a: RecKey,
    rk_b: RecKey,
) -> bool:
    for contrib in trace.contributions:
        if contrib.feature == "relative" and contrib.level == "agrees":
            return True
        if contrib.feature == "birth_interval" and contrib.level == "overlaps":
            return True
    if rk_a[0] == rk_b[0] and _shared_encounter_context(ev_a, ev_b):
        return True
    return False


def build_temporal_families(
    evidences: list[Evidence],
    strict_result: StrictResult,
    final_clusters: list[Any],
    keys: KeysProtocol,
    *,
    policy: Mapping[str, Any] | LinkagePolicy | Any = None,
    file_digests: Mapping[str, str] | None = None,
    diagnostics: dict[str, int] | None = None,
) -> FamilyResult:
    """Group V5 clusters and unlinked records into bounded temporal families."""
    from kavachmed.linkage import LinkageError

    cfg = parse_linkage_policy(policy)
    digests = file_digests or {}
    sorted_keys = sorted(strict_result.record_keys)

    record_to_canonical: dict[RecKey, tuple[int, str, str]] = {}
    cluster_by_canonical: dict[tuple[int, str, str], Any] = {}
    for cluster in final_clusters:
        cluster_by_canonical[cluster.canonical_key] = cluster
        for rk in cluster.members:
            record_to_canonical[rk] = cluster.canonical_key

    # Validate file digests upfront for all orphan records.
    for rk in sorted_keys:
        if rk not in record_to_canonical:
            rel_path, _loc = rk
            if not digests.get(rel_path):
                raise LinkageError("missing_file_digest")

    if cfg.family_mode == "single_graph":
        family_of: dict[RecKey, str] = {}
        shift_days: dict[str, int] = {}
        for ckey in sorted(cluster_by_canonical):
            cid = format_cluster_id(ckey)
            shift_days[cid] = derive_shift_days(keys, *_cluster_shift_fields(ckey))
        for rk in sorted_keys:
            ckey = record_to_canonical.get(rk)
            if ckey is not None:
                family_of[rk] = format_cluster_id(ckey)
            else:
                rel_path, loc = rk
                digest_hex = digests[rel_path]
                fam_id = f"orphan:{rel_path}\x00{loc}"
                family_of[rk] = fam_id
                shift_days[fam_id] = derive_shift_days(keys, "orphan", digest_hex, loc)
        return FamilyResult(family_of=family_of, shift_days=shift_days)

    # Temporal-family mode: seed indivisible V5 cluster nodes + singleton orphan nodes.
    by_key: dict[RecKey, Evidence] = {ev.key: ev for ev in evidences}
    shared_set = set(strict_result.shared_phones)

    node_of_rk: dict[RecKey, NodeKey] = {}
    all_nodes: list[NodeKey] = []
    comp_members: dict[NodeKey, list[RecKey]] = {}
    comp_clusters: dict[NodeKey, list[tuple[int, str, str]]] = {}
    comp_orphans: dict[NodeKey, list[RecKey]] = {}
    comp_node_count: dict[NodeKey, int] = {}

    for ckey in sorted(cluster_by_canonical):
        node: NodeKey = (0, ckey)
        all_nodes.append(node)
        members = sorted(cluster_by_canonical[ckey].members)
        comp_members[node] = members
        comp_clusters[node] = [ckey]
        comp_orphans[node] = []
        comp_node_count[node] = 1
        for rk in members:
            node_of_rk[rk] = node

    for rk in sorted_keys:
        if rk not in node_of_rk:
            node = (1, rk)
            all_nodes.append(node)
            comp_members[node] = [rk]
            comp_clusters[node] = []
            comp_orphans[node] = [rk]
            comp_node_count[node] = 1
            node_of_rk[rk] = node

    # Reuse V5 bounded candidate generator, then add bounded (file, identifying_skeleton) block.
    base_blocking = generate_candidates(
        evidences,
        policy=cfg,
        contested_bridges=strict_result.contested_bridges,
    )
    pair_budget = (
        cfg.pair_budget
        if cfg.pair_budget is not None
        else cfg.pair_budget_factor * len(sorted_keys)
    )
    candidates: list[tuple[RecKey, RecKey]] = list(base_blocking.candidates)
    seen_pairs: set[tuple[RecKey, RecKey]] = set(candidates)
    candidates_capped = base_blocking.skipped_by_cap

    file_skel_buckets: dict[tuple[str, str], list[RecKey]] = {}
    for rk in sorted_keys:
        ev = by_key[rk]
        for skel in _identifying_skeletons(ev.names):
            file_skel_buckets.setdefault((rk[0], skel), []).append(rk)

    for bkey in sorted(file_skel_buckets):
        members = file_skel_buckets[bkey]
        if len(members) < 2:
            continue
        if len(members) > cfg.max_bucket_size:
            candidates_capped += len(members) * (len(members) - 1) // 2
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                pair = (a, b) if a < b else (b, a)
                if pair in seen_pairs:
                    continue
                if len(candidates) < pair_budget:
                    seen_pairs.add(pair)
                    candidates.append(pair)
                else:
                    candidates_capped += 1

    pair_cache: dict[tuple[RecKey, RecKey], PairTrace] = {}

    def _score(ra: RecKey, rb: RecKey) -> PairTrace:
        pair = (ra, rb) if ra < rb else (rb, ra)
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

    skel_cache: dict[RecKey, set[str]] = {
        rk: set(_identifying_skeletons(by_key[rk].names)) for rk in sorted_keys
    }

    eligible_edges: list[tuple[float, NodeKey, NodeKey]] = []
    candidates_accepted = 0
    candidates_rejected = 0

    for rk_a, rk_b in candidates:
        na = node_of_rk[rk_a]
        nb = node_of_rk[rk_b]
        if na == nb:
            continue
        if not (skel_cache[rk_a] & skel_cache[rk_b]):
            candidates_rejected += 1
            continue
        tr = _score(rk_a, rk_b)
        if tr.veto_code is not None or round(tr.total_bits, 6) < cfg.tau_low_bits:
            candidates_rejected += 1
            continue
        if not _has_temporal_corroboration(by_key[rk_a], by_key[rk_b], tr, rk_a, rk_b):
            candidates_rejected += 1
            continue
        candidates_accepted += 1
        u, v = (na, nb) if na < nb else (nb, na)
        eligible_edges.append((-round(tr.total_bits, 6), u, v))

    eligible_edges.sort(key=lambda item: (item[0], item[1], item[2]))

    parent: dict[NodeKey, NodeKey] = {node: node for node in all_nodes}

    def _find(x: NodeKey) -> NodeKey:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pair_checks = 0
    joins_skipped_veto = 0
    joins_skipped_cap = 0
    joins_skipped_budget = 0

    for _neg_score, u, v in eligible_edges:
        ru = _find(u)
        rv = _find(v)
        if ru == rv:
            continue
        if len(comp_members[ru]) + len(comp_members[rv]) > cfg.max_component_size:
            joins_skipped_cap += 1
            continue

        veto_found = False
        budget_exceeded = False
        for ra in comp_members[ru]:
            for rb in comp_members[rv]:
                pair = (ra, rb) if ra < rb else (rb, ra)
                if pair not in pair_cache:
                    if pair_checks >= cfg.max_pair_checks:
                        budget_exceeded = True
                        break
                    pair_checks += 1
                    tr = _score(ra, rb)
                else:
                    tr = pair_cache[pair]
                if tr.veto_code is not None:
                    veto_found = True
                    break
            if veto_found or budget_exceeded:
                break

        if budget_exceeded:
            joins_skipped_budget += 1
            continue
        if veto_found:
            joins_skipped_veto += 1
            continue

        keep, drop = (ru, rv) if ru < rv else (rv, ru)
        parent[drop] = keep
        comp_members[keep] = sorted(comp_members[keep] + comp_members[drop])
        comp_clusters[keep] = sorted(comp_clusters[keep] + comp_clusters[drop])
        comp_orphans[keep] = sorted(comp_orphans[keep] + comp_orphans[drop])
        comp_node_count[keep] += comp_node_count[drop]
        del comp_members[drop]
        del comp_clusters[drop]
        del comp_orphans[drop]
        del comp_node_count[drop]

    family_of = {}
    shift_days = {}
    newly_grouped_unlinked = 0

    for root in sorted(comp_members):
        members = comp_members[root]
        clusters = comp_clusters[root]
        orphans = comp_orphans[root]
        if comp_node_count[root] == 1:
            if clusters:
                ckey = clusters[0]
                fam_id = format_cluster_id(ckey)
                shift_days[fam_id] = derive_shift_days(keys, *_cluster_shift_fields(ckey))
            else:
                rel_path, loc = orphans[0]
                digest_hex = digests[rel_path]
                fam_id = f"orphan:{rel_path}\x00{loc}"
                shift_days[fam_id] = derive_shift_days(keys, "orphan", digest_hex, loc)
        else:
            newly_grouped_unlinked += len(orphans)
            if clusters:
                min_ckey = clusters[0]
                fam_id = f"fam:{format_cluster_id(min_ckey)}"
                shift_days[fam_id] = derive_shift_days(
                    keys, "family", *_cluster_anchor_fields(min_ckey)
                )
            else:
                rel_path, loc = orphans[0]
                digest_hex = digests[rel_path]
                fam_id = f"fam:rec:{rel_path}\x00{loc}"
                shift_days[fam_id] = derive_shift_days(
                    keys, "family", "rec", digest_hex, loc
                )

        for rk in members:
            family_of[rk] = fam_id

    if diagnostics is not None:
        diagnostics["family_candidates_accepted"] = candidates_accepted
        diagnostics["family_candidates_rejected"] = candidates_rejected
        diagnostics["family_candidates_capped"] = candidates_capped
        diagnostics["family_newly_grouped_unlinked"] = newly_grouped_unlinked
        diagnostics["family_joins_skipped_veto"] = joins_skipped_veto
        diagnostics["family_joins_skipped_cap"] = joins_skipped_cap
        diagnostics["family_joins_skipped_budget"] = joins_skipped_budget

    return FamilyResult(
        family_of=family_of,
        shift_days=shift_days,
        candidates_accepted=candidates_accepted,
        candidates_rejected=candidates_rejected,
        candidates_capped=candidates_capped,
        newly_grouped_unlinked=newly_grouped_unlinked,
        joins_skipped_veto=joins_skipped_veto,
        joins_skipped_cap=joins_skipped_cap,
        joins_skipped_budget=joins_skipped_budget,
    )
