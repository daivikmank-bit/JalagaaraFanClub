"""Strict exact-evidence bridging, conflict vetoes, and shared-phone detection (Plan C V4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from kavachmed.evidence import Evidence, RecKey
from kavachmed.indic import NameToken

_DEFAULT_EXACT_NAMESPACES = frozenset({"aadhaar", "abha", "pan"})
_DEFAULT_MAX_PAIR_CHECKS = 2000
_DEFAULT_MAX_COMPONENT_SIZE = 200
_MAX_BIRTH_YEAR_GAP = 3


@dataclass(frozen=True)
class StrictCluster:
    """An accepted exact-evidence cluster with its deterministic canonical key."""

    canonical_key: tuple[int, str, str]  # (0, ns, core) for strong ID, (1, "adm", adm_key)
    members: tuple[RecKey, ...]
    bridge_keys: tuple[tuple[int, str, str], ...]


@dataclass(frozen=True)
class StrictResult:
    """Result of strict exact-evidence linkage."""

    record_keys: tuple[RecKey, ...]
    clusters: tuple[StrictCluster, ...]
    shared_phones: tuple[str, ...]
    phone_shared_records: tuple[RecKey, ...]
    contested_bridges: tuple[tuple[int, str, str], ...] = ()
    fuzzy_ineligible: frozenset[RecKey] = frozenset()


def _policy_get(policy: Mapping[str, Any] | Any, key: str, default: Any = None) -> Any:
    if policy is None:
        return default
    if isinstance(policy, Mapping):
        if key in policy:
            return policy[key]
        linkage_cfg = policy.get("linkage")
        if isinstance(linkage_cfg, Mapping) and key in linkage_cfg:
            return linkage_cfg[key]
        return default
    if hasattr(policy, key):
        return getattr(policy, key)
    linkage_cfg = getattr(policy, "linkage", None)
    if linkage_cfg is not None and hasattr(linkage_cfg, key):
        return getattr(linkage_cfg, key)
    return default


def _is_eligible_id_namespace(ns: str, policy: Any) -> bool:
    """Check whether an ID namespace is allowed as an exact bridge."""
    disabled = set(_policy_get(policy, "disabled_id_namespaces", ()))
    if ns in disabled:
        return False

    # Masked fragments and unscoped insurance IDs never form exact bridges.
    if ns in {"aadhaar_last4", "ins"}:
        return False

    extra_allowed = set(_policy_get(policy, "allowed_id_namespaces", ()))
    if ns in extra_allowed:
        return True

    if ns in _DEFAULT_EXACT_NAMESPACES:
        return True

    if ns.startswith("uhid:"):
        issuer = ns.split(":", 1)[1]
        if not issuer:
            return False
        year_reused_issuers = set(_policy_get(policy, "year_reused_uhid_issuers", ()))
        if issuer in year_reused_issuers:
            return False
        return True

    return False


def _is_eligible_adm_key(adm_key: str | None) -> bool:
    if not adm_key or ":" not in adm_key:
        return False
    issuer, core = adm_key.split(":", 1)
    return bool(issuer and core)


def _tokens_compatible(a: NameToken, b: NameToken) -> bool:
    if a.skeleton and b.skeleton and a.skeleton == b.skeleton:
        return True
    if a.latin and b.latin and a.latin == b.latin:
        return True
    # Initial compatibility: one is an initial matching the leading letter of the other.
    a_init = a.kind == "initial" or (a.latin is not None and len(a.latin) == 1)
    b_init = b.kind == "initial" or (b.latin is not None and len(b.latin) == 1)
    if (a_init or b_init) and a.latin and b.latin:
        return a.latin[0] == b.latin[0]
    return False


def _names_compatible(
    tokens_a: list[NameToken],
    tokens_b: list[NameToken],
) -> tuple[bool, bool]:
    """Return ``(both_have_identifying_evidence, compatible)``."""
    id_a = [t for t in tokens_a if t.kind in {"identifying", "initial"}]
    id_b = [t for t in tokens_b if t.kind in {"identifying", "initial"}]
    if not id_a or not id_b:
        return False, True

    # Require at least one non-initial identifying token on each side before calling a mismatch,
    # unless both are initials and differ.
    for ta in id_a:
        for tb in id_b:
            if _tokens_compatible(ta, tb):
                return True, True
    return True, False


def _birth_year_gap(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(0, a[0] - b[1], b[0] - a[1])


def _pair_has_veto(ev_a: Evidence, ev_b: Evidence, *, id_bridged: bool = False) -> tuple[bool, str | None]:
    """Check strict hard vetoes between two records.

    Hard vetoes:
    - conflicting non-None sex
    - birth-year intervals separated by > 3 years
    - conflicting identifying relatives
    - dissimilar patient names corroborated by another conflict (already caught above,
      or dissimilar patient names when sex/age/kin conflict is present).
    """
    if ev_a.sex is not None and ev_b.sex is not None and ev_a.sex != ev_b.sex:
        return True, "sex_conflict"

    if ev_a.birth_years is not None and ev_b.birth_years is not None:
        if _birth_year_gap(ev_a.birth_years, ev_b.birth_years) > _MAX_BIRTH_YEAR_GAP:
            return True, "birth_year_conflict"

    # Different STATED relations (W/o husband vs S/o-attendant son, or C/O) are not a conflict:
    # a patient legitimately has several relatives, and comparing them vetoed whole UHID
    # groups. Same or unknown relation types still veto, so homonyms with different
    # relatives stay apart.
    known_diff = (
        ev_a.kin_type is not None and ev_b.kin_type is not None
        and (ev_a.kin_type != ev_b.kin_type or ev_a.kin_type == "care_of")
    )
    # Records already joined by a non-contested strong ID: only a conflict on the SAME stated
    # relation vetoes (unseen label variants leave relation types unknown). Name-only scored
    # pairs (fs.py) keep the full relative veto, so homonyms stay apart.
    # Records joined by a non-contested strong ID: relatives are not compared at all (relative
    # mentions vary across documents: father, husband, attendant, misspellings). Sex and
    # birth-year vetoes above still apply. Name-only scored pairs keep the relative veto.
    if not id_bridged and not known_diff:
        kin_both, kin_compat = _names_compatible(ev_a.kin, ev_b.kin)
        if kin_both and not kin_compat:
            return True, "relative_conflict"

    return False, None


def _group_has_conflict(
    member_keys: list[RecKey],
    by_key: dict[RecKey, Evidence],
    max_pair_checks: int,
    id_bridged: bool = True,
) -> tuple[bool, str | None]:
    """Check a bucket or component for conflicts using O(N) summaries then bounded pairwise checks."""
    n = len(member_keys)
    if n < 2:
        return False, None

    # O(N) summary checks for sex and birth-year intervals.
    seen_sex: str | None = None
    max_lo: int | None = None
    min_hi: int | None = None
    for rk in member_keys:
        ev = by_key[rk]
        if ev.sex is not None:
            if seen_sex is None:
                seen_sex = ev.sex
            elif seen_sex != ev.sex:
                return True, "sex_conflict"
        if ev.birth_years is not None:
            lo, hi = ev.birth_years
            max_lo = lo if max_lo is None else max(max_lo, lo)
            min_hi = hi if min_hi is None else min(min_hi, hi)
            if max_lo - min_hi > _MAX_BIRTH_YEAR_GAP:
                return True, "birth_year_conflict"

    # Every ID-bridged pair veto is already covered by the summaries above.
    if id_bridged:
        return False, None

    # Bounded pairwise checks for relative conflicts.
    pair_count = n * (n - 1) // 2
    if pair_count > max_pair_checks:
        return True, "over_budget"

    for i in range(n):
        ev_i = by_key[member_keys[i]]
        for j in range(i + 1, n):
            ev_j = by_key[member_keys[j]]
            vetoed, reason = _pair_has_veto(ev_i, ev_j, id_bridged=id_bridged)
            if vetoed:
                return True, reason

    return False, None


class _UnionFind:
    def __init__(self, keys: list[RecKey]) -> None:
        self.parent: dict[RecKey, RecKey] = {k: k for k in keys}

    def find(self, x: RecKey) -> RecKey:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        cur = x
        while cur != root:
            nxt = self.parent[cur]
            self.parent[cur] = root
            cur = nxt
        return root

    def union(self, a: RecKey, b: RecKey) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if ra < rb:
            self.parent[rb] = ra
        else:
            self.parent[ra] = rb


def build_strict_clusters(
    evidences: list[Evidence],
    *,
    policy: Mapping[str, Any] | Any = None,
    diagnostics: dict[str, int] | None = None,
) -> StrictResult:
    """Build strict exact-evidence clusters and detect shared phones."""
    from kavachmed.linkage import LinkageError

    by_key: dict[RecKey, Evidence] = {}
    for ev in evidences:
        if ev.key in by_key:
            raise LinkageError("duplicate_record_key")
        by_key[ev.key] = ev

    sorted_keys = sorted(by_key)
    max_pair_checks = int(_policy_get(policy, "max_pair_checks", _DEFAULT_MAX_PAIR_CHECKS))
    max_component_size = int(_policy_get(policy, "max_component_size", _DEFAULT_MAX_COMPONENT_SIZE))

    # Index eligible strong IDs and admission keys.
    strong_buckets: dict[tuple[str, str], list[RecKey]] = {}
    adm_buckets: dict[str, list[RecKey]] = {}
    phone_buckets: dict[str, list[RecKey]] = {}

    for rk in sorted_keys:
        ev = by_key[rk]
        for ns in sorted(ev.ids):
            if not _is_eligible_id_namespace(ns, policy):
                continue
            # A record holding several distinct IDs of one issuer is a roster (patient index,
            # ward census), not one person; letting it bridge chains different patients.
            if len({c for c in ev.ids[ns] if c}) > 1:
                if diagnostics is not None:
                    diagnostics["roster_records_skipped"] = diagnostics.get("roster_records_skipped", 0) + 1
                continue
            for core in sorted(ev.ids[ns]):
                if core:
                    strong_buckets.setdefault((ns, core), []).append(rk)
        if _is_eligible_adm_key(ev.adm_key):
            assert ev.adm_key is not None
            adm_buckets.setdefault(ev.adm_key, []).append(rk)
        for ph in sorted(ev.phones):
            if ph:
                phone_buckets.setdefault(ph, []).append(rk)

    # Shared phone detection (never mutates caller Evidence; never creates/vetoes bridges).
    shared_phones: list[str] = []
    phone_shared_records_set: set[RecKey] = set()
    for ph in sorted(phone_buckets):
        members = phone_buckets[ph]
        if len(members) < 2:
            continue
        is_shared = False
        # Bounded check across distinct name signatures.
        limit = min(len(members), 50)
        for i in range(limit):
            for j in range(i + 1, limit):
                both_id, compat = _names_compatible(
                    by_key[members[i]].names,
                    by_key[members[j]].names,
                )
                if both_id and not compat:
                    is_shared = True
                    break
            if is_shared:
                break
        if is_shared:
            shared_phones.append(ph)
            phone_shared_records_set.update(members)

    # Evaluate each bridge bucket for contested evidence before creating edges.
    uf = _UnionFind(sorted_keys)
    record_strong_keys: dict[RecKey, list[tuple[int, str, str]]] = {rk: [] for rk in sorted_keys}
    record_adm_keys: dict[RecKey, list[tuple[int, str, str]]] = {rk: [] for rk in sorted_keys}
    contested_bridges: list[tuple[int, str, str]] = []
    fuzzy_ineligible: set[RecKey] = set()

    for ns, core in sorted(strong_buckets):
        members = strong_buckets[(ns, core)]
        if len(members) < 2:
            continue
        conflict, reason = _group_has_conflict(members, by_key, max_pair_checks)
        if conflict:
            contested_bridges.append((0, ns, core))
            fuzzy_ineligible.update(members)
            if diagnostics is not None:
                diagnostics["contested_bridges"] = diagnostics.get("contested_bridges", 0) + 1
                if reason == "over_budget":
                    diagnostics["over_budget_abstentions"] = (
                        diagnostics.get("over_budget_abstentions", 0) + 1
                    )
            continue
        anchor = members[0]
        bridge_tuple = (0, ns, core)
        for rk in members:
            uf.union(anchor, rk)
            record_strong_keys[rk].append(bridge_tuple)

    for adm_key in sorted(adm_buckets):
        members = adm_buckets[adm_key]
        if len(members) < 2:
            continue
        conflict, reason = _group_has_conflict(members, by_key, max_pair_checks)
        if conflict:
            fuzzy_ineligible.update(members)
            if diagnostics is not None:
                diagnostics["contested_bridges"] = diagnostics.get("contested_bridges", 0) + 1
                if reason == "over_budget":
                    diagnostics["over_budget_abstentions"] = (
                        diagnostics.get("over_budget_abstentions", 0) + 1
                    )
            continue
        anchor = members[0]
        bridge_tuple = (1, "adm", adm_key)
        for rk in members:
            uf.union(anchor, rk)
            record_adm_keys[rk].append(bridge_tuple)

    # Group candidate components and run cross-bucket veto checks.
    raw_components: dict[RecKey, list[RecKey]] = {}
    for rk in sorted_keys:
        raw_components.setdefault(uf.find(rk), []).append(rk)

    accepted_clusters: list[StrictCluster] = []
    for root in sorted(raw_components):
        members = sorted(raw_components[root])
        if len(members) < 2:
            continue
        if len(members) > max_component_size:
            fuzzy_ineligible.update(members)
            if diagnostics is not None:
                diagnostics["over_budget_abstentions"] = (
                    diagnostics.get("over_budget_abstentions", 0) + 1
                )
            continue

        conflict, reason = _group_has_conflict(members, by_key, max_pair_checks)
        if conflict:
            fuzzy_ineligible.update(members)
            if diagnostics is not None:
                if reason == "over_budget":
                    diagnostics["over_budget_abstentions"] = (
                        diagnostics.get("over_budget_abstentions", 0) + 1
                    )
                else:
                    diagnostics["cross_bucket_veto_abstentions"] = (
                        diagnostics.get("cross_bucket_veto_abstentions", 0) + 1
                    )
            continue

        strong_candidates = sorted(
            {bt for rk in members for bt in record_strong_keys[rk]}
        )
        adm_candidates = sorted(
            {bt for rk in members for bt in record_adm_keys[rk]}
        )
        all_bridges = strong_candidates + adm_candidates
        if not all_bridges:
            continue
        canonical = strong_candidates[0] if strong_candidates else adm_candidates[0]
        accepted_clusters.append(
            StrictCluster(
                canonical_key=canonical,
                members=tuple(members),
                bridge_keys=tuple(all_bridges),
            )
        )

    accepted_clusters.sort(key=lambda c: c.canonical_key)

    if diagnostics is not None:
        diagnostics["shared_phones"] = len(shared_phones)
        diagnostics["phone_shared_records"] = len(phone_shared_records_set)
        diagnostics["strict_clusters"] = len(accepted_clusters)

    return StrictResult(
        record_keys=tuple(sorted_keys),
        clusters=tuple(accepted_clusters),
        shared_phones=tuple(shared_phones),
        phone_shared_records=tuple(sorted(phone_shared_records_set)),
        contested_bridges=tuple(sorted(contested_bridges)),
        fuzzy_ineligible=frozenset(fuzzy_ineligible),
    )
