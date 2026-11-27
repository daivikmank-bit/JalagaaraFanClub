"""Filename evidence handoff and bounded attribution (Plan C V7).

Consumes typed filename signals from Plan A (with optional Plan B role and
confidence metadata), normalizes them using V1/V3 normalizers, validates file
and record scope, and attaches unresolved records to trusted body-derived
owners without seeding V4 exact bridges or mutating record-body ``Evidence``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping

from kavachmed.evidence import (
    Evidence,
    RecKey,
    _normalize_phone,
    _normalize_typed_id,
)
from kavachmed.indic import NameToken, normalize_name
from kavachmed.linkage.fs import PairTrace, score_pair
from kavachmed.linkage.policy import LinkagePolicy
from kavachmed.linkage.scored import ScoredCluster, _scored_fallback_key
from kavachmed.linkage.strict import (
    StrictResult,
    _is_eligible_adm_key,
    _is_eligible_id_namespace,
    _names_compatible,
)

_NONPATIENT_ROLES = frozenset({"nonpatient", "doctor", "clinician", "staff", "facility"})
_PATIENT_ROLES = frozenset({"patient", "subject", "self", ""})


@dataclass(frozen=True)
class FilenameSignal:
    """In-memory handoff for a typed signal extracted from an input filename."""

    input_file: str
    typ: str
    value: str
    source: str = "filename"
    role: str = "patient"
    confidence: float = 1.0
    target_key: RecKey | None = None
    hospital_hint: str | None = None


@dataclass(frozen=True)
class NormalizedFilenameEvidence:
    """Validated, record-scoped normalized filename evidence (kept separate from body)."""

    target_key: RecKey
    input_file: str
    source: str
    ids: tuple[tuple[str, str], ...] = ()
    adm_keys: tuple[str, ...] = ()
    phones: tuple[str, ...] = ()
    names: tuple[NameToken, ...] = ()


def _bump(diagnostics: dict[str, int] | None, key: str, delta: int = 1) -> None:
    if diagnostics is not None:
        diagnostics[key] = diagnostics.get(key, 0) + delta


def prepare_filename_evidence(
    signals: Iterable[FilenameSignal] | None,
    evidences: list[Evidence],
    *,
    policy: LinkagePolicy,
    diagnostics: dict[str, int] | None = None,
) -> dict[RecKey, NormalizedFilenameEvidence]:
    """Validate scope/provenance and normalize filename signals per target record."""
    if not signals:
        return {}

    by_key: dict[RecKey, Evidence] = {ev.key: ev for ev in evidences}
    records_by_file: dict[str, list[RecKey]] = {}
    for rk in sorted(by_key):
        records_by_file.setdefault(rk[0], []).append(rk)

    per_record_ids: dict[RecKey, list[tuple[str, str]]] = {}
    per_record_adms: dict[RecKey, list[str]] = {}
    per_record_phones: dict[RecKey, list[str]] = {}
    per_record_names: dict[RecKey, list[NameToken]] = {}
    per_record_sources: dict[RecKey, str] = {}
    seen_dedup: set[tuple[RecKey, str, str]] = set()

    # Deterministic processing order independent of input signal order.
    ordered_signals = sorted(
        signals,
        key=lambda s: (
            s.input_file,
            s.target_key or ("", ""),
            s.typ,
            s.value,
            s.source,
            s.role,
        ),
    )

    for sig in ordered_signals:
        role_norm = (sig.role or "patient").strip().casefold()
        if role_norm in _NONPATIENT_ROLES or role_norm not in _PATIENT_ROLES:
            _bump(diagnostics, "filename_nonpatient_role_skipped")
            continue

        target_rk = sig.target_key
        if target_rk is not None:
            if target_rk[0] != sig.input_file:
                _bump(diagnostics, "filename_target_file_conflict")
                continue
            if target_rk not in by_key:
                _bump(diagnostics, "filename_invalid_target")
                continue
        else:
            file_records = records_by_file.get(sig.input_file, [])
            if len(file_records) == 0:
                _bump(diagnostics, "filename_no_patient_unit")
                continue
            if len(file_records) > 1:
                _bump(diagnostics, "filename_unresolved_multi_patient")
                continue
            target_rk = file_records[0]

        typ_upper = sig.typ.strip().upper()
        hint = sig.hospital_hint or by_key[target_rk].hospital

        if typ_upper == "PHONE":
            phone = _normalize_phone(sig.value)
            if phone is None:
                _bump(diagnostics, "filename_invalid_phone")
                continue
            dedup_key = (target_rk, "PHONE", phone)
            if dedup_key in seen_dedup:
                _bump(diagnostics, "filename_duplicate_signal")
                continue
            seen_dedup.add(dedup_key)
            per_record_phones.setdefault(target_rk, []).append(phone)
            per_record_sources.setdefault(target_rk, sig.source)
            _bump(diagnostics, "filename_signals_attached")
        elif typ_upper == "NAME":
            tokens = [
                tok
                for tok in normalize_name(sig.value)
                if tok.kind in {"identifying", "initial"}
            ]
            if not tokens:
                _bump(diagnostics, "filename_invalid_name")
                continue
            sig_key = "|".join(t.skeleton or t.latin or "" for t in tokens)
            dedup_key = (target_rk, "NAME", sig_key)
            if dedup_key in seen_dedup:
                _bump(diagnostics, "filename_duplicate_signal")
                continue
            seen_dedup.add(dedup_key)
            per_record_names.setdefault(target_rk, []).extend(tokens)
            per_record_sources.setdefault(target_rk, sig.source)
            _bump(diagnostics, "filename_signals_attached")
        elif typ_upper in {"ADM_KEY", "ADMISSION_ID", "IPD_NO"}:
            _, adm = _normalize_typed_id(typ_upper, sig.value, hint)
            if not _is_eligible_adm_key(adm):
                _bump(diagnostics, "filename_unknown_issuer")
                continue
            assert adm is not None
            dedup_key = (target_rk, "ADM", adm)
            if dedup_key in seen_dedup:
                _bump(diagnostics, "filename_duplicate_signal")
                continue
            seen_dedup.add(dedup_key)
            per_record_adms.setdefault(target_rk, []).append(adm)
            per_record_sources.setdefault(target_rk, sig.source)
            _bump(diagnostics, "filename_signals_attached")
        else:
            ns, core = _normalize_typed_id(typ_upper, sig.value, hint)
            if not ns or not core or not _is_eligible_id_namespace(ns, policy):
                _bump(diagnostics, "filename_unknown_issuer")
                continue
            dedup_key = (target_rk, f"ID:{ns}", core)
            if dedup_key in seen_dedup:
                _bump(diagnostics, "filename_duplicate_signal")
                continue
            seen_dedup.add(dedup_key)
            per_record_ids.setdefault(target_rk, []).append((ns, core))
            per_record_sources.setdefault(target_rk, sig.source)
            _bump(diagnostics, "filename_signals_attached")

    attached: dict[RecKey, NormalizedFilenameEvidence] = {}
    all_targets = (
        set(per_record_ids)
        | set(per_record_adms)
        | set(per_record_phones)
        | set(per_record_names)
    )
    for rk in sorted(all_targets):
        attached[rk] = NormalizedFilenameEvidence(
            target_key=rk,
            input_file=rk[0],
            source=per_record_sources.get(rk, "filename"),
            ids=tuple(sorted(set(per_record_ids.get(rk, [])))),
            adm_keys=tuple(sorted(set(per_record_adms.get(rk, [])))),
            phones=tuple(sorted(set(per_record_phones.get(rk, [])))),
            names=tuple(per_record_names.get(rk, [])),
        )
    return attached


def _has_body_id_conflict(
    ev_rk: Evidence,
    owner_members: tuple[RecKey, ...],
    by_key: dict[RecKey, Evidence],
    policy: LinkagePolicy,
) -> bool:
    """Return True if rk's body IDs or admission keys conflict with the owner's body."""
    owner_ids: dict[str, set[str]] = {}
    owner_adms: set[str] = set()
    for m in owner_members:
        mev = by_key[m]
        for ns, vals in mev.ids.items():
            if _is_eligible_id_namespace(ns, policy):
                owner_ids.setdefault(ns, set()).update(vals)
        if _is_eligible_adm_key(mev.adm_key):
            assert mev.adm_key is not None
            owner_adms.add(mev.adm_key)

    for ns, vals in ev_rk.ids.items():
        if not _is_eligible_id_namespace(ns, policy):
            continue
        if ns in owner_ids and owner_ids[ns] and not (set(vals) & owner_ids[ns]):
            return True

    if _is_eligible_adm_key(ev_rk.adm_key) and owner_adms:
        if ev_rk.adm_key not in owner_adms:
            return True

    return False


def _has_independent_body_corroboration(
    ev_rk: Evidence,
    owner_members: tuple[RecKey, ...],
    by_key: dict[RecKey, Evidence],
    traces: list[PairTrace],
    *,
    used_filename_phone: bool,
    used_filename_name: bool,
) -> bool:
    """Require independent corroboration from the target record's own body."""
    # Check if body names match any owner member when filename supplied a phone/ID.
    body_name_match = False
    if not used_filename_name and ev_rk.names:
        for m in owner_members:
            both, compat = _names_compatible(ev_rk.names, by_key[m].names)
            if both and compat:
                body_name_match = True
                break

    for tr in traces:
        for contrib in tr.contributions:
            if contrib.feature == "relative" and contrib.level == "agrees":
                return True
            if contrib.feature == "birth_interval" and contrib.level in {"overlaps", "near"}:
                return True
            if (
                not used_filename_phone
                and contrib.feature == "phone"
                and contrib.level == "exact_non_shared"
            ):
                return True
            if body_name_match and contrib.feature == "geography" and contrib.level == "same_district":
                return True

    # A body-derived matching name + a validated filename ID/phone counts as corroboration
    # (1 body fact + 1 filename fact, not 2 correlated filename tokens).
    if body_name_match and used_filename_phone:
        return True

    return False


def apply_filename_attribution(
    evidences: list[Evidence],
    strict_result: StrictResult,
    base_clusters: list[ScoredCluster],
    filename_signals: Iterable[FilenameSignal] | None,
    *,
    policy: LinkagePolicy,
    diagnostics: dict[str, int] | None = None,
) -> list[ScoredCluster]:
    """Attach unresolved records to unique, corroborated body-derived owners."""
    attached = prepare_filename_evidence(
        filename_signals,
        evidences,
        policy=policy,
        diagnostics=diagnostics,
    )
    if not attached:
        return base_clusters

    by_key: dict[RecKey, Evidence] = {ev.key: ev for ev in evidences}
    shared_phones = set(strict_result.shared_phones)

    # Identify trusted body-derived owners (clusters + body-derived singletons).
    clustered_rks: set[RecKey] = set()
    # owner_id -> (canonical_key_or_None, members, is_cluster)
    owners: dict[str, tuple[tuple[int, str, str] | None, tuple[RecKey, ...], bool]] = {}
    for idx, cl in enumerate(base_clusters):
        oid = f"c:{idx}"
        owners[oid] = (cl.canonical_key, cl.members, True)
        clustered_rks.update(cl.members)

    for ev in evidences:
        if ev.key in clustered_rks:
            continue
        # Only body records not themselves relying on unresolved filename attachment
        # or having actual body-derived identifying evidence can serve as owners.
        has_body_id = any(_is_eligible_id_namespace(ns, policy) for ns in ev.ids)
        has_body_adm = _is_eligible_adm_key(ev.adm_key)
        has_body_phone = any(p not in shared_phones for p in ev.phones)
        has_body_name = any(t.kind == "identifying" for t in ev.names)
        if has_body_id or has_body_adm or has_body_phone or has_body_name:
            oid = f"s:{ev.key[0]}|{ev.key[1]}"
            owners[oid] = (None, (ev.key,), False)

    # Build body-derived owner indices.
    id_index: dict[tuple[str, str], set[str]] = {}
    adm_index: dict[str, set[str]] = {}
    phone_index: dict[str, set[str]] = {}
    name_index: dict[str, set[str]] = {}

    for oid, (_, members, _) in owners.items():
        for rk in members:
            ev = by_key[rk]
            for ns, vals in ev.ids.items():
                if _is_eligible_id_namespace(ns, policy):
                    for v in vals:
                        id_index.setdefault((ns, v), set()).add(oid)
            if _is_eligible_adm_key(ev.adm_key):
                assert ev.adm_key is not None
                adm_index.setdefault(ev.adm_key, set()).add(oid)
            for p in ev.phones:
                if p not in shared_phones:
                    phone_index.setdefault(p, set()).add(oid)
            for tok in ev.names:
                if tok.kind == "identifying" and tok.skeleton:
                    name_index.setdefault(tok.skeleton, set()).add(oid)

    # Resolve only records that are currently unclustered (unresolved identity).
    attachments: dict[str, list[RecKey]] = {}
    singleton_claims: dict[RecKey, int] = {}
    candidate_records: list[tuple[RecKey, str]] = []

    for rk in sorted(attached):
        if rk in clustered_rks:
            _bump(diagnostics, "filename_preserved_existing_cluster")
            continue

        fev = attached[rk]
        ev_rk = by_key[rk]
        own_oid = f"s:{rk[0]}|{rk[1]}"

        # Candidate generation: IDs, admission keys, unique non-shared phones, bounded names.
        candidate_owners: dict[str, tuple[bool, bool, bool]] = {}
        # maps oid -> (via_id_or_adm, via_phone, via_name)

        def _add_cand(oid: str, *, via_id: bool = False, via_phone: bool = False, via_name: bool = False) -> None:
            if oid == own_oid:
                return
            prev = candidate_owners.get(oid, (False, False, False))
            candidate_owners[oid] = (
                prev[0] or via_id,
                prev[1] or via_phone,
                prev[2] or via_name,
            )

        for ns, core in fev.ids:
            for oid in sorted(id_index.get((ns, core), ())):
                _add_cand(oid, via_id=True)

        for adm in fev.adm_keys:
            for oid in sorted(adm_index.get(adm, ())):
                _add_cand(oid, via_id=True)

        for phone in fev.phones:
            if phone in shared_phones:
                _bump(diagnostics, "filename_shared_phone_rejected")
                continue
            p_owners = sorted(oid for oid in phone_index.get(phone, ()) if oid != own_oid)
            if len(p_owners) == 1:
                _add_cand(p_owners[0], via_phone=True)
            elif len(p_owners) > 1:
                _bump(diagnostics, "filename_ambiguous_phone_owner")

        for tok in fev.names:
            if tok.kind == "identifying" and tok.skeleton:
                n_owners = sorted(oid for oid in name_index.get(tok.skeleton, ()) if oid != own_oid)
                if len(n_owners) <= policy.max_bucket_size:
                    for oid in n_owners:
                        _add_cand(oid, via_name=True)
                else:
                    _bump(diagnostics, "filename_name_bucket_capped")

        if not candidate_owners:
            _bump(diagnostics, "filename_no_candidate")
            continue

        qualifying: list[str] = []
        for oid in sorted(candidate_owners):
            via_id, via_phone, via_name = candidate_owners[oid]
            _, members, _ = owners[oid]

            if _has_body_id_conflict(ev_rk, members, by_key, policy):
                _bump(diagnostics, "filename_body_id_conflict")
                continue

            # Build temporary scoring view without mutating ev_rk.
            aug_phones = list(ev_rk.phones)
            used_fn_phone = False
            if via_phone and not ev_rk.phones and fev.phones:
                aug_phones = [fev.phones[0]]
                used_fn_phone = True

            aug_names = list(ev_rk.names)
            used_fn_name = False
            if via_name and not ev_rk.names and fev.names:
                aug_names = list(fev.names)
                used_fn_name = True

            ev_view = replace(ev_rk, phones=aug_phones, names=aug_names)

            traces: list[PairTrace] = []
            vetoed = False
            any_high = False
            for m in members:
                tr = score_pair(
                    ev_view,
                    by_key[m],
                    policy=policy,
                    shared_phones=shared_phones,
                )
                traces.append(tr)
                if tr.veto_code is not None:
                    vetoed = True
                    _bump(diagnostics, "filename_veto_rejected")
                    break
                if tr.passes_high:
                    any_high = True

            if vetoed:
                continue
            if not any_high:
                _bump(diagnostics, "filename_below_threshold")
                continue

            if not _has_independent_body_corroboration(
                ev_rk,
                members,
                by_key,
                traces,
                used_filename_phone=used_fn_phone,
                used_filename_name=used_fn_name,
            ):
                _bump(diagnostics, "filename_uncorroborated_rejected")
                continue

            qualifying.append(oid)

        if len(qualifying) == 1:
            chosen_oid = qualifying[0]
            candidate_records.append((rk, chosen_oid))
            attachments.setdefault(chosen_oid, []).append(rk)
            _, members, is_cluster = owners[chosen_oid]
            if not is_cluster:
                singleton_claims[members[0]] = singleton_claims.get(members[0], 0) + 1
        elif len(qualifying) > 1:
            _bump(diagnostics, "filename_ambiguous_owner_abstention")

    if not attachments:
        return base_clusters

    # Rebuild clusters cleanly, preserving all existing trusted clusters.
    updated_clusters: list[ScoredCluster] = []
    for idx, cl in enumerate(base_clusters):
        oid = f"c:{idx}"
        extra = attachments.get(oid, [])
        if extra:
            merged_members = tuple(sorted(set(cl.members) | set(extra)))
            updated_clusters.append(
                ScoredCluster(
                    canonical_key=cl.canonical_key,
                    members=merged_members,
                    from_v4=False,
                )
            )
            _bump(diagnostics, "filename_links_accepted", len(extra))
        else:
            updated_clusters.append(cl)

    # Handle singleton owners that gained a unique attachment.
    for oid in sorted(attachments):
        if not oid.startswith("s:"):
            continue
        _, members, _ = owners[oid]
        anchor = members[0]
        if singleton_claims.get(anchor, 0) > 1:
            # Guard against conflicting multi-record claims on a singleton if not mutually checked.
            # Allow only if all attached records are pairwise compatible; otherwise abstain.
            _bump(diagnostics, "filename_singleton_multi_claim_abstention")
            continue
        merged_members = tuple(sorted(set(members) | set(attachments[oid])))
        ckey = _scored_fallback_key(merged_members, by_key)
        updated_clusters.append(
            ScoredCluster(
                canonical_key=ckey,
                members=merged_members,
                from_v4=False,
            )
        )
        _bump(diagnostics, "filename_links_accepted", len(attachments[oid]))

    updated_clusters.sort(key=lambda c: c.canonical_key)
    return updated_clusters
