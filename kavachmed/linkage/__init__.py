"""Exact-evidence linkage, keyed pseudonyms, and temporal-family shifts (Plan C V4-V6)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol

from kavachmed.evidence import Evidence, RecKey
from kavachmed.linkage.family import build_temporal_families, format_cluster_id
from kavachmed.linkage.filename import FilenameSignal, apply_filename_attribution
from kavachmed.linkage.policy import LinkagePolicy, parse_linkage_policy
from kavachmed.linkage.pseudo import assign_pseudonyms
from kavachmed.linkage.scored import build_scored_clusters
from kavachmed.linkage.strict import build_strict_clusters

try:
    from kavachmed.contracts import Resolution
except ImportError:  # pragma: no cover - fallback before Plan A scaffolds contracts.py

    @dataclass
    class Resolution:
        """Record-to-patient, cluster, family, and date-shift mapping."""

        patient_of: dict[RecKey, str | None]
        cluster_of: dict[RecKey, str | None]
        family_of: dict[RecKey, str]
        shift_days: dict[str, int]


class KeysProtocol(Protocol):
    """Protocol for A's ``security.Keys``."""

    def keyed(self, purpose: str, *fields: str) -> bytes: ...


class LinkageError(ValueError):
    """Text-free linkage error carrying only a closed reason code."""

    def __init__(self, code: str, detail: str | None = None) -> None:
        self.code = code
        msg = f"{code}: {detail}" if detail else code
        super().__init__(msg)


def resolve(
    evidences: list[Evidence],
    keys: KeysProtocol,
    policy: Mapping[str, Any] | Any = None,
    *,
    file_digests: Mapping[str, str] | None = None,
    filename_signals: Iterable[FilenameSignal] | None = None,
    diagnostics: dict[str, int] | None = None,
) -> Resolution:
    """Resolve ``Evidence`` objects into strict clusters, PSNs, and shift families."""
    if keys is None:
        raise LinkageError("missing_keys")

    cfg = parse_linkage_policy(policy)

    if not evidences:
        return Resolution(
            patient_of={},
            cluster_of={},
            family_of={},
            shift_days={},
        )

    strict_result = build_strict_clusters(
        evidences,
        policy=cfg,
        diagnostics=diagnostics,
    )

    if cfg.mode == "exact_only":
        final_clusters = list(strict_result.clusters)
        record_keys = strict_result.record_keys
    else:
        scored_result = build_scored_clusters(
            evidences,
            strict_result,
            policy=cfg,
            diagnostics=diagnostics,
        )
        final_clusters = list(scored_result.clusters)
        record_keys = scored_result.record_keys

    if filename_signals is not None:
        final_clusters = apply_filename_attribution(
            evidences,
            strict_result,
            final_clusters,
            filename_signals,
            policy=cfg,
            diagnostics=diagnostics,
        )

    # Check for canonical key collisions across distinct clusters before emitting PSNs.
    seen_canonical: set[tuple[int, str, str]] = set()
    for cluster in final_clusters:
        if cluster.canonical_key in seen_canonical:
            raise LinkageError("canonical_collision")
        seen_canonical.add(cluster.canonical_key)

    psn_by_canonical = assign_pseudonyms(
        [c.canonical_key for c in final_clusters],
        keys,
    )

    patient_of: dict[RecKey, str | None] = {}
    cluster_of: dict[RecKey, str | None] = {}

    record_to_canonical: dict[RecKey, tuple[int, str, str]] = {}
    for c in final_clusters:
        for rk in c.members:
            record_to_canonical[rk] = c.canonical_key

    family_result = build_temporal_families(
        evidences,
        strict_result,
        final_clusters,
        keys,
        policy=cfg,
        file_digests=file_digests,
        diagnostics=diagnostics,
    )

    for rk in sorted(record_keys):
        canonical = record_to_canonical.get(rk)
        if canonical is not None:
            cluster_id = format_cluster_id(canonical)
            patient_of[rk] = psn_by_canonical[canonical]
            cluster_of[rk] = cluster_id
        else:
            patient_of[rk] = None
            cluster_of[rk] = None

    return Resolution(
        patient_of=patient_of,
        cluster_of=cluster_of,
        family_of=family_result.family_of,
        shift_days=family_result.shift_days,
    )


__all__ = [
    "FilenameSignal",
    "LinkageError",
    "Resolution",
    "resolve",
]
