"""B12: Global ID-core/exact sweeps and cluster-scoped name propagation (FR-D7)."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from kavachmed.contracts import RecKey, Record, Span

_DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_RE_TOKEN = re.compile(r"[A-Za-z]{2,}|[\u0900-\u097F]{2,}")
_RE_DIGIT_RUN = re.compile(r"(?:\(\d{2,5}\)[ \t]*)?\d[\d \t\-./()]{4,}\d")

_GENERIC_SINGLE_TOKENS = {
    "devi", "kumar", "kumari", "singh", "sharma", "verma", "gupta", "patel",
    "yadav", "khan", "begum", "lal", "prasad", "md", "mohd", "ji", "sahab",
    "देवी", "कुमार", "कुमारी", "सिंह", "शर्मा", "वर्मा", "जी", "साहब",
}


def _tok_hash(tokens: tuple[str, ...]) -> str:
    return hashlib.sha256(" ".join(tokens).encode("utf-8")).hexdigest()[:16]


def _overlaps_protected(field_idx: int, start: int, end: int, protected: list[Span]) -> bool:
    return any(p.field_idx == field_idx and not (end <= p.start or start >= p.end) for p in protected)


def _overlaps_any(field_idx: int, start: int, end: int, existing: list[Span]) -> bool:
    return any(s.field_idx == field_idx and not (end <= s.start or start >= s.end) for s in existing)


@dataclass
class PropagationIndex:
    """Corpus-wide sweep index built once per run."""

    phones_10: set[str] = field(default_factory=set)
    id_cores: dict[str, str] = field(default_factory=dict)
    exact_strings: dict[str, str] = field(default_factory=dict)
    cluster_name_hashes: dict[str, set[str]] = field(default_factory=dict)
    cluster_of: dict[RecKey, str | None] = field(default_factory=dict)
    family_of: dict[RecKey, str] = field(default_factory=dict)

    def scope_for(self, key: RecKey) -> str:
        return self.cluster_of.get(key) or self.family_of.get(key) or f"file:{key[0]}"


def build_index(
    records: list[Record],
    resolution: Any,
    spans_by_rec: dict[RecKey, list[Span]],
) -> PropagationIndex:
    """Build reusable global and cluster-scoped sweep indexes once per run."""
    cluster_of = getattr(resolution, "cluster_of", {}) if resolution is not None else {}
    family_of = getattr(resolution, "family_of", {}) if resolution is not None else {}
    idx = PropagationIndex(cluster_of=dict(cluster_of), family_of=dict(family_of))

    for rec in records:
        fields_by_idx = {f.idx: f.text for f in rec.fields}
        scope = idx.scope_for(rec.key)
        name_hashes = idx.cluster_name_hashes.setdefault(scope, set())

        for sp in spans_by_rec.get(rec.key, []):
            seg = fields_by_idx.get(sp.field_idx, "")[sp.start : sp.end].strip()
            if not seg:
                continue
            norm_digits = re.sub(r"\D", "", seg.translate(_DEVA_DIGITS))

            if sp.typ == "PHONE" and len(norm_digits) >= 10:
                idx.phones_10.add(norm_digits[-10:])
            elif sp.typ in {"MRN", "AADHAAR", "AADHAAR_VID", "ABHA_NUMBER"} and len(norm_digits) >= 5:
                idx.id_cores.setdefault(norm_digits, sp.typ)
                for t in re.findall(r"\d{5,}", seg.translate(_DEVA_DIGITS)):
                    idx.id_cores.setdefault(t, sp.typ)
            elif sp.typ in {"PAN", "EMAIL", "ABHA_ADDRESS", "DOB", "HOUSE", "VILLAGE", "LOCALITY", "LANDMARK"} and len(seg) >= 4:
                idx.exact_strings.setdefault(seg.lower(), sp.typ)

            if sp.typ in {"PATIENT_NAME", "RELATIVE_NAME"}:
                toks = [m.group().lower() for m in _RE_TOKEN.finditer(seg)]
                for n in (1, 2, 3):
                    for i in range(len(toks) - n + 1):
                        gram = tuple(toks[i : i + n])
                        if n == 1 and (len(gram[0]) < 3 or gram[0] in _GENERIC_SINGLE_TOKENS):
                            continue
                        name_hashes.add(_tok_hash(gram))

    if resolution is not None:
        try:
            setattr(resolution, "_prop_index", idx)
        except Exception:
            pass
    return idx


def sweep(
    rec: Record,
    resolution: Any,
    spans: list[Span],
    protected_spans: list[Span] | None = None,
    index: PropagationIndex | None = None,
) -> list[Span]:
    """Sweep a single record using the prebuilt corpus index (or a local fallback)."""
    if index is None and resolution is not None:
        index = getattr(resolution, "_prop_index", None)
    if index is None:
        index = build_index([rec], resolution, {rec.key: spans})

    prot = protected_spans or []
    scope = index.scope_for(rec.key)
    name_hashes = index.cluster_name_hashes.get(scope, set())
    existing = list(spans)
    added: list[Span] = []

    for fld in rec.fields:
        text = fld.text
        norm_text = text.translate(_DEVA_DIGITS)

        # 1. Global phone (any separator) & ID-core sweeps (skipping protected spans!)
        for m in _RE_DIGIT_RUN.finditer(norm_text):
            s, e = m.span()
            if _overlaps_protected(fld.idx, s, e, prot):
                continue
            digits = re.sub(r"\D", "", m.group())
            if len(digits) >= 10 and digits[-10:] in index.phones_10:
                if not _overlaps_any(fld.idx, s, e, existing + added):
                    added.append(Span(rec.key, fld.idx, s, e, "PHONE", "unknown", 0.90, "propagate"))
            elif digits in index.id_cores:
                if not _overlaps_any(fld.idx, s, e, existing + added):
                    added.append(Span(rec.key, fld.idx, s, e, index.id_cores[digits], "unknown", 0.90, "propagate"))

        # 2. Global exact string sweeps (DOB, PAN, email, removed address parts)
        low_text = text.lower()
        for needle, typ in sorted(index.exact_strings.items(), key=lambda kv: (-len(kv[0]), kv[0])):
            pos = 0
            while True:
                pos_idx = low_text.find(needle, pos)
                if pos_idx == -1:
                    break
                end = pos_idx + len(needle)
                if not _overlaps_protected(fld.idx, pos_idx, end, prot) and not _overlaps_any(fld.idx, pos_idx, end, existing + added):
                    added.append(Span(rec.key, fld.idx, pos_idx, end, typ, "unknown", 0.85, "propagate"))
                pos = end

        # 3. Cluster-scoped name 1-3-gram sweeps (protected spans skipped)
        if name_hashes:
            tok_matches = list(_RE_TOKEN.finditer(text))
            for n in (3, 2, 1):
                for i in range(len(tok_matches) - n + 1):
                    window = tok_matches[i : i + n]
                    gram = tuple(m.group().lower() for m in window)
                    if _tok_hash(gram) not in name_hashes:
                        continue
                    s, e = window[0].start(), window[-1].end()
                    if _overlaps_protected(fld.idx, s, e, prot):
                        continue
                    if _overlaps_any(fld.idx, s, e, existing + added):
                        continue
                    added.append(Span(rec.key, fld.idx, s, e, "PATIENT_NAME", "unknown", 0.80, "propagate"))

    combined = existing + added
    combined.sort(key=lambda sp: (sp.field_idx, sp.start, sp.end, sp.typ))
    return combined


def sweep_corpus(
    records: list[Record],
    resolution_or_cluster_of: Any,
    spans_by_rec: dict[RecKey, list[Span]],
    protected_by_rec: dict[RecKey, list[Span]] | None = None,
) -> dict[RecKey, list[Span]]:
    """Build index once and sweep every record in the corpus."""
    protected_by_rec = protected_by_rec or {}
    if isinstance(resolution_or_cluster_of, dict):
        class _Res:
            def __init__(self, c):
                self.cluster_of = c
                self.family_of = {}
        resolution: Any = _Res(resolution_or_cluster_of)
    else:
        resolution = resolution_or_cluster_of

    index = build_index(records, resolution, spans_by_rec)
    return {
        rec.key: sweep(
            rec,
            resolution,
            spans_by_rec.get(rec.key, []),
            protected_spans=protected_by_rec.get(rec.key, []),
            index=index,
        )
        for rec in records
    }
