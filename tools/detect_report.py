"""B11: Per-category and per-script detection report + E14 stack comparison."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from kavachmed.contracts import Context, Record, Span
from kavachmed.detect.arbiter import detect
from kavachmed.shield import protect


def _script_of(text: str) -> str:
    has_deva = any("\u0900" <= c <= "\u097f" for c in text)
    has_latin = any("A" <= c <= "Z" or "a" <= c <= "z" for c in text)
    if has_deva and has_latin:
        return "mixed"
    if has_deva:
        return "devanagari"
    if has_latin:
        return "latin"
    return "numeric"


def build_report(records: Iterable[Record], ctx: Context) -> dict[str, dict[str, int]]:
    """Summarize detected spans by type, source, and script."""
    by_typ: Counter[str] = Counter()
    by_src: Counter[str] = Counter()
    by_script: Counter[str] = Counter()
    protected_count = 0

    for rec in records:
        sh = protect(rec, ctx)
        protected_count += sum(1 for s in sh if s.protected)
        spans = detect(rec, ctx, sh)
        fields_by_idx = {f.idx: f.text for f in rec.fields}
        for sp in spans:
            by_typ[sp.typ] += 1
            by_src[sp.src] += 1
            seg = fields_by_idx.get(sp.field_idx, "")[sp.start : sp.end]
            by_script[_script_of(seg)] += 1

    return {
        "by_type": dict(sorted(by_typ.items())),
        "by_src": dict(sorted(by_src.items())),
        "by_script": dict(sorted(by_script.items())),
        "protected_spans": {"total": protected_count},
    }


def compare_profiles(records: list[Record], base_ctx: Context) -> dict[str, dict[str, dict[str, int]]]:
    """Compare recall_first vs precision_first (E14)."""
    out: dict[str, dict[str, dict[str, int]]] = {}
    for profile in ("recall_first", "precision_first"):
        pol = dict(base_ctx.policy)
        pol["detect"] = {**pol.get("detect", {}), "stack": profile}
        ctx = Context(
            policy=pol,
            lexicons=base_ctx.lexicons,
            coltypes=base_ctx.coltypes,
            doctor_registry=base_ctx.doctor_registry,
            keys=base_ctx.keys,
        )
        out[profile] = build_report(records, ctx)
    return out
