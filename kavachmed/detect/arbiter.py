"""B11: Profile-aware span arbitration and unified detection entry point (FR-D10)."""

from __future__ import annotations

from kavachmed.contracts import Context, Record, Span
from kavachmed.detect.address import segment_address
from kavachmed.detect.context import detect_context
from kavachmed.detect.default_deny import detect_default_deny
from kavachmed.detect.gazetteer import detect_gazetteer
from kavachmed.detect.hidden import detect_hidden
from kavachmed.detect.ids import detect_ids
from kavachmed.detect.labels import detect_labels
from kavachmed.shield import protect

import re as _re

# Our own release tokens: never re-detect a pseudonym or a typed placeholder as PII.
_RE_RELEASE_TOKEN = _re.compile(r"\bPSN-[0-9A-Z]{6,12}\b|\[[A-Z0-9_/-]{2,24}\]")
_EDGE_JUNK = " \t\r\n,;:.-/|()"

_PRECEDENCE = {
    "ids": 4,
    "shield": 3,
    "context": 2,
    "propagate": 2,
    "labels": 1,
    "crf": 1,
    "gazetteer": 1,
    "default_deny": 0,
}


def _overlaps(a: Span, b: Span) -> bool:
    return a.key == b.key and a.field_idx == b.field_idx and not (a.end <= b.start or a.start >= b.end)


def _trim_around_protected(sp: Span, protected: list[Span]) -> list[Span]:
    """Split/trim a non-identifier candidate around any overlapping protected spans."""
    intervals = [(sp.start, sp.end)]
    for p in protected:
        if p.key != sp.key or p.field_idx != sp.field_idx:
            continue
        next_intervals: list[tuple[int, int]] = []
        for s, e in intervals:
            if p.end <= s or p.start >= e:
                next_intervals.append((s, e))
            else:
                if s < p.start:
                    next_intervals.append((s, p.start))
                if p.end < e:
                    next_intervals.append((p.end, e))
        intervals = next_intervals
    return [
        Span(
            key=sp.key,
            field_idx=sp.field_idx,
            start=s,
            end=e,
            typ=sp.typ,
            role=sp.role,
            score=sp.score,
            src=sp.src,
            protected=sp.protected,
            exempt_kind=sp.exempt_kind,
        )
        for s, e in intervals
        if e > s
    ]


def arbitrate(
    candidates: list[Span],
    shield_spans: list[Span],
    ctx: Context,
) -> list[Span]:
    """Resolve overlapping spans with precedence: ids > shield > context > longest/highest."""
    detect_cfg = ctx.policy.get("detect", {})
    stack = detect_cfg.get("stack", "recall_first")
    threshold = float(
        detect_cfg.get("precision_threshold", 0.55)
        if stack == "precision_first"
        else detect_cfg.get("recall_threshold", 0.35)
    )

    protected = [s for s in shield_spans if s.protected] + [s for s in candidates if s.protected]
    filtered: list[Span] = []
    for sp in candidates:
        if sp.protected:
            continue
        if stack == "precision_first" and sp.src in {"default_deny", "gazetteer"}:
            continue
        if sp.score < threshold:
            continue
        if sp.src != "ids" and any(_overlaps(sp, p) for p in protected):
            filtered.extend(_trim_around_protected(sp, protected))
        else:
            filtered.append(sp)

    ordered = sorted(
        filtered,
        key=lambda s: (
            -_PRECEDENCE.get(s.src, 0),
            -(s.end - s.start),
            -s.score,
            s.field_idx,
            s.start,
            s.end,
            s.typ,
        ),
    )

    accepted: list[Span] = []
    for sp in ordered:
        accepted.extend(_trim_around_protected(sp, accepted))

    accepted.sort(key=lambda s: (s.field_idx, s.start, s.end, s.typ))
    return accepted


def _clean_spans(rec: Record, spans: list[Span]) -> list[Span]:
    """Trim separator/punctuation edges, drop letter-less spans and spans inside release tokens.

    Trimming around higher-precedence spans can leave fragments such as ", " or
    ", Basti"; as name spans they glue neighbouring placeholders together.
    """
    reserved: dict[int, list[tuple[int, int]]] = {}
    for fld in rec.fields:
        reserved[fld.idx] = [(m.start(), m.end()) for m in _RE_RELEASE_TOKEN.finditer(fld.text)]
    out: list[Span] = []
    for sp in spans:
        if sp.protected or sp.field_idx >= len(rec.fields):
            out.append(sp)
            continue
        text = rec.fields[sp.field_idx].text
        s, e = sp.start, sp.end
        if sp.src != "ids":
            while s < e and text[s] in _EDGE_JUNK:
                s += 1
            while e > s and text[e - 1] in _EDGE_JUNK and not (text[e - 1] == "." and e - s <= 3):
                e -= 1
        if e <= s or not any(ch.isalnum() for ch in text[s:e]):
            continue
        if any(rs <= s and e <= re_ for rs, re_ in reserved.get(sp.field_idx, [])):
            continue
        if (s, e) == (sp.start, sp.end):
            out.append(sp)
        else:
            out.append(Span(key=sp.key, field_idx=sp.field_idx, start=s, end=e, typ=sp.typ, role=sp.role,
                            score=sp.score, src=sp.src, protected=sp.protected, exempt_kind=sp.exempt_kind))
    return out


def detect(rec: Record, ctx: Context, shield_spans: list[Span] | None = None) -> list[Span]:
    """Run full detector stack and return arbitrated PII spans."""
    if shield_spans is None:
        shield_spans = protect(rec, ctx)

    id_spans = detect_ids(rec, ctx)
    ctx_spans = detect_context(rec, ctx)
    addr_spans = segment_address(rec, ctx)
    addr_protected = [s for s in addr_spans if s.protected]
    all_protected = shield_spans + addr_protected

    gaz_spans = detect_gazetteer(rec, ctx)
    hid_spans = detect_hidden(rec, ctx)
    lbl_spans = detect_labels(rec, ctx)
    pre_dd = id_spans + ctx_spans + addr_spans + gaz_spans + hid_spans + lbl_spans
    dd_spans = detect_default_deny(rec, ctx, all_protected, pre_dd)

    return _clean_spans(rec, arbitrate(pre_dd + dd_spans, all_protected, ctx))
