"""B7: Gazetteer features (never a gate) for Latin/Devanagari names and LGD places."""

from __future__ import annotations

import re

from kavachmed.contracts import Context, Record, Span

_RE_TOKEN = re.compile(r"[A-Za-z]{2,}|[\u0900-\u097F]{2,}")


def is_place_token(token: str, ctx: Context) -> bool:
    """Return True if token is in the LGD place lexicon."""
    places: set[str] = ctx.lexicons.get("places", set())
    return token.lower() in places


def is_name_token(token: str, ctx: Context) -> bool:
    """Return True if token is in the name gazetteer and not a clinical safe word."""
    low = token.lower()
    names: set[str] = ctx.lexicons.get("names", set())
    safe_words: set[str] = ctx.lexicons.get("safe_words", set())
    return low in names and low not in safe_words


def detect_gazetteer(rec: Record, ctx: Context) -> list[Span]:
    """Emit soft-scored name feature spans (feature, never a gate)."""
    names: set[str] = ctx.lexicons.get("names", set())
    if not names:
        return []
    safe_words: set[str] = ctx.lexicons.get("safe_words", set())

    spans: list[Span] = []
    for fld in rec.fields:
        if fld.vtype in {"path", "header", "key"}:
            continue
        text = fld.text
        hits: list[tuple[int, int]] = []
        for m in _RE_TOKEN.finditer(text):
            tok = m.group()
            low = tok.lower()
            if low in safe_words:
                continue
            if low in names:
                hits.append((m.start(), m.end()))

        # Merge adjacent gazetteer hits into multi-token name spans with higher score
        merged: list[tuple[int, int, int]] = []
        for s, e in hits:
            if merged and text[merged[-1][1] : s].strip() == "":
                ms, _, cnt = merged[-1]
                merged[-1] = (ms, e, cnt + 1)
            else:
                merged.append((s, e, 1))

        for s, e, cnt in merged:
            score = 0.50 if cnt >= 2 else 0.36
            spans.append(
                Span(
                    key=rec.key,
                    field_idx=fld.idx,
                    start=s,
                    end=e,
                    typ="PATIENT_NAME",
                    role="unknown",
                    score=score,
                    src="gazetteer",
                )
            )
    return spans
