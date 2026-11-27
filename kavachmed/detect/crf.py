"""B15 (P1, cut-first): Optional CRF voter (weight <= 0.25, never sole evidence)."""

from __future__ import annotations

import re
from pathlib import Path

from kavachmed.contracts import Context, Record, Span

_RE_TOKEN = re.compile(r"[A-Za-z]{2,}|[\u0900-\u097F]{2,}")

try:
    import pycrfsuite  # type: ignore[import-not-found]
    HAS_CRFSUITE = True
except ImportError:
    pycrfsuite = None
    HAS_CRFSUITE = False


def token_features(tokens: list[str], idx: int, ctx: Context) -> dict[str, object]:
    """Extract shape, script, affix, and gazetteer features for a token."""
    tok = tokens[idx]
    low = tok.lower()
    names: set[str] = ctx.lexicons.get("names", set())
    places: set[str] = ctx.lexicons.get("places", set())
    safe_words: set[str] = ctx.lexicons.get("safe_words", set())
    feats: dict[str, object] = {
        "bias": 1.0,
        "is_title": tok[0].isupper() and tok[1:].islower(),
        "is_upper": tok.isupper(),
        "is_deva": "\u0900" <= tok[0] <= "\u097f",
        "in_names": low in names,
        "in_places": low in places,
        "in_safe": low in safe_words,
        "suf3": low[-3:] if len(low) >= 3 else low,
    }
    if idx > 0:
        feats["prev_low"] = tokens[idx - 1].lower()
    if idx + 1 < len(tokens):
        feats["next_low"] = tokens[idx + 1].lower()
    return feats


def detect_crf(rec: Record, ctx: Context, model_path: Path | None = None) -> list[Span]:
    """Run optional CRF tagger if pycrfsuite and model exist; weight capped at <= 0.25."""
    weight = min(0.25, float(ctx.policy.get("detect", {}).get("crf_weight", 0.25)))
    if not HAS_CRFSUITE or model_path is None or not model_path.exists():
        return []

    tagger = pycrfsuite.Tagger()
    tagger.open(str(model_path))
    spans: list[Span] = []

    for fld in rec.fields:
        if fld.vtype in {"path", "header", "key"}:
            continue
        matches = list(_RE_TOKEN.finditer(fld.text))
        if not matches:
            continue
        toks = [m.group() for m in matches]
        feats = [token_features(toks, i, ctx) for i in range(len(toks))]
        tags = tagger.tag(feats)
        for m, tag in zip(matches, tags):
            if tag in {"B-PER", "I-PER"}:
                spans.append(
                    Span(
                        key=rec.key,
                        field_idx=fld.idx,
                        start=m.start(),
                        end=m.end(),
                        typ="PATIENT_NAME",
                        role="unknown",
                        score=weight,
                        src="crf",
                    )
                )
    return spans
