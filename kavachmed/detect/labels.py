"""B14: Label induction (FR-D5) and static promoted-label artefact for Layer B."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from kavachmed.contracts import Context, Field, Record, Span
from kavachmed.detect.ids import detect_ids

_RE_KV_LINE = re.compile(
    r"^[ \t]*([A-Za-z\u0900-\u097F][A-Za-z0-9_ \t\u0900-\u097F./-]{1,32})[ \t]*[:=][ \t]*(.+?)[ \t]*$",
    re.M,
)
_RE_NAME_SHAPE = re.compile(
    r"^(?:[A-Z][a-z]{1,25}(?:[ \t]+[A-Z][a-z]{1,25}){0,3}|[\u0900-\u097F]{2,}(?:[ \t]+[\u0900-\u097F]{2,}){0,3})$"
)

_SEED_LABELS: dict[str, str] = {
    "name": "PATIENT_NAME",
    "patient name": "PATIENT_NAME",
    "pt name": "PATIENT_NAME",
    "naam": "PATIENT_NAME",
    "नाम": "PATIENT_NAME",
    "relative": "RELATIVE_NAME",
    "guardian": "RELATIVE_NAME",
    "father name": "RELATIVE_NAME",
    "husband name": "RELATIVE_NAME",
    "phone": "PHONE",
    "mobile": "PHONE",
    "contact": "PHONE",
    "मोबाइल": "PHONE",
    "aadhaar": "AADHAAR",
    "aadhar": "AADHAAR",
    "abha": "ABHA_NUMBER",
    "pan": "PAN",
    "email": "EMAIL",
    "uhid": "MRN",
    "mrn": "MRN",
    "address": "LOCALITY",
    "pata": "LOCALITY",
    "पता": "LOCALITY",
}

_NON_PII_LABELS = {
    "bp", "temp", "pulse", "spo2", "weight", "height", "bmi", "diagnosis", "dx",
    "rx", "plan", "advice", "ward", "bed", "date", "time", "age", "sex", "gender",
    "hb", "wbc", "rbs", "notes", "note", "complaint", "complaints", "history",
}


def _norm_label(lbl: str) -> str:
    return re.sub(r"[ \t_\-]+", " ", lbl.strip().lower())


def _bigram_cosine(a: str, b: str) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    ba = {a[i : i + 2] for i in range(len(a) - 1)}
    bb = {b[i : i + 2] for i in range(len(b) - 1)}
    return len(ba & bb) / ((len(ba) * len(bb)) ** 0.5)


def _edit_distance(a: str, b: str) -> int:
    if abs(len(a) - len(b)) > 2:
        return 3
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            dp[j] = prev if ca == cb else 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[-1]


def _looks_like_pii_value(val: str, ctx: Context) -> str | None:
    v = val.strip()
    if not v or v.lower() in {"na", "n/a", "none", "nil", "normal", "negative", "positive"}:
        return None
    rec = Record(key=("_induce", "0"), fields=[Field(0, "v", v)])
    id_spans = detect_ids(rec, ctx)
    if id_spans:
        return id_spans[0].typ
    if _RE_NAME_SHAPE.match(v):
        safe_words: set[str] = ctx.lexicons.get("safe_words", set())
        toks = [t.lower() for t in v.split()]
        if any(t not in safe_words for t in toks):
            return "PATIENT_NAME"
    return None


def _role_for_typ(typ: str) -> str:
    if typ == "RELATIVE_NAME":
        return "relative"
    if typ == "PATIENT_NAME":
        return "patient"
    return "unknown"


def induce_labels(records: list[Record], ctx: Context) -> dict[str, str]:
    """Promote labels/keys/headers where >=60% of >=3 values look like PII or close to a seed."""
    values_by_label: dict[str, list[str]] = {}

    for rec in records:
        for fld in rec.fields:
            if fld.vtype in {"path", "header", "key"}:
                lbl = _norm_label(fld.text)
                if lbl and lbl not in _NON_PII_LABELS:
                    values_by_label.setdefault(lbl, [])
                continue
            if fld.path and fld.path not in {"doc", "note", "text"}:
                lbl = _norm_label(fld.path)
                if lbl and lbl not in _NON_PII_LABELS:
                    values_by_label.setdefault(lbl, []).append(fld.text)
            for m in _RE_KV_LINE.finditer(fld.text):
                lbl = _norm_label(m.group(1))
                if lbl and lbl not in _NON_PII_LABELS:
                    values_by_label.setdefault(lbl, []).append(m.group(2))

    promoted: dict[str, str] = {}
    for lbl in sorted(values_by_label):
        if lbl in _SEED_LABELS:
            promoted[lbl] = _SEED_LABELS[lbl]
            continue

        if len(lbl) >= 4:
            for seed, typ in sorted(_SEED_LABELS.items()):
                if len(seed) >= 4 and (_edit_distance(lbl, seed) <= 2 or _bigram_cosine(lbl, seed) >= 0.72):
                    promoted[lbl] = typ
                    break
        if lbl in promoted:
            continue

        vals = [v for v in values_by_label[lbl] if v.strip()]
        if len(vals) >= 3:
            hits = [t for v in vals if (t := _looks_like_pii_value(v, ctx)) is not None]
            if len(hits) / len(vals) >= 0.60:
                counts = Counter(hits)
                promoted[lbl] = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    return promoted


def write_promoted_labels(promoted: dict[str, str], out_path: Path) -> None:
    """Write static per-run promoted label artefact for Layer B check B3."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "kavachmed-promoted-labels/1", "labels": dict(sorted(promoted.items()))}
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def detect_labels(rec: Record, ctx: Context, promoted_labels: dict[str, str] | None = None) -> list[Span]:
    """Emit spans for values under induced/promoted labels (including unknown/canary tokens)."""
    labels = promoted_labels if promoted_labels is not None else ctx.lexicons.get("promoted_labels", {})
    if not labels:
        return []

    spans: list[Span] = []
    for fld in rec.fields:
        if fld.vtype in {"path", "header", "key"}:
            continue
        norm_path = _norm_label(fld.path)
        if norm_path in labels and fld.text.strip():
            typ = labels[norm_path]
            spans.append(
                Span(
                    key=rec.key,
                    field_idx=fld.idx,
                    start=0,
                    end=len(fld.text),
                    typ=typ,
                    role=_role_for_typ(typ),  # type: ignore[arg-type]
                    score=0.80,
                    src="labels",
                )
            )
            continue

        for m in _RE_KV_LINE.finditer(fld.text):
            lbl = _norm_label(m.group(1))
            if lbl in labels:
                typ = labels[lbl]
                spans.append(
                    Span(
                        key=rec.key,
                        field_idx=fld.idx,
                        start=m.start(2),
                        end=m.end(2),
                        typ=typ,
                        role=_role_for_typ(typ),  # type: ignore[arg-type]
                        score=0.80,
                        src="labels",
                    )
                )

    return spans
