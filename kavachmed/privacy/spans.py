"""Owner D record-level privacy spans: dates (D1/D2), ages (D3), districts (D4).

Fills the gap between D's modules and A's edit builder for non-FHIR records.
The detector emits only generic redaction spans, so without these spans every
date, age and district would be redacted rather than shifted, banded or kept.

* ``DATE``/``DOB`` spans come from ``dates.find`` using the per-file date order
  from ``dates.infer_date_orders``. ``edits`` shifts DATE (family offset) and
  replaces DOB with ``[DOB]`` (FR-P4: DOB suppressed).
* ``AGE`` spans cover true patient ages (structured age columns, ``48 Y / F``,
  ``Age: 48``, ``आयु: 48 वर्ष``). Durations are skipped. ``edits`` swaps each
  one for the record's patient-level band from ``PrivacyPlan.age_band``.
* District names in district-typed fields, address fields, or after a
  ``Distt.``/``District``/``जिला`` cue get *protected* spans (``exempt_kind=
  "district"``) so the district survives when ``location.keep == district``
  (FR-P4). Village/tehsil/house/PIN stay unprotected, so they are redacted.

Everything is deterministic: spans come back sorted, and no set is iterated
unsorted.
"""
from __future__ import annotations

import re
from typing import Any, Literal

from kavachmed import dates
from kavachmed.contracts import Record, Span
from kavachmed.privacy import bands
from kavachmed.privacy.location import LGDIndex, get_lgd_index

_DISTRICT_COL_RE = re.compile(r"(?:^|[/_\s])(?:district|distt?|jila|zila|जिला|ज़िला)$", re.IGNORECASE)
_ADDRESS_COL_RE = re.compile(r"(?:address|addr|pata|patta|पता|residence|location)", re.IGNORECASE)
_DISTRICT_CUE_RE = re.compile(r"(?:Distt?\.?|District|Jila|Zila|जिला|ज़िला)\s*[:\-]?\s*$", re.IGNORECASE)
_SKIP_VTYPES = frozenset({"key", "header", "path"})

_DISTRICT_RE_CACHE: dict[int, re.Pattern[str] | None] = {}


def _district_regex(idx: LGDIndex) -> re.Pattern[str] | None:
    """One alternation over every district surface (Latin + Devanagari), longest first."""
    cached = _DISTRICT_RE_CACHE.get(id(idx))
    if cached is not None or id(idx) in _DISTRICT_RE_CACHE:
        return cached
    surfaces = sorted({k for k in idx.district_norm_to_info if len(k) >= 3}, key=lambda s: (-len(s), s))
    pat = None
    if surfaces:
        alt = "|".join(re.escape(s) for s in surfaces)
        pat = re.compile(rf"(?<![\wऀ-ॿ])(?:{alt})(?![\wऀ-ॿ])", re.IGNORECASE)
    _DISTRICT_RE_CACHE[id(idx)] = pat
    return pat


def summarizable(loc: str) -> bool:
    """Data records plus the JSON ``@meta`` record (top-level patient/collected fields).

    Filename, header and key channels (``@filename``/``@header``/``@keys``) never get
    date/age/district handling: they are renamed wholesale by the platform.
    """
    return not loc.startswith("@") or loc == "@meta"


def file_date_order(records: list[Record], policy: dict[str, Any]) -> Literal["dmy", "mdy"]:
    """Date order for one source file (header hint > unambiguous votes > hospital > default)."""
    pub = [r for r in records if not r.key[1].startswith("@")]
    if not pub:
        return "mdy" if str((policy.get("dates") or {}).get("default_order", "dmy")).lower() == "mdy" else "dmy"
    orders = dates.infer_date_orders(pub, policy)
    return orders.get(pub[0].key[0], "dmy")


def record_summary(rec: Record, idx: LGDIndex | None = None) -> dict[str, Any]:
    """Compact, PII-free QI summary kept from pass 1: true ages and the canonical district."""
    from kavachmed.privacy.location import extract_district_from_record

    lgd = idx or get_lgd_index()
    out: dict[str, Any] = {"ages": sorted(bands.extract_ages_from_record(rec))}
    d = extract_district_from_record(rec, lgd)
    if d and lgd.lookup_district(d):
        out["district"] = lgd.lookup_district(d)[1]  # type: ignore[index]
    return out


def _age_spans(rec: Record) -> list[Span]:
    spans: list[Span] = []
    for fld in rec.fields:
        if fld.vtype in _SKIP_VTYPES or not fld.text:
            continue
        txt = fld.text
        col = fld.path.split("/")[-1].strip()
        if bands._AGE_STRUCTURED_KEYS.match(col):
            m = re.fullmatch(r"\s*([0-9०-९]{1,3})(?:\s*(?:Y(?:rs?|ears?)?|वर्ष|साल))?\s*", txt, re.IGNORECASE)
            if m and bands._parse_int(m.group(1)) is not None:
                s, e = (0, len(txt)) if fld.vtype in ("int", "float") else (m.start(1), m.end(1))
                spans.append(_mk(rec, fld.idx, s, e, "AGE"))
                continue
        for rx in (bands._AGE_SLASH_SEX_RE, bands._AGE_ANCHOR_RE):
            for m in rx.finditer(txt):
                before = txt[max(0, m.start() - 16): m.start()]
                after = txt[m.end(): m.end() + 16]
                if bands._DURATION_BEFORE_RE.search(before):
                    continue
                if rx is bands._AGE_ANCHOR_RE and bands._DURATION_AFTER_RE.search(after):
                    continue
                if bands._parse_int(m.group("age")) is None:
                    continue
                spans.append(_mk(rec, fld.idx, m.start("age"), m.end("age"), "AGE"))
    return spans


_DATE_COL_RE = re.compile(r"date|dob|birth|janm|tithi|dt\b|admitted|discharged|reported|collected|registered", re.IGNORECASE)


def _date_spans(rec: Record, policy: dict[str, Any], order: Literal["dmy", "mdy"]) -> list[Span]:
    spans: list[Span] = []
    found_fields: set[int] = set()
    for ref in dates.find(rec, {"policy": policy}, order_override=order):
        fld = rec.fields[ref.field_idx]
        typ = "DOB" if dates._is_dob_context(fld.path, fld.text, ref.start) else "DATE"
        spans.append(_mk(rec, ref.field_idx, ref.start, ref.end, typ))
        found_fields.add(ref.field_idx)
    # Fail closed: a date-labelled column whose value the finder cannot parse (unseen format)
    # is still suppressed (DOB -> [DOB], other dates -> [DATE] placeholder via edits).
    for fld in rec.fields:
        if fld.idx in found_fields or fld.vtype in _SKIP_VTYPES or fld.vtype in ("int", "float"):
            continue
        col = fld.path.split("/")[-1]
        txt = fld.text
        if _DATE_COL_RE.search(col) and sum(ch.isdigit() for ch in txt) >= 4 and len(txt.strip()) <= 40:
            s = len(txt) - len(txt.lstrip()); e = len(txt.rstrip())
            typ = "DOB" if dates._is_dob_context(fld.path, txt, 0) else "DATE"
            spans.append(_mk(rec, fld.idx, s, e, typ))
    return spans


def _district_spans(rec: Record, idx: LGDIndex) -> list[Span]:
    pat = _district_regex(idx)
    if pat is None:
        return []
    spans: list[Span] = []
    for fld in rec.fields:
        if fld.vtype in _SKIP_VTYPES or not fld.text:
            continue
        col = fld.path.split("/")[-1].strip()
        district_col = bool(_DISTRICT_COL_RE.search(col))
        address_col = bool(_ADDRESS_COL_RE.search(col))
        for m in pat.finditer(fld.text):
            cue = bool(_DISTRICT_CUE_RE.search(fld.text[max(0, m.start() - 12): m.start()]))
            if district_col or address_col or cue:
                spans.append(Span(
                    key=rec.key, field_idx=fld.idx, start=m.start(), end=m.end(),
                    typ="DISTRICT", role="none", score=1.0, src="shield",
                    protected=True, exempt_kind="district",
                ))
    return spans


def _mk(rec: Record, fidx: int, s: int, e: int, typ: str) -> Span:
    return Span(key=rec.key, field_idx=fidx, start=s, end=e, typ=typ, role="patient",
                score=0.9, src="context", protected=False)


def privacy_spans(
    rec: Record,
    policy: dict[str, Any],
    order: Literal["dmy", "mdy"],
    idx: LGDIndex | None = None,
) -> list[Span]:
    """All D-owned spans for one record, sorted; ages that overlap a date are dropped."""
    if not summarizable(rec.key[1]):
        return []
    date_sp = _date_spans(rec, policy, order)
    age_sp = [a for a in _age_spans(rec)
              if not any(d.field_idx == a.field_idx and d.start < a.end and a.start < d.end for d in date_sp)]
    out = date_sp + age_sp
    keep = str((policy.get("location") or {}).get("keep", "district"))
    if keep == "district":
        out += _district_spans(rec, idx or get_lgd_index())
    return sorted(out, key=lambda s: (s.field_idx, s.start, s.end, s.typ))
