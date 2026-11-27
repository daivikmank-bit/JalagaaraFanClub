"""B2 & B3: Typed identifiers (FR-D1) and extended shapes (FR-D9)."""

from __future__ import annotations

import ipaddress
import re
from typing import Iterable

from kavachmed.contracts import Context, Record, Span
from kavachmed.local_ids import LABELED_LOCAL, parse_local_id, patient_id_header

_DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

# Verhoeff tables (used strictly as a confidence booster, never a gate)
_V_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
_V_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)


def _verhoeff_valid(digits: str) -> bool:
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _V_D[c][_V_P[i % 8][int(ch)]]
    return c == 0


# Negative-context regexes to protect clinical numerics (FR-D9)
_RE_CLIN_NUM_CTX = re.compile(
    r"\b(?:lab\s*seq|batch|lot|mmol/l|mg/dl|g/dl|iu/l|u/l|/cumm|pkg|package|dose|inr|rs\.?|rupees|bp|spo2|ref\s*range|range)\b",
    re.I,
)
_RE_POS_PHONE_CTX = re.compile(r"\b(?:phone|ph|mob|mobile|tel|telephone|contact|call|whatsapp)\b|फ़ोन|मोबाइल", re.I)
_RE_POS_AADHAAR_CTX = re.compile(r"(?:\baadhaar|\baadhar|\buidai|आधार)\s*(?:(?:no\.?|number|id)\s*)?[:#-]?\s*$", re.I)
_RE_POS_IP_CTX = re.compile(r"\b(?:ip|ip\s+address|ipv4|ipv6)\s*[:#-]?\s*$", re.I)
_RE_FREQ_DOSE = re.compile(r"\b[0-3]-[0-3]-[0-3](?:-[0-3])?\b")

# FR-D1 patterns
_RE_AADHAAR_VID = re.compile(r"(?<![0-9A-Za-z])(\d{4}[- .]?\d{4}[- .]?\d{4}[- .]?\d{4}|\d{16})(?![0-9A-Za-z])")
_RE_AADHAAR = re.compile(r"(?<![0-9A-Za-z])([2-9]\d{3}[- .]?\d{4}[- .]?\d{4})(?![0-9A-Za-z])")
_RE_AADHAAR_MASKED = re.compile(r"(?<![0-9A-Za-z])([Xx*]{4}[- .]?[Xx*]{4}[- .]?\d{4})(?![0-9A-Za-z])")
_RE_ABHA_NUM = re.compile(r"(?<![0-9A-Za-z])(91[- .]?\d{4}[- .]?\d{4}[- .]?\d{4})(?![0-9A-Za-z])")
_RE_ABHA_ADDR = re.compile(r"(?<![0-9A-Za-z])([A-Za-z0-9._-]{3,}@(?:abdm|sbx))(?![0-9A-Za-z])")
_RE_PAN = re.compile(r"\b([A-Z]{5}\d{4}[A-Z])\b")
_RE_EMAIL = re.compile(r"\b([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")
_RE_PHONE = re.compile(
    r"(?<![0-9A-Za-z])((?:\+?91[- ]?|0091[- ]?|0)?(?:\(\d{3}\)[- ]?\d{3}[- ]?\d{4}|[6-9]\d{4}[- ]?\d{5}|[6-9]\d{2}[- ]?\d{3}[- ]?\d{4}|[6-9]\d[- ]?\d{4}[- ]?\d{4}))(?![0-9A-Za-z])"
)
_RE_PHONE_MASKED = re.compile(
    r"(?<![0-9A-Za-z*])((?:\+91[ -]?|0)?[0-9Xx*](?:[ -]?[0-9Xx*]){9})(?![0-9A-Za-z*])"
)
_RE_MRN_LABELED = re.compile(
    r"\b(?:UHID|MRN|IPD|OPD|CR\s*No\.?|ADM|Claim|Policy|PMJAY|AB-PMJAY|Patient\s*ID|Ref\.?\s*No\.?)\s*(?:No\.?|ID|#|:|-)?\s*[:#-]?\s*"
    r"([A-Z]{1,6}(?:[-/ ]\d{1,4})?[-/ ]\d{4,10}|\d{5,12})\b",
    re.I,
)
_RE_MRN_STRUCTURAL = re.compile(r"\b([A-Z]{2,6}[-/]\d{1,4}[-/]\d{4,10})\b")
_RE_MRN_PREFIXED = re.compile(r"\b((?:UHID|MRN|IPD|OPD|ADM|CLM|POL|PMJAY)[-/ ]?\d{2,4}[-/]\d{4,8})\b", re.I)
_RE_MRN_COMPACT = re.compile(r"\b([A-Z]{2,6}(?:[-/][A-Z]{2,6})?[-/]?\d{4,10}(?:[-/]\d{5,10})?)\b")
_RE_CLINICAL_CODE_PREFIX = re.compile(r"^(?:PKG|CGHS|ECHS|ICD|LOINC|LAB|CBC|WBC)[-/]?\d", re.I)

# FR-D9 patterns
_RE_EPIC = re.compile(r"\b([A-Z]{3}\d{7})\b")
_RE_DL = re.compile(r"\b([A-Z]{2}[- ]?\d{2}[- ]?(?:19|20)\d{2}\d{7})\b")
_RE_PASSPORT = re.compile(r"\b(?:passport|pp)\s*(?:no\.?|#|:)?\s*([A-PR-WYa-pr-wy][1-9]\d{6})\b", re.I)
_RE_GSTIN = re.compile(r"\b(\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z])\b")
_RE_UPI = re.compile(
    r"\b([A-Za-z0-9._-]{2,}@(?:ok[a-z]+|ybl|paytm|ibl|axl|sbi|icici|hdfcbank|upi|axisbank|kotak))\b",
    re.I,
)
_RE_URL = re.compile(r"(https?://[^\s<>\"',;]+)")
_RE_IP = re.compile(r"\b((?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?))\b")
_RE_IPV6 = re.compile(r"(?<![0-9A-Za-z:])([0-9A-Fa-f:]*:[0-9A-Fa-f:.]+)(?![0-9A-Za-z:])")
_RE_DEVICE_SERIAL = re.compile(
    r"\b(?:device\s*serial|serial\s*(?:no\.?|number|#)?|IMEI|S/N)\s*[:#-]?\s*([A-Za-z0-9][A-Za-z0-9-]{4,20})\b",
    re.I,
)
_RE_VEHICLE = re.compile(r"\b([A-Z]{2}\d{2}[A-Z]{1,3}\d{4})\b")
_RE_DOCTOR_REG = re.compile(
    r"\b(?:Dr\.?\s*Reg\.?|Reg\.?|Registration)\s*(?:No\.?|#|:)?\s*(?:[A-Z]{2,6}[-/ ]?)?(\d{4,8})\b",
    re.I,
)


def _has_neg_context(text: str, start: int, end: int, window: int = 24) -> bool:
    left = text[max(0, start - window) : start]
    right = text[end : min(len(text), end + window)]
    return bool(_RE_CLIN_NUM_CTX.search(left) or _RE_CLIN_NUM_CTX.search(right))


def _emit(
    spans: list[Span],
    rec: Record,
    field_idx: int,
    start: int,
    end: int,
    typ: str,
    score: float,
    role: str = "patient",
) -> None:
    spans.append(
        Span(
            key=rec.key,
            field_idx=field_idx,
            start=start,
            end=end,
            typ=typ,
            role=role,  # type: ignore[arg-type]
            score=score,
            src="ids",
        )
    )


def detect_ids(rec: Record, ctx: Context) -> list[Span]:
    """Detect typed identifiers (FR-D1) and extended shapes (FR-D9)."""
    spans: list[Span] = []
    detect_cfg = ctx.policy.get("detect", {})
    enable_plates = bool(detect_cfg.get("vehicle_plates", True))
    doc_reg_digits = bool(detect_cfg.get("doctor_reg_remove_digits", True))

    for fld in rec.fields:
        if fld.vtype in {"path", "header", "key"}:
            continue
        raw = fld.text
        norm = raw.translate(_DEVA_DIGITS)
        coltype = (rec.meta.get("coltypes_by_idx", {}).get(fld.idx)
                   or ctx.coltypes.get(fld.path) or rec.meta.get("coltypes", {}).get(fld.path))

        # Column-typed fast path for structured ID columns (e.g. uhid=412345)
        if ((coltype == "id" or patient_id_header(fld.path) or fld.idx in rec.meta.get("patient_id_fields", ()))
            and (parse_local_id(raw) or (
                coltype == "id" and re.fullmatch(r"[A-Za-z0-9/-]{4,24}", norm.strip())
                and any(c.isdigit() for c in norm)
            ))):
            _emit(spans, rec, fld.idx, 0, len(raw), "MRN", 0.95)

        for m in LABELED_LOCAL.finditer(norm):
            _emit(spans, rec, fld.idx, *m.span("value"), "MRN", 0.95)

        # Track 16-digit VIDs so 12-digit Aadhaar regex doesn't double-match inside them
        vid_ranges: list[tuple[int, int]] = []
        for m in _RE_AADHAAR_VID.finditer(norm):
            s, e = m.span(1)
            if not _has_neg_context(norm, s, e):
                vid_ranges.append((s, e))
                _emit(spans, rec, fld.idx, s, e, "AADHAAR_VID", 0.92)

        for m in _RE_ABHA_NUM.finditer(norm):
            s, e = m.span(1)
            _emit(spans, rec, fld.idx, s, e, "ABHA_NUMBER", 0.97)

        for m in _RE_ABHA_ADDR.finditer(norm):
            s, e = m.span(1)
            _emit(spans, rec, fld.idx, s, e, "ABHA_ADDRESS", 0.95)

        for m in _RE_AADHAAR_MASKED.finditer(norm):
            s, e = m.span(1)
            _emit(spans, rec, fld.idx, s, e, "AADHAAR_MASKED", 0.90)

        for m in _RE_AADHAAR.finditer(norm):
            s, e = m.span(1)
            if any(vs <= s and e <= ve for vs, ve in vid_ranges):
                continue
            left = norm[max(0, s - 32):s]
            if coltype != "aadhaar" and not _RE_POS_AADHAAR_CTX.search(left) and _has_neg_context(norm, s, e):
                continue
            digits = re.sub(r"\D", "", m.group(1))
            score = 0.98 if _verhoeff_valid(digits) else 0.85
            _emit(spans, rec, fld.idx, s, e, "AADHAAR", score)

        for m in _RE_GSTIN.finditer(norm):
            s, e = m.span(1)
            _emit(spans, rec, fld.idx, s, e, "GSTIN", 0.95)

        for m in _RE_PAN.finditer(norm):
            s, e = m.span(1)
            _emit(spans, rec, fld.idx, s, e, "PAN", 0.96)

        for m in _RE_UPI.finditer(norm):
            s, e = m.span(1)
            _emit(spans, rec, fld.idx, s, e, "UPI", 0.93)

        for m in _RE_EMAIL.finditer(norm):
            s, e = m.span(1)
            if not norm[s:e].lower().endswith(("@abdm", "@sbx")):
                _emit(spans, rec, fld.idx, s, e, "EMAIL", 0.95)

        for m in _RE_PHONE.finditer(norm):
            s, e = m.span(1)
            if any(vs <= s and e <= ve for vs, ve in vid_ranges):
                continue
            left_phone = norm[max(0, s - 16) : s]
            has_pos_phone = coltype == "phone" or bool(_RE_POS_PHONE_CTX.search(left_phone))
            if not has_pos_phone and _has_neg_context(norm, s, e, window=14):
                continue
            _emit(spans, rec, fld.idx, s, e, "PHONE", 0.94)

        for m in _RE_PHONE_MASKED.finditer(norm):
            s, e = m.span(1)
            compact = re.sub(r"[ -]", "", m.group(1))
            if compact.startswith("+91"):
                compact = compact[3:]
            elif len(compact) == 11 and compact.startswith("0"):
                compact = compact[1:]
            if not any(ch in "Xx*" for ch in compact) or sum(ch.isdigit() for ch in compact) < 2:
                continue
            left_phone = norm[max(0, s - 16):s]
            has_pos_phone = coltype == "phone" or bool(_RE_POS_PHONE_CTX.search(left_phone))
            if not has_pos_phone and (compact[0] not in "6789" or _has_neg_context(norm, s, e, window=14)):
                continue
            _emit(spans, rec, fld.idx, s, e, "PHONE", 0.90)

        for m in _RE_MRN_LABELED.finditer(norm):
            s, e = m.span(1)
            if _RE_FREQ_DOSE.fullmatch(m.group(1)):
                continue
            _emit(spans, rec, fld.idx, s, e, "MRN", 0.95)

        for m in _RE_MRN_STRUCTURAL.finditer(norm):
            s, e = m.span(1)
            if not _has_neg_context(norm, s, e):
                _emit(spans, rec, fld.idx, s, e, "MRN", 0.92)

        for m in _RE_MRN_COMPACT.finditer(norm):
            s, e = m.span(1)
            if not _RE_CLINICAL_CODE_PREFIX.match(m.group(1)) and not _has_neg_context(norm, s, e):
                _emit(spans, rec, fld.idx, s, e, "MRN", 0.85)

        for m in _RE_MRN_PREFIXED.finditer(norm):
            s, e = m.span(1)
            _emit(spans, rec, fld.idx, s, e, "MRN", 0.93)

        for m in _RE_EPIC.finditer(norm):
            s, e = m.span(1)
            if not _has_neg_context(norm, s, e):
                _emit(spans, rec, fld.idx, s, e, "EPIC", 0.88)

        for m in _RE_DL.finditer(norm):
            s, e = m.span(1)
            _emit(spans, rec, fld.idx, s, e, "DRIVING_LICENCE", 0.90)

        for m in _RE_PASSPORT.finditer(norm):
            s, e = m.span(1)
            _emit(spans, rec, fld.idx, s, e, "PASSPORT", 0.88)

        for m in _RE_URL.finditer(norm):
            s, e = m.span(1)
            _emit(spans, rec, fld.idx, s, e, "URL", 0.90, role="none")

        for m in _RE_IP.finditer(norm):
            s, e = m.span(1)
            if _RE_POS_IP_CTX.search(norm[max(0, s - 24):s]) or not _has_neg_context(norm, s, e):
                _emit(spans, rec, fld.idx, s, e, "IP_ADDRESS", 0.85, role="none")

        for m in _RE_IPV6.finditer(norm):
            s, e = m.span(1)
            address = m.group(1).rstrip(".")
            e = s + len(address)
            try:
                ipaddress.IPv6Address(address)
            except ipaddress.AddressValueError:
                continue
            if _RE_POS_IP_CTX.search(norm[max(0, s - 24):s]) or not _has_neg_context(norm, s, e):
                _emit(spans, rec, fld.idx, s, e, "IP_ADDRESS", 0.88, role="none")

        for m in _RE_DEVICE_SERIAL.finditer(norm):
            s, e = m.span(1)
            if not _has_neg_context(norm, s, e):
                _emit(spans, rec, fld.idx, s, e, "DEVICE_SERIAL", 0.88, role="none")

        if enable_plates:
            for m in _RE_VEHICLE.finditer(norm):
                s, e = m.span(1)
                if not _has_neg_context(norm, s, e):
                    _emit(spans, rec, fld.idx, s, e, "VEHICLE_PLATE", 0.86)

        if doc_reg_digits:
            for m in _RE_DOCTOR_REG.finditer(norm):
                s, e = m.span(1)
                _emit(spans, rec, fld.idx, s, e, "DOCTOR_REG", 0.90, role="doctor")

    return spans
