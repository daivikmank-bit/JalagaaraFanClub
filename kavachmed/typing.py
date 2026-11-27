"""B1: Semantic typing of columns and keys (PRD §6.4)."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from kavachmed.contracts import Record

_DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

_HEADER_PRIORS: dict[str, set[str]] = {
    "name": {"name", "patient_name", "pt_name", "full_name", "naam", "नाम", "मरीज का नाम", "patient"},
    "relative": {"relative", "father_name", "husband_name", "spouse", "guardian", "care_of", "पिता", "पति"},
    "phone": {"phone", "mobile", "mob", "contact", "tel", "telephone", "मोबाइल", "फोन", "संपर्क"},
    "aadhaar": {"aadhaar", "aadhar", "uid", "uidai", "आधार"},
    "abha": {"abha", "abha_no", "abha_number", "abha_id", "abha_address", "health_id"},
    "pan": {"pan", "pan_no", "pan_number"},
    "email": {"email", "e-mail", "mail"},
    "id": {"uhid", "mrn", "ipd", "opd", "cr_no", "reg_no", "patient_id", "claim_id", "policy_no", "pmjay"},
    "pin": {"pin", "pincode", "pin_code", "postal", "zip", "पिन"},
    "date": {"date", "dob", "birth_date", "janm", "janm_tithi", "visit_date", "adm_date", "जन्म", "तारीख", "दिनांक"},
    "age": {"age", "umar", "umra", "उम्र", "आयु"},
    "sex": {"sex", "gender", "ling", "लिंग"},
    "address": {"address", "addr", "pata", "gaon", "village", "jila", "district", "tehsil", "mohalla", "पता", "गांव", "जिला"},
}

_RE_PHONE = re.compile(r"^(?:\+91[- ]?|0)?[6-9]\d{4}[- ]?\d{5}$")
_RE_AADHAAR = re.compile(r"^[2-9]\d{3}[- ]?\d{4}[- ]?\d{4}$")
_RE_ABHA = re.compile(r"^91[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}$")
_RE_ABHA_ADDR = re.compile(r"^[A-Za-z0-9._-]{3,}@(?:abdm|sbx)$")
_RE_PAN = re.compile(r"^[A-Z]{5}\d{4}[A-Z]$")
_RE_EMAIL = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
_RE_PIN = re.compile(r"^[1-8]\d{5}$")
_RE_DATE = re.compile(
    r"^(?:\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|"
    r"\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*(?:\s+\d{2,4})?)$",
    re.I,
)
_RE_AGE = re.compile(r"^(?:\d{1,3}|(?:\d{1,3}\s*(?:y|yr|yrs|years|m|months|वर्ष|साल)))$", re.I)
_RE_SEX = re.compile(r"^(?:m|f|o|male|female|other|transgender|पुरुष|महिला|स्त्री)$", re.I)
_RE_ID = re.compile(r"^[A-Z]{2,6}[-/ ]?(?:\d{2,4}[-/])?\d{4,10}$")
_RE_ADDR_CUE = re.compile(
    r"\b(?:h\.?\s*no|ward|mohalla|gali|colony|nagar|near|village|gram|tehsil|taluka|dist|district|pin)\b|"
    r"(?:वार्ड|मोहल्ला|गली|नगर|गांव|ग्राम|तहसील|जिला|पिन)",
    re.I,
)
_RE_NAME_LIKE = re.compile(r"^(?:[A-Z][a-z]{1,25}(?:\s+[A-Z][a-z]{1,25}){0,3}|[\u0900-\u097F]{2,}(?:\s+[\u0900-\u097F]{2,}){0,3})$")


def _norm_digits(val: str) -> str:
    return val.translate(_DEVA_DIGITS).strip()


def _vote_value(raw: str) -> str | None:
    val = _norm_digits(raw)
    if not val or val.lower() in {"na", "n/a", "null", "none", "-", "--"}:
        return None
    if _RE_ABHA.match(val) or _RE_ABHA_ADDR.match(val):
        return "abha"
    if _RE_AADHAAR.match(val):
        return "aadhaar"
    if _RE_PHONE.match(val):
        return "phone"
    if _RE_PAN.match(val):
        return "pan"
    if _RE_EMAIL.match(val):
        return "email"
    if _RE_SEX.match(val):
        return "sex"
    if _RE_PIN.match(val):
        return "pin"
    if _RE_DATE.match(val):
        return "date"
    if _RE_AGE.match(val):
        digits = re.match(r"^\d+", val)
        if digits and 0 <= int(digits.group()) <= 120:
            return "age"
    if _RE_ID.match(val):
        return "id"
    if _RE_ADDR_CUE.search(val) or ("," in val and len(val) > 15 and any(ch.isdigit() for ch in val)):
        return "address"
    if _RE_NAME_LIKE.match(val) and not any(ch.isdigit() for ch in val):
        return "name"
    return "free_text"


def _header_prior(col_name: str) -> str | None:
    norm = re.sub(r"[\s\-_]+", "_", col_name.strip().lower())
    plain = col_name.strip().lower()
    for typ, keys in _HEADER_PRIORS.items():
        if norm in keys or plain in keys:
            return typ
    for typ, keys in _HEADER_PRIORS.items():
        for k in keys:
            if len(k) >= 3 and (k in norm or k in plain):
                return typ
    return None


def type_columns(records_or_parsed: Any) -> dict[str, str]:
    """Infer semantic column/key types by voting over cell values with header priors."""
    records: list[Record] = (
        records_or_parsed.records if hasattr(records_or_parsed, "records") else list(records_or_parsed)
    )
    by_col: dict[str, list[str]] = {}
    for rec in records:
        for fld in rec.fields:
            if fld.vtype in {"path", "header", "key"}:
                continue
            by_col.setdefault(fld.path, []).append(fld.text)

    coltypes: dict[str, str] = {}
    for col in sorted(by_col):
        votes: Counter[str] = Counter()
        for val in by_col[col]:
            v = _vote_value(val)
            if v is not None:
                votes[v] += 1

        prior = _header_prior(col)
        total = sum(votes.values())
        if total == 0:
            coltypes[col] = prior or "free_text"
            continue

        top_type, top_count = votes.most_common(1)[0]
        # Ambiguous tie (<= 50% or tied top-2) falls back to free_text unless header prior breaks it
        second_count = votes.most_common(2)[1][1] if len(votes) >= 2 else 0
        if top_count == second_count:
            top2 = {t for t, _ in votes.most_common(2)}
            if top2.issubset({"aadhaar", "abha", "id"}):
                coltypes[col] = "id"
            elif prior and votes.get(prior, 0) > 0:
                coltypes[col] = prior
            else:
                coltypes[col] = "free_text"
            continue

        # Header prior "id" overrides 6-digit "pin" votes (e.g. uhid=412345)
        if prior == "id" and top_type in {"pin", "id"}:
            coltypes[col] = "id"
        elif top_type != "free_text" and top_count / total > 0.5:
            coltypes[col] = top_type
        elif prior and votes.get(prior, 0) / total >= 0.3:
            coltypes[col] = prior
        elif prior and top_type == "free_text" and total <= 2:
            coltypes[col] = prior
        else:
            coltypes[col] = "free_text"

    for rec in records:
        rec.meta["coltypes"] = coltypes
    return coltypes
