"""B1: Semantic typing of columns and keys (PRD §6.4)."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from kavachmed.contracts import Record
from kavachmed.local_ids import parse_local_id

_DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

_HEADER_PRIORS: dict[str, set[str]] = {
    "name": {"name", "patient_name", "pt_name", "full_name", "naam", "नाम", "मरीज का नाम", "patient", "nm", "pt", "rogi", "insured", "beneficiary"},
    "relative": {"relative", "father_name", "husband_name", "spouse", "guardian", "care_of", "पिता", "पति"},
    "phone": {"phone", "mobile", "mob", "contact", "tel", "telephone", "मोबाइल", "फोन", "संपर्क", "dooraabhash", "callback_no"},
    "aadhaar": {"aadhaar", "aadhar", "uid", "uidai", "आधार", "aadhaar_sankhya", "aadhr_num"},
    "abha": {"abha", "abha_no", "abha_number", "abha_id", "abha_address", "health_id"},
    "pan": {"pan", "pan_no", "pan_number", "payer_pan"},
    "email": {"email", "e-mail", "mail"},
    "id": {"uhid", "mrn", "ipd", "opd", "cr_no", "reg_no", "patient_id", "claim_id", "policy_no", "pmjay", "hosp_id", "hosp_no", "ref", "vzc_id"},
    "pin": {"pin", "pincode", "pin_code", "postal", "zip", "पिन", "postcode"},
    "date": {"date", "dob", "birth_date", "janm", "janm_tithi", "janm_din", "visit_date", "adm_date", "tithi", "born", "registered", "collected", "reported", "admitted", "discharged", "जन्म", "तारीख", "दिनांक"},
    "age": {"age", "umar", "umra", "aayu", "yrs", "उम्र", "आयु"},
    "sex": {"sex", "gender", "ling", "लिंग"},
    "address": {"address", "addr", "pata", "gaon", "village", "jila", "zila", "district", "tehsil", "taluka", "mohalla", "पता", "गांव", "जिला"},
    "free_text": {"test", "test_name", "t", "dx", "diagnosis", "complaint", "advice", "procedure", "package", "unit", "u", "v", "value", "result"},
}

_LAB_TEST_WORDS = frozenset({
    "haemoglobin", "hemoglobin", "blood", "sugar", "creatinine", "platelet", "platelets",
    "bilirubin", "serum", "hba1c", "cholesterol", "triglycerides", "potassium", "sodium",
    "urea", "uric", "thyroid", "tsh", "cbc", "rbs", "fbs", "ppbs", "esr", "crp", "alp",
    "sgot", "sgpt", "count", "random", "fasting", "postprandial", "urine", "wbc", "rbc",
    "ferritin", "troponin", "albumin", "calcium", "lipid", "liver", "kidney", "renal",
    "routine", "profile", "panel", "test", "glucose", "electrolytes",
})

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
    if _RE_ID.match(val) or parse_local_id(val) is not None:
        return "id"
    if _RE_ADDR_CUE.search(val) or ("," in val and len(val) > 15 and any(ch.isdigit() for ch in val)):
        return "address"
    if _RE_NAME_LIKE.match(val) and not any(ch.isdigit() for ch in val):
        if any(w.casefold() in _LAB_TEST_WORDS for w in val.split()):
            return "free_text"
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


def _infer_type(header: str, votes: Counter[str]) -> str:
    prior = _header_prior(header)
    total = sum(votes.values())
    if not total:
        return prior or "free_text"
    ranked = votes.most_common(2)
    top_type, top_count = ranked[0]
    if len(ranked) == 2 and top_count == ranked[1][1]:
        if {t for t, _ in ranked}.issubset({"aadhaar", "abha", "id"}):
            return "id"
        return prior if prior and votes[prior] else "free_text"
    if prior == "id" and top_type in {"pin", "id"}:
        return "id"
    if top_type != "free_text" and top_count / total > 0.5:
        return top_type
    if prior and (votes[prior] / total >= 0.3 or (top_type == "free_text" and total <= 2)):
        return prior
    return "free_text"


def type_columns(records_or_parsed: Any) -> dict[str, str]:
    """Pool only direct JSON row scalars across rows of the same collection."""
    records: list[Record] = (
        records_or_parsed.records if hasattr(records_or_parsed, "records") else list(records_or_parsed)
    )
    json_rows = getattr(records_or_parsed, "fmt", None) == "json"
    votes: dict[tuple[str | None, str], Counter[str]] = {}
    locations: list[tuple[Record, int, str, tuple[str | None, str]]] = []
    for rec in records:
        loc = rec.key[1]
        parent, _, index = loc.rpartition("/")
        direct = rec.meta.get("direct_fields", ()) if json_rows and index.isascii() and index.isdigit() else ()
        for fld in rec.fields:
            if fld.vtype in {"path", "header", "key"}:
                continue
            key = (None, fld.path)
            if fld.idx in direct and fld.path.startswith(loc + "/"):
                key = (parent, fld.path[len(loc) + 1:])
            counts = votes.setdefault(key, Counter())
            vote = _vote_value(fld.text)
            if vote:
                counts[vote] += 1
            locations.append((rec, fld.idx, fld.path, key))

    inferred = {key: _infer_type(key[1], counts) for key, counts in votes.items()}
    coltypes: dict[str, str] = {}
    for rec, idx, path, key in locations:
        value_type = inferred[key]
        rec.meta.setdefault("coltypes_by_idx", {})[idx] = value_type
        if key[0] is not None or path not in coltypes:
            coltypes[path] = value_type
    for rec in records:
        rec.meta["coltypes"] = coltypes
    return coltypes
