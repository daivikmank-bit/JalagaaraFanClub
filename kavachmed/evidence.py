"""Typed linkage evidence extraction from a Record and detector Spans (Plan C V3).

Implements C5 / FR-L1: turns one parsed ``Record`` and its ``Span`` objects
into a normalized ``Evidence`` object without touching the filesystem or
importing dev-only tooling.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from kavachmed.indic import NameToken, normalize_name

try:
    from kavachmed.contracts import Field, RecKey, Record, Span
except ImportError:  # pragma: no cover - fallback before Plan A scaffolds contracts.py
    RecKey = tuple[str, str]

    @dataclass(frozen=True)
    class Field:
        idx: int
        path: str
        text: str
        vtype: str = "str"

    @dataclass
    class Record:
        key: RecKey
        fields: list[Field]
        meta: dict[str, Any] = field(default_factory=dict)

    @dataclass(frozen=True)
    class Span:
        key: RecKey
        field_idx: int
        start: int
        end: int
        typ: str
        role: str
        score: float
        src: str
        protected: bool = False
        exempt_kind: str | None = None


@dataclass
class Evidence:
    """Normalized record-scoped linkage evidence (FR-L1)."""

    key: RecKey
    names: list[NameToken] = field(default_factory=list)
    kin: list[NameToken] = field(default_factory=list)
    kin_type: str | None = None
    sex: str | None = None
    birth_years: tuple[int, int] | None = None
    ids: dict[str, list[str]] = field(default_factory=dict)
    phones: list[str] = field(default_factory=list)
    phone_shared: bool = False
    geo: dict[str, Any] = field(default_factory=dict)
    hospital: str | None = None
    adm_key: str | None = None
    dates: list[str] = field(default_factory=list)


class DateProvider(Protocol):
    """Protocol for D's date extraction results."""

    def find(self, rec: Record) -> list[dict[str, Any]]: ...


LgdLookup = Callable[[str, str, dict[str, str]], str | None]


# ---------------------------------------------------------------------------
# Constants & normalization tables
# ---------------------------------------------------------------------------

_KIN_CUE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?:s/o|son\s+of|पुत्र|सुत)\b", re.IGNORECASE), "son_of"),
    (re.compile(r"\b(?:d/o|daughter\s+of|पुत्री|सुता)\b", re.IGNORECASE), "daughter_of"),
    (re.compile(r"\b(?:w/o|wife\s+of|पत्नी)\b", re.IGNORECASE), "wife_of"),
    (re.compile(r"\b(?:c/o|care\s+of)\b", re.IGNORECASE), "care_of"),
)

_KIN_FIELD_MAP: dict[str, str] = {
    "s/o": "son_of",
    "so": "son_of",
    "son_of": "son_of",
    "son of": "son_of",
    "d/o": "daughter_of",
    "do": "daughter_of",
    "daughter_of": "daughter_of",
    "daughter of": "daughter_of",
    "w/o": "wife_of",
    "wo": "wife_of",
    "wife_of": "wife_of",
    "wife of": "wife_of",
    "c/o": "care_of",
    "co": "care_of",
    "care_of": "care_of",
    "care of": "care_of",
}

_KIN_TO_SEX: dict[str, str] = {
    "son_of": "M",
    "daughter_of": "F",
    "wife_of": "F",
}

_HONORIFIC_TO_SEX: dict[str, str] = {
    "shri": "M",
    "mr": "M",
    "bhai": "M",
    "sahab": "M",
    "saheb": "M",
    "sardar": "M",
    "smt": "F",
    "mrs": "F",
    "ms": "F",
    "miss": "F",
    "kumari": "F",
    "devi": "F",
    "begum": "F",
    "bibi": "F",
    "didi": "F",
    "mata": "F",
    "bai": "F",
}

_HINDI_GENDER_WORDS: dict[str, str] = {
    "पुरुष": "M",
    "महिला": "F",
    "स्त्री": "F",
    "male": "M",
    "man": "M",
    "boy": "M",
    "female": "F",
    "woman": "F",
    "girl": "F",
    "m": "M",
    "f": "F",
}

_SEX_FIELD_PATHS = ("sex", "gender", "patient_sex", "patient_gender", "लिंग")
_REL_FIELD_PATHS = ("relation", "rel_type", "kin_type", "care_of_type")

_EXCLUDED_NAME_ROLES = {"doctor", "sender", "third_party", "none"}

# Structured ID patterns
_SLASH_ID_RE = re.compile(r"^([A-Za-z]{2,8})[/-](\d{2,4})[/-]([A-Za-z0-9]{3,16})$")
_PREFIXED_ID_RE = re.compile(r"^([A-Za-z]{2,8})(\d{2,4})?[-/]?(\d{4,12})$")
_GENERIC_ID_PREFIXES = frozenset({"uhid", "mrn", "id", "regno", "reg", "crno", "cr", "no", "pid"})
_PAN_RE = re.compile(r"^[A-Z]{5}\d{4}[A-Z]$")
_ISO_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_DMY_DATE_RE = re.compile(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{4})$")
_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
_AGE_NUM_RE = re.compile(r"(\d{1,3})")


def _fold_unicode_digits(text: str) -> str:
    """Convert any Unicode decimal digits (e.g. Devanagari) to ASCII digits."""
    out: list[str] = []
    for ch in text:
        if ch.isdigit():
            try:
                out.append(str(unicodedata.digit(ch)))
                continue
            except ValueError:
                pass
        out.append(ch)
    return "".join(out)


def _dedup_extend(target: list[Any], items: list[Any]) -> None:
    seen = set(target)
    for item in items:
        if item not in seen:
            seen.add(item)
            target.append(item)


def _normalize_sex_value(raw: str) -> str | None:
    cleaned = _fold_unicode_digits(unicodedata.normalize("NFKC", raw)).casefold().strip()
    cleaned = re.sub(r"[^a-z\u0900-\u097f]", "", cleaned)
    return _HINDI_GENDER_WORDS.get(cleaned)


def _normalize_kin_cue(raw: str) -> str | None:
    cleaned = unicodedata.normalize("NFKC", raw).casefold().strip()
    if cleaned in _KIN_FIELD_MAP:
        return _KIN_FIELD_MAP[cleaned]
    for pat, label in _KIN_CUE_PATTERNS:
        if pat.search(raw):
            return label
    return None


def _extract_local_kin_cue(field_text: str, span_start: int) -> str | None:
    """Scan only the local prefix immediately preceding a relative span."""
    prefix = field_text[max(0, span_start - 32) : span_start]
    matches: list[tuple[int, str]] = []
    for pat, label in _KIN_CUE_PATTERNS:
        for m in pat.finditer(prefix):
            matches.append((m.end(), label))
    if not matches:
        return None
    matches.sort(key=lambda x: x[0])
    return matches[-1][1]


def _normalize_phone(raw: str) -> str | None:
    folded = _fold_unicode_digits(unicodedata.normalize("NFKC", raw)).strip()
    if not folded:
        return None
    # Reject strings containing letters (e.g. alphanumeric IDs or doses).
    if re.search(r"[A-Za-z]", folded):
        return None
    digits = re.sub(r"[\s\-\(\)\.+]", "", folded)
    if not digits.isdigit():
        return None
    if len(digits) == 10:
        return digits
    if len(digits) == 11 and digits.startswith("0"):
        return digits[1:]
    if len(digits) == 12 and digits.startswith("91"):
        return digits[2:]
    if len(digits) == 13 and digits.startswith("091"):
        return digits[3:]
    return None


def _normalize_typed_id(
    typ: str,
    raw: str,
    hospital_hint: str | None,
) -> tuple[str | None, str | None]:
    """Return ``(namespace, normalized_core)`` or ``(None, None)`` if unscoped/invalid."""
    folded = _fold_unicode_digits(unicodedata.normalize("NFKC", raw)).strip()
    if not folded:
        return None, None
    t = typ.upper()

    if t == "AADHAAR":
        digits = re.sub(r"[\s\-]", "", folded)
        if digits.isdigit() and len(digits) == 12:
            return "aadhaar", digits
        return None, None

    if t == "AADHAAR_MASKED":
        digits = re.sub(r"[^0-9]", "", folded)
        if len(digits) == 4:
            return "aadhaar_last4", digits
        return None, None

    if t == "ABHA_NUMBER":
        digits = re.sub(r"[\s\-]", "", folded)
        if digits.isdigit() and len(digits) == 14:
            return "abha", digits
        return None, None

    if t == "PAN":
        alnum = re.sub(r"\s+", "", folded).upper()
        if _PAN_RE.match(alnum):
            return "pan", alnum
        return None, None

    if t in {"MRN", "UHID"}:
        compact = re.sub(r"\s+", "", folded)
        m = _SLASH_ID_RE.match(compact)
        if m:
            issuer, _yr, core = m.group(1).casefold(), m.group(2), m.group(3).casefold()
            return f"uhid:{issuer}", core
        # Issuer-prefixed local IDs (JJH-0966021, DHH2026-071345, UHID-009559): the letter
        # prefix names the issuing system. Generic prefixes (UHID/MRN/ID) only bridge with a
        # long core, so short local serials from different hospitals cannot collide.
        m2 = _PREFIXED_ID_RE.match(compact)
        if m2:
            prefix, core = m2.group(1).casefold(), m2.group(3).casefold()
            if prefix not in _GENERIC_ID_PREFIXES:
                return f"uhid:{prefix}", core
            if hospital_hint:
                issuer = re.sub(r"[^a-z0-9]", "", hospital_hint.casefold())
                if issuer:
                    return f"uhid:{issuer}", core
            if len(core) >= 6:
                return f"uhid:{prefix}", core
        # Local unscoped MRN requires a confirmed hospital issuer to bridge.
        if hospital_hint:
            issuer = re.sub(r"[^a-z0-9]", "", hospital_hint.casefold())
            core = re.sub(r"[^a-z0-9]", "", compact.casefold())
            if issuer and core:
                return f"uhid:{issuer}", core
        return None, None

    if t == "INSURANCE_ID":
        compact = re.sub(r"[\s\-]+", "", folded).upper()
        if len(compact) < 4:
            return None, None
        if hospital_hint:
            issuer = re.sub(r"[^a-z0-9]", "", hospital_hint.casefold())
            if issuer:
                return f"ins:{issuer}", compact
        return "ins", compact

    if t in {"ADM_KEY", "ADMISSION_ID", "IPD_NO"}:
        compact = re.sub(r"\s+", "", folded).casefold()
        if not compact:
            return None, None
        if hospital_hint:
            issuer = re.sub(r"[^a-z0-9]", "", hospital_hint.casefold())
            if issuer:
                return None, f"{issuer}:{compact}"
        m = _SLASH_ID_RE.match(compact)
        if m:
            return None, f"{m.group(1)}:{m.group(3)}"
        return None, None

    return None, None


def _parse_year(date_str: str, date_order: str = "DMY") -> int | None:
    s = _fold_unicode_digits(unicodedata.normalize("NFKC", date_str)).strip()
    m_iso = _ISO_DATE_RE.match(s)
    if m_iso:
        return int(m_iso.group(1))
    m_dmy = _DMY_DATE_RE.match(s)
    if m_dmy:
        p1, p2, yr = int(m_dmy.group(1)), int(m_dmy.group(2)), int(m_dmy.group(3))
        if date_order == "AMBIGUOUS":
            # Ambiguous day/month order still has an unambiguous 4-digit year,
            # unless caller marks the entire date order unresolved.
            return None
        if p1 <= 31 and p2 <= 31:
            return yr
    m_yr = _YEAR_RE.search(s)
    if m_yr and date_order != "AMBIGUOUS":
        return int(m_yr.group(1))
    return None


def _intersect_interval(
    cur: tuple[int, int] | None,
    new: tuple[int, int],
) -> tuple[tuple[int, int] | None, bool]:
    """Return ``(intersection, conflict)``."""
    if cur is None:
        return new, False
    lo = max(cur[0], new[0])
    hi = min(cur[1], new[1])
    if lo > hi:
        return None, True
    return (lo, hi), False


def extract(
    rec: Record,
    spans: list[Span],
    *,
    date_provider: DateProvider | None = None,
    lgd_lookup: LgdLookup | None = None,
    diagnostics: dict[str, int] | None = None,
    filename_signals: dict[str, list[str]] | None = None,
) -> Evidence:
    """Extract normalized ``Evidence`` from ``rec`` and validated ``spans``.

    ``filename_signals`` (platform's ``rename.extract_filename_signals`` dict) is optional:
    a 10-digit mobile in the filename is added as phone evidence. Name/ID filename signals
    go to ``linkage.resolve`` (V7) so they can be attributed with corroboration.
    """
    fields_by_idx: dict[int, Field] = {f.idx: f for f in rec.fields}
    hospital = rec.meta.get("hospital")
    if isinstance(hospital, str):
        hospital = hospital.strip() or None
    else:
        hospital = None

    # Validate, scope, and deduplicate spans in deterministic (field_idx, start, end) order.
    valid_spans: list[Span] = []
    seen_span_keys: set[tuple[int, int, int, str, str]] = set()
    for sp in sorted(spans, key=lambda s: (s.field_idx, s.start, s.end, s.typ, s.role)):
        if sp.key != rec.key or sp.protected:
            continue
        fld = fields_by_idx.get(sp.field_idx)
        if fld is None:
            continue
        # Filename signals enter in V7; skip @filename / path fields in V3.
        if fld.vtype == "path" or fld.path == "@filename" or fld.path.startswith("@filename/"):
            continue
        if not (0 <= sp.start < sp.end <= len(fld.text)):
            continue
        dedup_key = (sp.field_idx, sp.start, sp.end, sp.typ, sp.role)
        if dedup_key in seen_span_keys:
            continue
        seen_span_keys.add(dedup_key)
        valid_spans.append(sp)

    names: list[NameToken] = []
    kin: list[NameToken] = []
    kin_cues: list[str] = []
    patient_honorific_sexes: list[str] = []
    ids: dict[str, list[str]] = {}
    phones: list[str] = []
    adm_candidates: list[str] = []
    dob_years: list[int] = []
    age_values: list[int] = []
    geo_raw: dict[str, str] = {}

    for sp in valid_spans:
        fld = fields_by_idx[sp.field_idx]
        surface = fld.text[sp.start : sp.end]
        typ = sp.typ.upper()
        role = sp.role

        if typ == "PATIENT_NAME":
            if role in _EXCLUDED_NAME_ROLES or role == "relative":
                continue
            tokens = list(normalize_name(surface))
            _dedup_extend(names, [t for t in tokens if t.kind != "honorific"])
            for tok in tokens:
                if tok.kind == "honorific" and tok.latin in _HONORIFIC_TO_SEX:
                    patient_honorific_sexes.append(_HONORIFIC_TO_SEX[tok.latin])

        elif typ == "RELATIVE_NAME":
            if role in _EXCLUDED_NAME_ROLES or role == "patient":
                continue
            tokens = list(normalize_name(surface))
            _dedup_extend(kin, [t for t in tokens if t.kind != "honorific"])
            cue = _extract_local_kin_cue(fld.text, sp.start)
            if cue:
                kin_cues.append(cue)

        elif typ == "PHONE":
            if role in _EXCLUDED_NAME_ROLES:
                continue
            norm_phone = _normalize_phone(surface)
            if norm_phone and norm_phone not in phones:
                phones.append(norm_phone)

        elif typ in {"ADM_KEY", "ADMISSION_ID", "IPD_NO"}:
            _, adm_norm = _normalize_typed_id(typ, surface, hospital)
            if adm_norm and adm_norm not in adm_candidates:
                adm_candidates.append(adm_norm)

        elif typ in {"AADHAAR", "AADHAAR_MASKED", "ABHA_NUMBER", "PAN", "MRN", "UHID", "INSURANCE_ID"}:
            ns, core = _normalize_typed_id(typ, surface, hospital)
            if ns and core:
                if (
                    diagnostics is not None
                    and hospital
                    and ns.startswith("uhid:")
                ):
                    hosp_norm = re.sub(r"[^a-z0-9]", "", hospital.casefold())
                    if hosp_norm and ns != f"uhid:{hosp_norm}":
                        diagnostics["issuer_mismatches"] = diagnostics.get("issuer_mismatches", 0) + 1
                bucket = ids.setdefault(ns, [])
                if core not in bucket:
                    bucket.append(core)

        elif typ == "DOB":
            yr = _parse_year(surface, str(rec.meta.get("date_order") or "DMY"))
            if yr is not None:
                dob_years.append(yr)

        elif typ == "AGE":
            m_age = _AGE_NUM_RE.search(_fold_unicode_digits(surface))
            if m_age:
                age_val = int(m_age.group(1))
                if 0 <= age_val <= 125:
                    age_values.append(age_val)

        elif typ in {"DISTRICT", "TEHSIL", "VILLAGE", "PIN", "STATE"}:
            cleaned_geo = unicodedata.normalize("NFKC", surface).strip()
            if cleaned_geo:
                geo_raw.setdefault(typ.lower(), cleaned_geo)

    # Explicit field-level sex and relation cues (never from relative fields).
    explicit_sexes: list[str] = []
    for fld in rec.fields:
        leaf = fld.path.rsplit("/", 1)[-1].casefold()
        if leaf in _SEX_FIELD_PATHS:
            sx = _normalize_sex_value(fld.text)
            if sx:
                explicit_sexes.append(sx)
        elif leaf in _REL_FIELD_PATHS:
            cue = _normalize_kin_cue(fld.text)
            if cue:
                kin_cues.append(cue)

    # Resolve kinship: conflicting cues -> None.
    distinct_kin = set(kin_cues)
    kin_type = next(iter(distinct_kin)) if len(distinct_kin) == 1 else None

    # Sex cascade: explicit field -> kin cue -> patient honorific -> Hindi gender word.
    sex: str | None = None
    if explicit_sexes:
        if len(set(explicit_sexes)) == 1:
            sex = explicit_sexes[0]
    elif kin_type and kin_type in _KIN_TO_SEX:
        sex = _KIN_TO_SEX[kin_type]
    elif patient_honorific_sexes:
        if len(set(patient_honorific_sexes)) == 1:
            sex = patient_honorific_sexes[0]
    else:
        gender_word_hits: set[str] = set()
        for fld in rec.fields:
            if fld.vtype == "path" or fld.path.startswith("@"):
                continue
            for tok in re.findall(r"[A-Za-z\u0900-\u097f]+", unicodedata.normalize("NFKC", fld.text)):
                mapped = _HINDI_GENDER_WORDS.get(tok.casefold())
                if mapped in {"M", "F"} and tok.casefold() not in {"m", "f"}:
                    gender_word_hits.add(mapped)
        if len(gender_word_hits) == 1:
            sex = next(iter(gender_word_hits))

    # Dates & birth-year interval.
    dates_out: list[str] = []
    encounter_years: list[int] = []
    date_entries: list[dict[str, Any]] = []
    if date_provider is not None:
        date_entries = list(date_provider.find(rec))
    elif isinstance(rec.meta.get("dates"), list):
        date_entries = list(rec.meta["dates"])

    for entry in date_entries:
        if not isinstance(entry, dict):
            continue
        raw_d = entry.get("text") or entry.get("iso") or entry.get("date")
        if not isinstance(raw_d, str) or not raw_d.strip():
            continue
        role_d = str(entry.get("role") or "clinical").casefold()
        order_d = str(entry.get("order") or rec.meta.get("date_order") or "DMY")
        if role_d == "dob":
            yr = _parse_year(raw_d, order_d)
            if yr is not None:
                dob_years.append(yr)
        else:
            if raw_d not in dates_out:
                dates_out.append(raw_d)
            if role_d in {"encounter", "admission", "visit", "discharge"} and order_d != "AMBIGUOUS":
                yr = _parse_year(raw_d, order_d)
                if yr is not None:
                    encounter_years.append(yr)

    birth_years: tuple[int, int] | None = None
    interval_conflict = False
    for yr in dob_years:
        birth_years, conflict = _intersect_interval(birth_years, (yr, yr))
        if conflict:
            interval_conflict = True
            break

    if not interval_conflict and age_values and encounter_years:
        for age_v in age_values:
            for enc_y in encounter_years:
                birth_years, conflict = _intersect_interval(
                    birth_years, (enc_y - age_v - 1, enc_y - age_v)
                )
                if conflict:
                    interval_conflict = True
                    break
            if interval_conflict:
                break

    if interval_conflict:
        birth_years = None

    # Geography via LGD lookup (with parent context).
    geo: dict[str, Any] = {}
    if "pin" in geo_raw:
        pin_digits = re.sub(r"\D", "", _fold_unicode_digits(geo_raw["pin"]))
        if len(pin_digits) == 6:
            geo["pin"] = pin_digits

    if lgd_lookup is not None:
        context = dict(geo_raw)
        for level in ("state", "district", "tehsil", "village"):
            if level in geo_raw:
                code = lgd_lookup(level, geo_raw[level], context)
                if code:
                    geo[f"{level}_code"] = code

    adm_key = adm_candidates[0] if len(adm_candidates) == 1 else None

    for d in (filename_signals or {}).get("digits", []):
        if len(d) == 10 and d[0] in "6789" and d not in phones:
            phones.append(d)

    return Evidence(
        key=rec.key,
        names=names,
        kin=kin,
        kin_type=kin_type,
        sex=sex,
        birth_years=birth_years,
        ids=ids,
        phones=phones,
        phone_shared=False,
        geo=geo,
        hospital=hospital,
        adm_key=adm_key,
        dates=dates_out,
    )
