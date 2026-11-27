"""B8: Default-deny name arbitration (FR-D11)."""

from __future__ import annotations

import re

from kavachmed.contracts import Context, Record, Span

_RE_TOKEN = re.compile(r"[A-Za-z]{2,}|[\u0900-\u097F]{2,}")
_RE_DEVA_LETTER = re.compile(r"[\u0904-\u0939\u0958-\u0961\u0972-\u097F]")
_RE_CLINICAL_SUFFIX = re.compile(
    r"(?:itis|osis|emia|pathy|oma|ectomy|otomy|ostomy|scopy|uria|algia|plasty|megaly|rrhoea|rrhea|penia"
    r"|ology|ologist|iatry|iatric|cillin|mycin|cycline|floxacin|azole|prazole|pril|sartan|olol|dipine"
    r"|statin|formin|gliptin|mab|tinib|vir|thiazide|semide|sone|olone|caine|parin|dronate|triptan)$",
    re.I,
)
# Fields whose key names clinical content never get name *guesses* (default-deny); typed
# identifier rules and context grammar still run there.
CLINICAL_KEYS = frozenset({
    "drug", "drugs", "medicine", "medicines", "medication", "medications", "dose", "strength",
    "frequency", "route", "duration", "test", "test_name", "tests", "result", "results", "value",
    "unit", "units", "ref_range", "reference_range", "range", "diagnosis", "dx", "display", "code",
    "icd", "icd10", "loinc", "procedure", "procedures", "department", "dept", "specialty", "ward",
    "bed", "package", "package_code", "amount", "amount_inr", "status", "category", "type",
})
# Record-structure labels ("Patient Name:", "Mobile:") are not names in any script.
_LABEL_WORDS = frozenset({
    "patient", "name", "pt", "age", "sex", "gender", "male", "female", "address", "mobile", "phone",
    "contact", "father", "mother", "husband", "wife", "guardian", "relative", "attendant", "informant",
    "kin", "admitted", "discharged", "diagnosis", "labs", "ref", "reg", "seen", "routine",
    "district", "distt", "zila", "jila", "village", "tehsil", "taluka", "state", "pin", "uhid", "mrn", "ipd", "opd", "dob",
    "date", "visit", "ward", "bed", "no", "id", "nam", "naam", "ling", "umar", "pata", "gaon", "jila",
    # Devanagari age/duration units and labels ("48 वर्ष", "3 दिन से")
    "वर्ष", "साल", "माह", "महीने", "दिन", "सप्ताह", "आयु", "उम्र", "नाम", "पता", "पुरुष", "महिला", "जिला",
})
_DISTRICT_KEYS = frozenset({"district", "dist", "distt", "zila", "jila", "जिला", "ज़िला"})
# Village/locality/address fields: a district-named village ("Gopalganj") is still a village.
_RE_SUBADDRESS_KEY = re.compile(r"village|gaon|gram|mohalla|locality|street|tehsil|address|addr|pata|patta|पता|गांव|ग्राम", re.I)
_RE_RESERVED = re.compile(r"\bPSN-[0-9A-Za-z]{6,12}\b|\[[A-Z0-9_/-]{2,20}\]")


def _overlaps(start: int, end: int, blocked: list[tuple[int, int]]) -> bool:
    return any(not (end <= bs or start >= be) for bs, be in blocked)


def detect_default_deny(
    rec: Record,
    ctx: Context,
    shield_spans: list[Span],
    existing_spans: list[Span] | None = None,
) -> list[Span]:
    """Emit default-deny name candidates for unlabelled Title-Case/ALL-CAPS Latin & Devanagari words."""
    safe_words: set[str] = ctx.lexicons.get("safe_words", set())
    name_gazetteer: set[str] = ctx.lexicons.get("names", set())
    places: set[str] = ctx.lexicons.get("places", set())
    score = float(ctx.policy.get("detect", {}).get("default_deny", {}).get("score", 0.45))

    blocked_by_field: dict[int, list[tuple[int, int]]] = {}
    for sp in shield_spans:
        if sp.protected and sp.key == rec.key:
            blocked_by_field.setdefault(sp.field_idx, []).append((sp.start, sp.end))
    for sp in existing_spans or []:
        if sp.key == rec.key:
            blocked_by_field.setdefault(sp.field_idx, []).append((sp.start, sp.end))

    out: list[Span] = []
    for fld in rec.fields:
        if fld.vtype in {"path", "header", "key"}:
            continue
        text = fld.text
        leaf_key = fld.path.split("/")[-1].strip().lower()
        if leaf_key in CLINICAL_KEYS or leaf_key in _DISTRICT_KEYS:
            continue
        blocked = list(blocked_by_field.get(fld.idx, []))
        # Reserve whole PSNs and [PLACEHOLDER]s before word tokenization
        for rm in _RE_RESERVED.finditer(text):
            blocked.append((rm.start(), rm.end()))

        candidates: list[tuple[int, int]] = []
        for m in _RE_TOKEN.finditer(text):
            s, e = m.span()
            tok = m.group()
            if _overlaps(s, e, blocked):
                continue
            low = tok.lower()
            in_place_part = bool(_RE_SUBADDRESS_KEY.search(fld.path.split("/")[-1]))
            if low in safe_words or low in _LABEL_WORDS or (
                low in places and low not in name_gazetteer and not in_place_part
            ):
                continue
            if _RE_CLINICAL_SUFFIX.search(tok):
                continue
            # part of an alphanumeric code (SG050A, HbA1c, T2DM): not a word, not a name
            if (s > 0 and text[s - 1].isdigit()) or (e < len(text) and text[e].isdigit()):
                continue

            is_deva_word = bool(_RE_DEVA_LETTER.search(tok))
            is_title_or_caps = tok[0].isupper() and (tok.isupper() or tok[1:].islower())
            is_lower_gazetteer = tok.islower() and (low in name_gazetteer)

            if is_deva_word or is_title_or_caps or is_lower_gazetteer:
                candidates.append((s, e))

        # Merge adjacent candidates separated only by whitespace
        merged: list[tuple[int, int]] = []
        for s, e in candidates:
            if merged and text[merged[-1][1] : s].strip() == "":
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))

        for s, e in merged:
            out.append(
                Span(
                    key=rec.key,
                    field_idx=fld.idx,
                    start=s,
                    end=e,
                    typ="PATIENT_NAME",
                    role="unknown",
                    score=score,
                    src="default_deny",
                )
            )

    return out
