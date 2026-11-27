"""B4: Clinical Shield (PRD §6.5, FR-D4, FR-R7)."""

from __future__ import annotations

import re

from typing import Any

from kavachmed.contracts import Context, Record, Span

_RE_DOCTOR_CTX = re.compile(
    r"\b(?:Dr\.?|डॉ\.?|Consultant[ \t]*:|Attending[ \t]*:|Referred[ \t]+by|ref_doctor|Practitioner[ \t]*:)[ \t]*"
    r"(?:(?:Dr\.?|डॉ\.?)[ \t]*)?"  # "Consultant: Dr. X" — the title is not the name
    r"((?!Dr\b)[A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+){0,3}|[\u0900-\u097F]{2,}(?:[ \t]+[\u0900-\u097F]{2,}){0,3})",
)
_RE_KIN_AFTER_DOCTOR = re.compile(
    r"\s*[(,\-]?\s*(?:the\s+)?(?:daughter|son|wife|husband|mother|father|brother|sister|relative|attendant|"
    r"guardian|beti|beta|patni|pati|bhai|behen|बेटी|बेटा|पत्नी|पति)\b",
    re.I,
)
_RE_DRUG_DOSE = re.compile(
    r"\b(?:Tab|Cap|Inj|Syp|Syr|Oint|Drops|Neb|Inhaler)\.?\s+[A-Za-z][A-Za-z0-9-]*"
    r"(?:\s+\d+(?:\.\d+)?\s*(?:mg|mcg|ug|g|gm|gms|ml|IU|%))?"
    r"(?:\s+(?:[0-3]-[0-3]-[0-3](?:-[0-3])?|OD|BD|TDS|QID|HS|SOS|STAT|PRN))?"
    r"(?:\s*[xX×]\s*\d+\s*(?:d|day|days|wk|wks|week|weeks|mo|month|months))?",
    re.I,
)
_RE_FREQ = re.compile(r"\b(?:[0-3]-[0-3]-[0-3](?:-[0-3])?|OD|BD|TDS|QID|HS|SOS|STAT|PRN)\b")
_RE_LAB_LABEL = re.compile(
    r"\b(?:ha?emoglobin|platelets?|"
    r"serum(?=[ \t]+(?:creatinine|urea|glucose|bilirubin|sodium|potassium|calcium|albumin|cholesterol|iron|protein|ferritin)\b)|"
    r"(?:random|fasting|postprandial)[ \t]+(?:blood[ \t]+)?(?:sugar|glucose))\b",
    re.I,
)
_RE_NAME_LABEL_END = re.compile(
    r"(?:\b(?:patient[ \t]+name|pt\.?[ \t]+name|name)|नाम)[ \t]*[:=-][ \t]*$",
    re.I,
)
_RE_VITALS_LABS = re.compile(
    r"\b(?:BP\s*\d{2,3}/\d{2,3}|SpO2\s*\d{2,3}\s*%?|Temp\s*\d{2,3}(?:\.\d+)?\s*[FC]?|"
    r"Pulse\s*\d{2,3}(?:/min)?|HbA1c(?:\s*\d{1,2}(?:\.\d+)?%?)?|Hb(?:\s*\d{1,2}(?:\.\d+)?\s*(?:g/dL)?)?|"
    r"(?:CBC|LFT|KFT|RFT|TSH|SGOT|SGPT|Creatinine|Bilirubin|Urea|Sodium|Potassium|Platelets|ESR|CRP|RBS|FBS|PPBS|"
    r"LDL|HDL|Triglycerides|Cholesterol|Uric\s*Acid|Calcium|Albumin|Globulin|ALP|GGT|Troponin|Ferritin|"
    r"WBC(?:\s*[\d,]+\s*(?:/cumm|/mm3)?)?)|"
    r"ref\s*range\s*\d{1,4}-\d{1,4}|"
    r"Lab\s*Seq\s*\d+\s*(?:mmol/L|mg/dL|IU/L)?)\b",
    re.I,
)
_RE_ICD_LOINC = re.compile(r"\b(?:[A-TV-Z]\d{2}(?:\.\d{1,2})?|\d{1,5}-\d)\b")
_RE_PKG_INR = re.compile(
    r"\b(?:(?:PKG|AB-PMJAY|CGHS|ECHS)[-/ ]?[A-Z0-9-]{3,12}|(?:INR|Rs\.?|₹)\s*[\d,]+(?:\.\d{1,2})?)\b",
    re.I,
)
_RE_EPONYM = re.compile(
    r"\b([A-Z][a-z]+(?:'s|s)?\s+(?:disease|syndrome|palsy|lymphoma|tumor|tumour))\b"
)
_RE_WARD = re.compile(
    r"\b(?:Ward\s*\d+(?:\s*\([A-Za-z]+\))?|Bed\s*\d+|(?:ICU|NICU|PICU|MICU|SICU|OPD|IPD)\b)"
)
_RE_HISTORY = re.compile(
    r"\b(?:K/C/O|H/O|known[ \t]+case[ \t]+of|history[ \t]+of)[ \t]+(.+?)(?=[.;\n]|$|\b(?:reviewed|seen|advised|examined|by[ \t]+patient|patient|attendant|relative)\b)",
    re.I,
)
_RE_ASHA = re.compile(r"\bASHA\s+(?:worker|workers|didi)\b")
_RE_ADDR_HINT = re.compile(
    r"\b(?:h\.?\s*no|house\s*no|mohalla|village|vill\.?|gram|tehsil|district|pin)\b|पता|मोहल्ला|गांव|जिला",
    re.I,
)
_RE_DISTRICT = re.compile(
    r"\b(?:District|Dist\.?)[ \t]+[A-Za-z]+(?:[ \t]+[A-Za-z]+)?|(?:जिला)[ \t]+[\u0900-\u097F]+(?:[ \t]+[\u0900-\u097F]+)?",
    re.I,
)

_SYMPTOM_HINTS = {
    "fever", "cough", "cold", "pain", "headache", "vomiting", "diarrhea", "diarrhoea",
    "breathlessness", "dyspnea", "weakness", "dizziness", "rash", "swelling", "bleeding",
    "chest", "abdominal", "bukhar", "khansi", "dard", "ulti", "dast", "chakkar",
    "बुखार", "खांसी", "दर्द", "उल्टी", "दस्त",
}
_RE_COMPLAINT_PREFIX = re.compile(
    r"(?i:C/O)[ \t]+(?:(?:severe|mild|acute|chronic)[ \t]+)?"
    r"(?:" + "|".join(re.escape(word) for word in sorted(_SYMPTOM_HINTS, key=len, reverse=True)) + r")\b"
    r"(?:[ \t]+(?:and|with)[ \t]+(?:"
    + "|".join(re.escape(word) for word in sorted(_SYMPTOM_HINTS, key=len, reverse=True))
    + r")\b)*(?:[ \t]+(?:for|x)[ \t]+\d+[ \t]*(?:days?|d|wks?|weeks?|yrs?|years?|months?)\b)?",
    re.I,
)


def _add(
    spans: list[Span],
    rec: Record,
    field_idx: int,
    start: int,
    end: int,
    exempt_kind: str,
    role: str = "none",
) -> None:
    spans.append(
        Span(
            key=rec.key,
            field_idx=field_idx,
            start=start,
            end=end,
            typ="CLINICAL",
            role=role,  # type: ignore[arg-type]
            score=1.0,
            src="shield",
            protected=True,
            exempt_kind=exempt_kind,
        )
    )


def protect(rec: Record, ctx: Context | dict[str, Any]) -> list[Span]:
    """Emit protected clinical spans before PII arbitration (FR-D4)."""
    if isinstance(ctx, dict):
        ctx = Context(
            policy=ctx.get("policy", {}),
            lexicons=ctx.get("lexicons", {}),
            coltypes=ctx.get("coltypes", {}),
            doctor_registry=set(ctx.get("doctor_registry", ())),
            keys=ctx.get("keys"),
        )
    spans: list[Span] = []
    reg_lower = {name.strip().lower(): name.strip() for name in ctx.doctor_registry if name and name.strip()}
    drugs: set[str] = ctx.lexicons.get("drugs", set())

    for fld in rec.fields:
        if fld.vtype in {"path", "header", "key"}:
            continue
        text = fld.text
        coltype = ctx.coltypes.get(fld.path) or rec.meta.get("coltypes", {}).get(fld.path)
        is_addr_field = coltype == "address" or bool(_RE_ADDR_HINT.search(text))

        # 1. Doctors (context + full-name registry match only, single-line)
        for m in _RE_DOCTOR_CTX.finditer(text):
            name = m.group(1).strip()
            s, e = m.span(0)
            # "Dr. Anuradha (daughter of patient)": kinship right after the name means a
            # relative who happens to carry the title — not protected as a treating doctor
            if _RE_KIN_AFTER_DOCTOR.match(text[e:e + 40]):
                continue
            kind = "doctor_registry" if name.lower() in reg_lower else "doctor_context"
            _add(spans, rec, fld.idx, s, e, kind, role="doctor")

        if reg_lower:
            text_lower = text.lower()
            for full_lower in sorted(reg_lower):
                start = 0
                while True:
                    idx = text_lower.find(full_lower, start)
                    if idx == -1:
                        break
                    end = idx + len(full_lower)
                    left_ok = idx == 0 or not text[idx - 1].isalpha()
                    right_ok = end == len(text) or not text[end].isalpha()
                    if left_ok and right_ok:
                        _add(spans, rec, fld.idx, idx, end, "doctor_registry", role="doctor")
                    start = end

        # 2. Drugs / dose / frequency + drug lexicon
        for m in _RE_DRUG_DOSE.finditer(text):
            _add(spans, rec, fld.idx, m.start(), m.end(), "dose")
        for m in _RE_FREQ.finditer(text):
            _add(spans, rec, fld.idx, m.start(), m.end(), "dose")
        if drugs:
            for m in re.finditer(r"\b[A-Za-z]{4,}\b", text):
                if m.group().lower() in drugs:
                    _add(spans, rec, fld.idx, m.start(), m.end(), "drug")

        # 3. Vitals & labs
        for m in _RE_VITALS_LABS.finditer(text):
            _add(spans, rec, fld.idx, m.start(), m.end(), "lab")
        for m in _RE_LAB_LABEL.finditer(text):
            if not _RE_NAME_LABEL_END.search(text[max(0, m.start() - 40):m.start()]):
                _add(spans, rec, fld.idx, m.start(), m.end(), "lab")

        # 4. ICD / LOINC / package codes / INR
        for m in _RE_ICD_LOINC.finditer(text):
            _add(spans, rec, fld.idx, m.start(), m.end(), "code")
        for m in _RE_PKG_INR.finditer(text):
            _add(spans, rec, fld.idx, m.start(), m.end(), "code")

        # 5. Eponyms
        for m in _RE_EPONYM.finditer(text):
            _add(spans, rec, fld.idx, m.start(), m.end(), "eponym")

        # 6. Wards / departments (skip residential Ward in address fields)
        if not is_addr_field:
            for m in _RE_WARD.finditer(text):
                _add(spans, rec, fld.idx, m.start(), m.end(), "ward")

        # 7. History (K/C/O, H/O) — narrowed to actual clinical phrase
        for m in _RE_HISTORY.finditer(text):
            _add(spans, rec, fld.idx, m.start(), m.end(), "history")

        # 8. Protect only the recognized complaint, not subsequent people or notes.
        for m in _RE_COMPLAINT_PREFIX.finditer(text):
            _add(spans, rec, fld.idx, m.start(), m.end(), "history")

        # 9. ASHA worker
        for m in _RE_ASHA.finditer(text):
            _add(spans, rec, fld.idx, m.start(), m.end(), "ward")

        # 10. District preservation (B9 policy: keep district)
        for m in _RE_DISTRICT.finditer(text):
            _add(spans, rec, fld.idx, m.start(), m.end(), "district")

    return spans
