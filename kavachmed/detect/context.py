"""B6: Context grammar for names, kinship, honorifics, Hinglish, and third-party cues (FR-D2)."""

from __future__ import annotations

import re

from kavachmed.contracts import Context, Record, Span
from kavachmed.detect.address import _co_is_relative

_LATIN_PART = r"(?:[A-Z][a-zA-Z.]{1,25}|[A-Z]\.?)"
_DEVA_PART = r"[\u0900-\u097F]{2,}"
_NAME_TOKEN_SEQ = rf"({_LATIN_PART}(?:[ \t]+{_LATIN_PART}){{0,3}}|{_DEVA_PART}(?:[ \t]+{_DEVA_PART}){{0,3}})"

_RE_LABEL_PATIENT = re.compile(
    rf"(?i:Pt\.?[ \t]*Name|Patient[ \t]*Name|Patient|Name|नाम|मरीज[ \t]*का[ \t]*नाम)[ \t]*[:\-][ \t]*{_NAME_TOKEN_SEQ}"
)
_RE_KINSHIP = re.compile(
    rf"\b(?i:S/o|D/o|W/o|C/O|पुत्र|पुत्री|पत्नी|पिता|पति)[ \t]*[:\-]?[ \t]*{_NAME_TOKEN_SEQ}"
)
_RE_THIRD_PARTY = re.compile(
    rf"\b(?i:called[ \t]+by|witnessed[ \t]+by|attendant|informant|sample[ \t]+received[ \t]+from|raised_by)[ \t]*[:\-]?[ \t]*{_NAME_TOKEN_SEQ}"
)
_RE_CHAT_SENDER = re.compile(
    r"(?:^|\n)[ \t]*(?:\[[^\]]+\][ \t]*)?([A-Z][a-z]{1,20}(?:[ \t]+(?:Bhaiya|Didi|Ji|Sahab|[A-Z][a-z]{1,20})){0,2}|"
    r"[\u0900-\u097F]{2,}(?:[ \t]+(?:भैया|दीदी|जी|साहब|[\u0900-\u097F]{2,})){0,2})[ \t]*:",
)
_RE_HINGLISH = re.compile(
    r"\b([A-Z][a-z]{2,}(?:[ \t]+[A-Z][a-z]{2,})?|[\u0900-\u097F]{2,}(?:[ \t]+[\u0900-\u097F]{2,}){0,2})[ \t]+(?:ka|ki|ke|ko|ne|का|की|के|को)[ \t]+"
    r"(?:(?:\d+[ \t]+(?:din|day|days|hafte|weeks?|mahine|months?)[ \t]+se[ \t]+)?)"
    r"(?:BP|bp|sugar|bukhar|dard|report|ilaj|dawai|बुखार|दर्द|जांच)\b",
)
_RE_HINGLISH_LOWER = re.compile(
    r"(?<![A-Za-z])([a-z]{3,20})[ \t]+(?:ko|ka|ki|ke|ne)[ \t]+"
    r"(?:\d+[ \t]+(?:din|dino|hafte|mahine|saal)[ \t]+se[ \t]+)?"
    r"(?:bp|sugar|bukhar|bukhaar|dard|khansi|ulti|dast|chakkar|kamzori|saans|jukam|report|ilaj|dawai|takleef|pareshani)\b"
)
_HINGLISH_STOP = frozenset({
    "unko", "inko", "usko", "isko", "mujhe", "hamko", "aapko", "unka", "inka", "uska", "iska", "mera",
    "tera", "patient", "mareez", "mariz", "bachche", "bachcha", "beti", "beta", "pati", "patni", "maa",
    "papa", "bhai", "behen", "dadi", "dada", "nani", "nana", "sab", "kal", "aaj", "raat", "subah",
})
_RE_HONORIFIC_LATIN = re.compile(
    r"\b([A-Z][a-z]{2,}(?:[ \t]+[A-Z][a-z]{2,})?"
    r"[ \t]+(?:ji|sahab|saab|sahib|didi|bhaiya|bhai|babu|amma|[a-z]{2,12}ji))\b",
    re.I,
)
_RE_HONORIFIC_DEVA_PRE = re.compile(r"(?:श्री|श्रीमती)[ \t]+([\u0900-\u097F]{2,}(?:[ \t]+[\u0900-\u097F]{2,})?)")
_RE_HONORIFIC_DEVA_POST = re.compile(r"([\u0900-\u097F]{2,})[ \t]+(?:जी|साहब|देवी)")
_RE_SURNAME_FIRST = re.compile(r"\b([A-Z][a-z]{2,},[ \t]*[A-Z][a-z]{2,})\b")

_CLINICAL_CHAT_PREFIXES = {
    "name", "pt", "pt name", "pt. name", "patient", "patient name", "full name",
    "bp", "temp", "pulse", "spo2", "ward", "bed", "date", "time", "note", "notes", "rx", "dx",
    "नाम", "पता", "उम्र", "आयु", "मरीज का नाम",
}
# Address/record-structure words that make "X, Y" an address or label list, not "Surname, First".
_NOT_NAME_WORDS = {
    "near", "opp", "behind", "distt", "dist", "district", "tehsil", "taluka", "block", "village",
    "vill", "gram", "post", "mohalla", "colony", "nagar", "ward", "road", "lane", "gali", "marg",
    "basti", "sector", "city", "state", "pin", "patient", "name", "age", "sex", "male", "female",
}
_COMPLAINT_WORDS = {"fever", "cough", "cold", "pain", "headache", "vomiting", "diarrhea", "weakness", "bukhar", "dard"}
_KINSHIP_HONORIFICS = {"bhaiya", "didi", "ji", "sahab", "भैया", "दीदी", "जी", "साहब"}


def _emit(
    spans: list[Span],
    rec: Record,
    field_idx: int,
    start: int,
    end: int,
    typ: str,
    role: str,
    score: float,
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
            src="context",
        )
    )


def detect_context(rec: Record, ctx: Context) -> list[Span]:
    """Detect named entities using labels, kinship, honorifics, Hinglish, and third-party cues."""
    spans: list[Span] = []

    for fld in rec.fields:
        if fld.vtype in {"path", "header", "key"}:
            continue
        text = fld.text

        for m in _RE_LABEL_PATIENT.finditer(text):
            _emit(spans, rec, fld.idx, m.start(1), m.end(1), "PATIENT_NAME", "patient", 0.92)

        for m in _RE_KINSHIP.finditer(text):
            cand = m.group(1).strip()
            if cand.lower().split()[0] in _COMPLAINT_WORDS:
                continue
            if m.group(0).lstrip().lower().startswith("c/o") and not _co_is_relative(text[m.end(0) - len(m.group(1)) - 1:]):
                continue  # PRD C/O resolver: complaint, not care-of
            _emit(spans, rec, fld.idx, m.start(1), m.end(1), "RELATIVE_NAME", "relative", 0.90)

        for m in _RE_THIRD_PARTY.finditer(text):
            _emit(spans, rec, fld.idx, m.start(1), m.end(1), "RELATIVE_NAME", "third_party", 0.88)

        for m in _RE_CHAT_SENDER.finditer(text):
            cand = m.group(1).strip()
            low_cand = re.sub(r"\s+", " ", cand.lower())
            if low_cand in _CLINICAL_CHAT_PREFIXES:
                continue
            toks_low = set(low_cand.split())
            role = "relative" if (toks_low & _KINSHIP_HONORIFICS) else "sender"
            _emit(spans, rec, fld.idx, m.start(1), m.end(1), "RELATIVE_NAME", role, 0.85)

        for m in _RE_HINGLISH.finditer(text):
            _emit(spans, rec, fld.idx, m.start(1), m.end(1), "PATIENT_NAME", "patient", 0.82)

        # lower-case Hinglish subject ("rukmini ko 3 din se bukhar hai"): a word that is not
        # clinical/Hinglish vocabulary before ko/ka/ki/ke/ne + a complaint is a person
        safe = ctx.lexicons.get("safe_words", set())
        for m in _RE_HINGLISH_LOWER.finditer(text):
            w = m.group(1)
            if w in safe or w in _HINGLISH_STOP:
                continue
            _emit(spans, rec, fld.idx, m.start(1), m.end(1), "PATIENT_NAME", "patient", 0.80)

        for m in _RE_HONORIFIC_LATIN.finditer(text):
            _emit(spans, rec, fld.idx, m.start(1), m.end(1), "PATIENT_NAME", "patient", 0.84)

        for m in _RE_HONORIFIC_DEVA_PRE.finditer(text):
            _emit(spans, rec, fld.idx, m.start(1), m.end(1), "PATIENT_NAME", "patient", 0.86)

        for m in _RE_HONORIFIC_DEVA_POST.finditer(text):
            _emit(spans, rec, fld.idx, m.start(1), m.end(1), "PATIENT_NAME", "patient", 0.84)

        for m in _RE_SURNAME_FIRST.finditer(text):
            parts = [w.strip().lower() for w in m.group(1).split(",")]
            if any(w in _NOT_NAME_WORDS or w in ctx.lexicons.get("places", set())
                   or w in ctx.lexicons.get("safe_words", set()) for w in parts):
                continue
            _emit(spans, rec, fld.idx, m.start(1), m.end(1), "PATIENT_NAME", "patient", 0.80)

    return spans
