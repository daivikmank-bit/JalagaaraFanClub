"""B9: Address segmenter (FR-D3) — drops granular address parts, keeps district."""

from __future__ import annotations

import re

from kavachmed.contracts import Context, Record, Span

_DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

_RE_CO_PREFIX = re.compile(r"^\s*C/O[ \t]+([^,\n]+)", re.I)
_RE_HOUSE = re.compile(
    r"\b(?:H\.?\s*No\.?|House\s*No\.?|Plot\s*No\.?|Flat\s*No\.?)\s*[:#-]?\s*[A-Za-z0-9/-]+|"
    r"\bHouse\s+\d+[A-Za-z0-9/-]*|"
    r"(?:मकान\s*नं\.?)\s*[:#-]?\s*[A-Za-z0-9\u0966-\u096F/-]+",
    re.I,
)
_RE_LOCALITY = re.compile(
    r"\b(?:Ward\s*(?:No\.?)?\s*\d+|(?:Mohalla|Gali|Colony|Nagar|Sector|Block)\s+[A-Za-z0-9 ]+?)(?=,|$|\n)|"
    r"(?:वार्ड\s*[\d\u0966-\u096F]+|(?:मोहल्ला|गली|कॉलोनी|नगर)\s+[\u0900-\u097F ]+?)(?=,|$|\n)",
    re.I,
)
_RE_LANDMARK = re.compile(
    r"\b(?:near|opp\.?|opposite|behind|beside)\s+[^,\n]+|(?:के\s+पास|के\s+सामने|के\s+पीछे)\s+[^,\n]+",
    re.I,
)
_RE_VILLAGE = re.compile(
    r"\b(?:Village|Vill\.?|Gram|Tehsil|Taluka|Tal\.?|Post)\s+[^,\n]+|(?:गांव|ग्राम|तहसील)\s+[^,\n]+",
    re.I,
)
_RE_DISTRICT = re.compile(
    r"\b(?:District|Dist\.?)\s+[A-Za-z]+(?:[ \t]+[A-Za-z]+)?|(?:जिला)\s+[\u0900-\u097F]+(?:[ \t]+[\u0900-\u097F]+)?",
    re.I,
)
_RE_PIN_LABELED = re.compile(r"(?:\bPIN(?:\s*Code)?|पिन(?:\s*कोड)?)\s*[:\-]?\s*([1-8]\d{5})\b", re.I)
_RE_PIN_BARE = re.compile(r"\b([1-8]\d{5})\b")
_RE_ADDR_CONTEXT = re.compile(
    r"\b(?:h\.?\s*no|house|ward|mohalla|gali|colony|village|vill\.?|gram|tehsil|district|dist\.?|pin|nivas|niwas|sthan|abode)\b|"
    r"पता|निवास|मकान|वार्ड|मोहल्ला|गांव|ग्राम|तहसील|जिला|पिन",
    re.I,
)


def segment_address(rec: Record, ctx: Context) -> list[Span]:
    """Drop house/ward/mohalla/landmark/village/tehsil/PIN while protecting district."""
    spans: list[Span] = []

    def _emit(fld_idx: int, s: int, e: int, typ: str, role: str = "none", protected: bool = False) -> None:
        spans.append(
            Span(
                key=rec.key,
                field_idx=fld_idx,
                start=s,
                end=e,
                typ=typ,
                role=role,  # type: ignore[arg-type]
                score=1.0 if protected else 0.90,
                src="shield" if protected else "context",
                protected=protected,
                exempt_kind="district" if protected else None,
            )
        )

    for fld in rec.fields:
        if fld.vtype in {"path", "header", "key"}:
            continue
        text = fld.text
        norm = text.translate(_DEVA_DIGITS)
        coltype = ctx.coltypes.get(fld.path) or rec.meta.get("coltypes", {}).get(fld.path)
        leaf_key = fld.path.split("/")[-1].strip().lower()
        if leaf_key in _CLINICAL_FIELD_KEYS:
            continue  # lab values, codes, wards, departments: never address parts
        key_is_addr = bool(_RE_ADDR_COL.search(leaf_key)) or leaf_key in {"pin", "pincode", "pin_code", "zip"}
        is_addr_ctx = key_is_addr or bool(_RE_ADDR_CONTEXT.search(text)) or (
            coltype in {"address", "pin"} and not fld.path.startswith("lines/")
        )

        for m in _RE_CO_PREFIX.finditer(text):
            if not _co_is_relative(m.group(1)):
                continue  # PRD C/O resolver: "C/O fever x 3 days" is a complaint
            _emit(fld.idx, m.start(1), m.end(1), "RELATIVE_NAME", role="relative")
        for m in _RE_HOUSE.finditer(text):
            _emit(fld.idx, m.start(), m.end(), "HOUSE")
        for m in _RE_LOCALITY.finditer(text):
            if m.group(0).lower().startswith("ward") and not (key_is_addr or _RE_ADDR_CONTEXT_STRICT.search(text)):
                continue  # "Ward 4 (Medicine)" is a hospital ward unless the text is an address
            _emit(fld.idx, m.start(), m.end(), "LOCALITY")
        for m in _RE_LANDMARK.finditer(text):
            _emit(fld.idx, m.start(), m.end(), "LANDMARK")
        for m in _RE_VILLAGE.finditer(text):
            _emit(fld.idx, m.start(), m.end(), "VILLAGE")

        pin_spans: list[tuple[int, int]] = []
        for m in _RE_PIN_LABELED.finditer(norm):
            s, e = m.span(1)
            pin_spans.append((s, e))
            _emit(fld.idx, s, e, "PIN")

        if is_addr_ctx and coltype != "id":
            for m in _RE_PIN_BARE.finditer(norm):
                s, e = m.span(1)
                if any(ps <= s and e <= pe for ps, pe in pin_spans):
                    continue
                if s > 0 and norm[s - 1] == "/":
                    continue
                _emit(fld.idx, s, e, "PIN")

        # FR-D3: in address fields keep only the district/state. Any comma segment that is not
        # exactly a known district/state (or a "Distt./District X" segment) is a locality,
        # even when its words also appear in the LGD list (e.g. "Basti Ramnagar").
        leaf = fld.path.split("/")[-1]
        is_free_text_line = fld.path.startswith(("lines/", "text", "doc")) or leaf.isdigit()
        regions: list[tuple[int, int]] = []
        if _RE_ADDR_COL.search(leaf):  # key names an address (column typing alone is too noisy)
            regions.append((0, len(text)))
        else:
            # "Address (attendant): ..." inside free text: segment the rest of that line
            for lm in _RE_ADDR_LABEL.finditer(text):
                eol = text.find("\n", lm.end())
                regions.append((lm.end(), len(text) if eol < 0 else eol))
        for r_s, r_e in regions:
            places = ctx.lexicons.get("places", set())
            covered = [(sp.start, sp.end) for sp in spans if sp.field_idx == fld.idx]
            for m in _RE_SEGMENT.finditer(text, r_s, r_e):
                s, e = m.start(), m.end()
                seg = text[s:e].strip(" .")
                low = re.sub(r"^(?:distt?\.?|district|jila|जिला)\s*", "", seg.lower()).strip(" .")
                if not seg or low in places or low in _ADDR_KEEP:
                    continue
                if any(cs < e and s < ce for cs, ce in covered):
                    continue  # a specific rule (VILLAGE, HOUSE, PIN, ...) already typed part of it
                if not any(ch.isalnum() for ch in seg):
                    continue
                # bare digit segments are house numbers (PINs were emitted above)
                _emit(fld.idx, s, e, "LOCALITY" if any(ch.isalpha() for ch in seg) else "HOUSE")

    return spans


_CLINICAL_FIELD_KEYS = frozenset({
    "value", "result", "results", "unit", "units", "ref_range", "reference_range", "range", "code",
    "loinc", "icd", "icd10", "ward", "bed", "department", "dept", "drug", "dose", "strength",
    "package_code", "amount", "amount_inr", "test", "test_name", "display", "diagnosis",
})
# Address words other than "ward" (a hospital ward is not an address cue on its own)
_RE_ADDR_CONTEXT_STRICT = re.compile(
    r"\b(?:address|addr|pata|h\.?\s*no|house|mohalla|gali|colony|village|vill\.?|gram|tehsil|district|dist\.?|pin|post|thana)\b|"
    r"पता|मकान|मोहल्ला|गांव|ग्राम|तहसील|जिला|पिन",
    re.I,
)
_RE_CO_COMPLAINT = re.compile(
    r"^\s*[:\-]?\s*(?:[a-z0-9]|(?i:pain|fever|cough|headache|vomiting|weakness|giddiness|swelling|burning)\b)"
    r"|(?i:\b(?:x|for|since)\s*\d+\s*(?:days?|weeks?|wks?|months?|yrs?|years?|hrs?|hours?)\b)"
    r"|(?i:\d+\s*(?:din|hafte|mahine|दिन|हफ्ते|महीने))"
)
_RE_CO_NAME_START = re.compile(r"^\s*[:\-]?\s*(?:Shri|Smt|Sri|Mr|Mrs|Late|श्री|श्रीमती)?\.?\s*(?:[A-Z][a-zA-Z.]+|[\u0900-\u097F]{2,})")


def _co_is_relative(after: str) -> bool:
    """C/O resolver (PRD §6.5): care-of relative only if a name starts it and nothing clinical follows."""
    head = after.split(",")[0]
    if _RE_CO_COMPLAINT.search(head):
        return False
    return bool(_RE_CO_NAME_START.match(head))


_RE_ADDR_COL = re.compile(r"address|addr|pata|patta|पता|residence|locality|ghar|nivas|niwas|sthan|abode|निवास", re.I)
_RE_SEGMENT = re.compile(r"[^,\n;]+")
_RE_ADDR_LABEL = re.compile(r"(?i:\b(?:address|addr|pata|residence)\b)[^:\n]{0,30}:|पता\s*:")
_ADDR_KEEP = {"india", "c/o"}
