"""Layer B: independent default-deny residual re-scan on bytes read back from disk (D5, FR-R1..R6).

STRICT INDEPENDENCE RULE (FR-R1):
May ONLY import stdlib, optional `regex`, static lexicon loaders, and `kavachmed.contracts`.
MUST NEVER import from `detect`, `shield`, `evidence`, `linkage`, `residual_a`, `typing`, `propagate`, or `privacy`.
"""
from __future__ import annotations

import base64
import bisect
import binascii
import csv
import gzip
import html
import io
import re
import unicodedata
from pathlib import Path

from kavachmed.contracts import Hit

_LEX_ROOT = Path(__file__).resolve().parent.parent / "config" / "lexicons"

_ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\u200e\u200f\ufeff\u00ad]")
_PCT_HEX_RE = re.compile(r"(?:%[0-9a-fA-F]{2})+")
_DIGIT_XLATE = str.maketrans(
    "०१२३४५६७८९٠١٢٣٤٥٦٧٨٩০১২৩৪৫৬৭৮৯",
    "012345678901234567890123456789",
)
_HOMOGLYPH_XLATE = str.maketrans(
    {
        "\u0410": "A", "\u0430": "a",
        "\u0412": "B", "\u0432": "b",
        "\u0415": "E", "\u0435": "e",
        "\u041a": "K", "\u043a": "k",
        "\u041c": "M", "\u043c": "m",
        "\u041d": "H", "\u043d": "h",
        "\u041e": "O", "\u043e": "o",
        "\u0420": "P", "\u0440": "p",
        "\u0421": "C", "\u0441": "c",
        "\u0422": "T", "\u0442": "t",
        "\u0425": "X", "\u0445": "x",
        "\u0423": "Y", "\u0443": "y",
        "\u0406": "I", "\u0456": "i",
        "\u0391": "A", "\u03b1": "a",
        "\u0392": "B", "\u03b2": "b",
        "\u0395": "E", "\u03b5": "e",
        "\u0396": "Z", "\u03b6": "z",
        "\u0397": "H", "\u03b7": "n",
        "\u0399": "I", "\u03b9": "i",
        "\u039a": "K", "\u03ba": "k",
        "\u039c": "M", "\u03bc": "m",
        "\u039d": "N", "\u03bd": "v",
        "\u039f": "O", "\u03bf": "o",
        "\u03a1": "P", "\u03c1": "p",
        "\u03a4": "T", "\u03c4": "t",
        "\u03a7": "X", "\u03c7": "x",
    }
)

_SAFE_ABBR = {
    "ABDM", "ADM", "AGE", "ALL", "AM", "API", "ATT", "BD", "BP", "CAD", "CAP",
    "CBC", "CKD", "COPD", "COVID", "CSV", "CT", "CXR", "DM", "DM2", "DNA",
    "DOB", "DR", "ECG", "ENT", "ESR", "FBS", "FHIR", "GMT", "HB", "HBA1C",
    "HDL", "HIV", "HR", "HS", "HTN", "ICD", "ICU", "ID", "IHD", "IN", "INR",
    "IPD", "IST", "IU", "IV", "JSON", "KCO", "LDL", "LFT", "LOINC", "MCG",
    "MD", "MG", "ML", "MMHG", "MRI", "MRN", "NA", "NLEM", "NO", "OD", "OK",
    "OPD", "OR", "OT", "PAN", "PIN", "PM", "PMJAY", "PO", "PPBS", "PRN", "PSN",
    "QID", "RBC", "RBS", "RFT", "RNA", "RS", "SOS", "SPO2", "STAT", "SYP",
    "SYR", "TAB", "TB", "TDS", "TSH", "TSV", "TXT", "UHID", "URI", "URN",
    "URTI", "UTC", "UTF", "UTI", "UUID", "WBC", "WHO", "XHTML", "XML", "XRAY",
    "ASHA",
}

_PLACEHOLDER_RE = re.compile(r"^\[[A-Z0-9_/-]{2,24}\]$")
_ANY_PLACEHOLDER_RE = re.compile(r"\[[A-Z0-9_/-]{2,24}\]")
_PSN_OR_ID_RE = re.compile(r"^(?:PSN|ID|TEL)-[0-9A-HJKMNP-TV-Z]{6,10}$")
_PSN_OR_ID_INLINE_RE = re.compile(r"\b(?:PSN|ID|TEL)-[0-9A-HJKMNP-TV-Z]{6,10}\b")
_SAFE_MEDICAL_EPONYMS_RE = re.compile(
    r"\b(?:bell'?s\s+palsy|addison'?s\s+disease|hodgkin'?s?\s+lymphoma|parkinson'?s\s+disease|"
    r"alzheimer'?s\s+disease|cushing'?s\s+syndrome|crohn'?s\s+disease|graves'?\s+disease|"
    r"wilson'?s\s+disease|paget'?s\s+disease)\b",
    re.I,
)

_SEED_PII_LABELS = {
    "name", "patient_name", "patientname", "pt_name", "full_name", "fullname",
    "first_name", "firstname", "last_name", "lastname", "naam", "rogi_ka_naam",
    "relative_name", "father_name", "mother_name", "guardian", "attendant",
    "informant", "pita_pati_ka_naam", "phone", "mobile", "mobile_no", "contact_no",
    "aadhaar", "aadhar", "aadhaar_no", "aadhar_no", "abha", "abha_no", "pan",
    "payer_pan", "email", "dob", "birthdate", "birth_date", "janm_tithi",
    "village", "gaon", "tehsil", "mohalla", "house_no", "pin", "pincode",
    "pin_code", "zip", "postalcode", "patient_ref", "uhid", "mrn", "pmjay_id",
    "policy_no", "raised_by", "exported_by", "source_path",
    "नाम", "पता", "गाँव", "गांव", "तहसील", "जन्म तिथि", "आधार", "मोबाइल",
}

_LATIN_TOK_RE = re.compile(r"\b[A-Za-z][A-Za-z.'-]{1,32}\b")
_DEVA_TOK_RE = re.compile(r"[\u0900-\u097F]{2,32}")

_DIGIT_GROUP_RE = re.compile(r"(?<![A-Za-z0-9])\d(?:[\s./-]?\d){5,19}(?![A-Za-z0-9])")
_UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)
_ISO_DATE_RE = re.compile(r"\b(?:19|20)\d{2}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2})?")
_NUM_DATE_RE = re.compile(r"\b\d{1,2}[/.-]\d{1,2}[/.-](?:19|20)?\d{2}\b")
_B2_EXEMPT_BEFORE_RE = re.compile(
    r'(?:rs\.?|inr|₹|pkg|package|loinc|icd|ref|range|dose|bp|temp)\s*[:#/-]?\s*$|'
    r'"(?:loinc|unit|range|ref_range)"\s*:\s*"[^"]*"\s*,\s*"value"\s*:\s*$',
    re.I,
)
_B2_EXEMPT_AFTER_RE = re.compile(
    r'^\s*(?:mg|mcg|ml|g/dl|iu|/cumm|/ul|/µl|cells|mmhg|%|/-)\b|'
    r'^\s*,?\s*"unit"\s*:\s*"(?:/ul|/µl|/cumm|cells|g/dl|mg/dl|mmol/l|iu/l|u/l|mmhg|%|ng/ml|pg/ml)"',
    re.I,
)

_B4_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("AADHAAR", re.compile(r"(?<!\d)[2-9]\d{3}[\s.-]?\d{4}[\s.-]?\d{4}(?!\d)")),
    ("AADHAAR_VID", re.compile(r"(?<!\d)[2-9]\d{3}(?:[\s.-]?\d{4}){3}(?!\d)")),
    ("AADHAAR_MASKED", re.compile(r"(?<![A-Za-z0-9])[Xx*]{4}[\s.-]?[Xx*]{4}[\s.-]?\d{4}(?!\d)")),
    ("ABHA", re.compile(r"(?<!\d)91[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}(?!\d)")),
    ("ABHA_ADDR", re.compile(r"\b[A-Za-z0-9._-]+@(?:abdm|sbx)\b", re.I)),
    ("PHONE", re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")),
    ("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
    ("EPIC", re.compile(r"\b[A-Z]{3}\d{7}\b")),
    ("GSTIN", re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b")),
    ("EMAIL", re.compile(r"[\w.+-]+@(?!(?:redacted\.invalid)\b)[\w-]+\.[\w.]{2,}")),
    ("UPI", re.compile(r"\b[\w.-]+@(?:ok[a-z]+|ybl|ibl|axl|paytm|upi)\b", re.I)),
    ("MRN", re.compile(r"\b(?:SNJ|JJH|ASS|DHH|UHID|MRN|IPD|OPD|ADM)[-/]\d{2,4}[-/]\d{4,8}\b", re.I)),
    ("PLATE", re.compile(r"\b[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}\b")),
]

_B3_KV_RE = re.compile(
    r"(?m)^\s*(?P<label>[A-Za-z\u0900-\u097F][A-Za-z0-9\u0900-\u097F _./-]{1,32})\s*:\s*(?P<val>[^\n;|]{1,80})"
)

_B64_RE = re.compile(r"[A-Za-z0-9+/]{24,}={0,2}")
_HEX_ESC_RE = re.compile(r"(?:\\x[0-9a-fA-F]{2})+")
_UNI_ESC_RE = re.compile(r"(?:\\u[0-9a-fA-F]{4})+")

_CLINICAL_SUFFIXES = (
    "itis", "osis", "emia", "pathy", "oma", "ectomy", "otomy", "ostomy", "scopy", "uria", "algia",
    "plasty", "megaly", "rrhea", "rrhoea", "penia", "phobia", "lysis", "gram", "graphy",
    "ology", "ologist", "iatry", "iatric", "cillin", "mycin", "cycline", "floxacin", "azole", "prazole",
    "pril", "sartan", "olol", "dipine", "statin", "formin", "gliptin", "mab", "tinib", "thiazide",
    "semide", "olone", "caine", "parin", "dronate", "triptan",
)


def _pct_unquote(text: str) -> str:
    if "%" not in text:
        return text

    def _repl(m: re.Match[str]) -> str:
        try:
            return bytes.fromhex(m.group(0).replace("%", "")).decode("utf-8", errors="ignore")
        except ValueError:
            return m.group(0)

    return _PCT_HEX_RE.sub(_repl, text)


class _StaticLexicons:
    def __init__(self, lex_dir: Path | None = None) -> None:
        base = lex_dir or _LEX_ROOT
        self.safe_words: set[str] = set()
        self.names: set[str] = set()
        self.districts: set[str] = set()
        self.states: set[str] = set()

        safe_p = base / "safe" / "safe_words.txt.gz"
        if safe_p.is_file():
            with gzip.open(safe_p, "rt", encoding="utf-8") as f:
                for line in f:
                    w = line.strip().casefold()
                    if w:
                        self.safe_words.add(w)

        names_p = base / "names" / "names_seed.txt.gz"
        if names_p.is_file():
            with gzip.open(names_p, "rt", encoding="utf-8") as f:
                for line in f:
                    w = line.strip().casefold()
                    if w:
                        self.names.add(w)

        dist_p = base / "places" / "lgd_districts.tsv"
        if dist_p.is_file():
            for line in dist_p.read_text(encoding="utf-8").splitlines()[1:]:
                parts = line.split("\t")
                if len(parts) >= 4:
                    self.districts.add(parts[2].strip().casefold())
                    if parts[3].strip():
                        self.districts.add(parts[3].strip().casefold())

        # Static clinical vocabulary (shared file, not B's code) + drug/condition lexicon tokens;
        # name-gazetteer tokens (both name lists) are subtracted so names are never cleared.
        for gen_names in (base / "names" / "generic_names_seed.txt", base / "names" / "indianpersona_names.txt"):
            if not gen_names.is_file():
                continue
            self.names.update(w.strip().casefold() for w in gen_names.read_text(encoding="utf-8").splitlines()
                              if w.strip() and not w.startswith("#"))
        clinical: set[str] = set()
        vocab_p = base / "clinical" / "clinical_vocab.txt"
        if vocab_p.is_file():
            for line in vocab_p.read_text(encoding="utf-8").splitlines():
                if not line.strip().startswith("#"):
                    clinical.update(w.casefold() for w in line.split())
        for fname in ("drugs.txt", "symptoms_conditions.txt"):
            fp = base / "clinical" / fname
            if fp.is_file():
                for line in fp.read_text(encoding="utf-8").splitlines():
                    if line.strip() and not line.startswith("#"):
                        clinical.update(w.casefold() for w in line.replace("-", " ").split() if len(w) > 1)
        self.safe_words |= clinical - self.names

        st_p = base / "places" / "lgd_states.tsv"
        if st_p.is_file():
            for line in st_p.read_text(encoding="utf-8").splitlines()[1:]:
                parts = line.split("\t")
                if len(parts) >= 3:
                    self.states.add(parts[1].strip().casefold())
                    if parts[2].strip():
                        self.states.add(parts[2].strip().casefold())


_LEX_CACHE: _StaticLexicons | None = None


def _get_lexicons() -> _StaticLexicons:
    global _LEX_CACHE
    if _LEX_CACHE is None:
        _LEX_CACHE = _StaticLexicons()
    return _LEX_CACHE


def normalize_for_scan(text: str) -> str:
    """Layer B independent normalization: NFKC, zero-width strip, homoglyph fold, Indic digits -> ASCII."""
    s = unicodedata.normalize("NFKC", text)
    s = _ZERO_WIDTH_RE.sub("", s)
    s = s.translate(_HOMOGLYPH_XLATE)
    s = s.translate(_DIGIT_XLATE)
    return s


def _decode_hidden_payloads(text: str) -> list[tuple[int, int, str]]:
    """Extract (start, end, decoded_text) from base64, %xx, HTML entities, \\uXXXX, and \\xXX."""
    extras: list[tuple[int, int, str]] = []
    for m in _B64_RE.finditer(text):
        tok = m.group(0)
        try:
            raw = base64.b64decode(tok + "=" * (-len(tok) % 4))
            dec = raw.decode("utf-8", errors="ignore")
            if sum(ch.isprintable() for ch in dec) > len(dec) * 0.7 and len(dec.strip()) >= 4:
                extras.append((m.start(), m.end(), dec))
        except (binascii.Error, ValueError):
            pass
    if "%" in text:
        uq = _pct_unquote(text)
        if uq != text:
            extras.append((0, 0, uq))
    if "&" in text and ";" in text:
        he = html.unescape(text)
        if he != text:
            extras.append((0, 0, he))
    for m in _UNI_ESC_RE.finditer(text):
        try:
            extras.append((m.start(), m.end(), m.group(0).encode("ascii").decode("unicode_escape")))
        except Exception:
            pass
    for m in _HEX_ESC_RE.finditer(text):
        try:
            extras.append((m.start(), m.end(), m.group(0).encode("ascii").decode("unicode_escape")))
        except Exception:
            pass
    return extras


def _in_clinical(start: int, end: int, clinical_index: list[tuple[int, int, str]]) -> tuple[bool, str | None]:
    for cs, ce, kind in clinical_index:
        if cs <= start and end <= ce:
            return True, kind
    return False, None


# Keys/headers whose numbers are clinical or financial values, never identifiers. Identifier
# keys (uhid, phone, aadhaar, ...) are deliberately absent, so their values are still checked.
_XML_STRUCTURAL_ATTRS = frozenset({
    "version", "encoding", "standalone", "root", "codesystem", "codesystemname", "codesystemversion",
    "classcode", "moodcode", "typecode", "determinercode", "templateid", "type", "use", "mediatype",
    "representation", "nullflavor",
})
_NUMERIC_CLINICAL_KEYS = frozenset({
    "value", "result", "results", "count", "ref_range", "reference_range", "range", "normal_range",
    "amount", "amount_inr", "total", "bill", "bill_amount", "claim_amount", "approved_amount",
    "package_rate", "rate", "cost", "price", "dose", "strength", "quantity", "qty", "platelets",
    "platelet_count", "wbc", "tlc", "rbc", "hb", "units",
})
_JSON_KEY_BEFORE_RE = re.compile(r'"([^"\n]{1,40})"\s*:\s*"?[^"\n,{}\[\]]*$')


def _field_key_at(buf: str, pos: int, csv_header_end: int, out_path: str) -> str | None:
    """Normalized JSON key or CSV column header owning offset `pos` (None if unknown)."""
    if csv_header_end != -1:
        line_start = buf.rfind("\n", 0, pos) + 1
        if line_start <= csv_header_end:
            return None
        delim = "\t" if out_path.lower().endswith(".tsv") else ","
        col, inq = 0, False
        for ch in buf[line_start:pos]:
            if ch == '"':
                inq = not inq
            elif ch == delim and not inq:
                col += 1
        hdr = next(csv.reader(io.StringIO(buf[:csv_header_end]), delimiter=delim), [])
        return hdr[col].strip().strip("[]").casefold().replace(" ", "_") if col < len(hdr) else None
    m = _JSON_KEY_BEFORE_RE.search(buf[max(0, pos - 80):pos])
    return m.group(1).strip().strip("[]").casefold().replace(" ", "_") if m else None


def _is_valid_b3_value(val: str, lex: _StaticLexicons) -> bool:
    v = val.strip().strip('"\'')
    # A value ends where the next "Label:" (or a wide gap) begins: "UHID: [MRN]  IPD No: [MRN]"
    v = re.split(r"\s{2,}|\s+(?=[A-Za-zऀ-ॿ][A-Za-zऀ-ॿ ./]{0,20}:)", v)[0].strip()
    if not v or _PLACEHOLDER_RE.match(v) or _PSN_OR_ID_RE.match(v):
        return True
    stripped = _PSN_OR_ID_INLINE_RE.sub("", _ANY_PLACEHOLDER_RE.sub(" ", v))
    stripped = re.sub(r"\b[SDWC]\s*/\s*O\b", " ", stripped, flags=re.I)  # kinship markers
    if not any(ch.isalnum() for ch in stripped):
        return True
    # Leftover issuer/code prefixes of a redacted ID ("IPD[MRN]", "SNJ/[MRN]") carry no identity
    if all(re.fullmatch(r"[A-Z]{2,5}", t) for t in re.findall(r"[A-Za-z0-9]+", stripped)):
        return True
    if _ISO_DATE_RE.fullmatch(v) or _NUM_DATE_RE.fullmatch(v):
        return True
    if re.fullmatch(r"\d{1,3}-\d{1,3}|85\+|Adult|Child|Senior", v):
        return True
    low = v.casefold()
    if low in lex.districts or low in lex.states:
        return True
    return False


def _scan_single_buffer(
    buf: str,
    out_path: str,
    clinical_index: list[tuple[int, int, str]],
    lex: _StaticLexicons,
    policy: dict[str, object],
    promoted_labels: set[str],
    is_decoded_extra: bool = False,
) -> list[Hit]:
    hits: list[Hit] = []
    norm_buf = normalize_for_scan(buf)
    res_cfg = policy.get("residual")
    res_dict = res_cfg if isinstance(res_cfg, dict) else {}
    min_digs = int(res_dict.get("b2_min_digits", 6))
    long_digs = int(res_dict.get("b2_long_digits", 10))
    all_labels = _SEED_PII_LABELS | {lbl.strip().casefold() for lbl in promoted_labels}

    json_key_spans: list[tuple[int, int]] = []
    xml_markup_spans: list[tuple[int, int]] = []  # tag/attr names + structural attrs (incl. OIDs)
    if not is_decoded_extra and norm_buf.lstrip().startswith(("{", "[")):
        for km in re.finditer(r'"([^"\n]{1,64})"\s*:', norm_buf):
            json_key_spans.append((km.start(1), km.end(1)))
    if not is_decoded_extra and (out_path.lower().endswith(".xml") or norm_buf.lstrip().startswith(("<?xml", "<!--"))):
        # XML markup (tag and attribute NAMES, declarations, structural OID/namespace values)
        # is schema like a JSON key; quoted attribute values inside tags stay content.
        for tm in re.finditer(r"<[^<>]*>", norm_buf):
            cur = tm.start()
            for am in re.finditer(r"""([A-Za-z_:][-A-Za-z0-9_:.]*)\s*=\s*(["'])(.*?)\2""", tm.group(0)):
                vs = tm.start() + am.start(3)
                ve = tm.start() + am.end(3)
                aname = am.group(1).casefold()
                if aname.startswith("xmlns") or aname in _XML_STRUCTURAL_ATTRS:
                    continue  # whole attribute (name + value) is markup
                json_key_spans.append((cur, vs))
                xml_markup_spans.append((cur, vs))
                cur = ve
            json_key_spans.append((cur, tm.end()))
            xml_markup_spans.append((cur, tm.end()))

    csv_header_end = -1
    if not is_decoded_extra and out_path.lower().endswith((".csv", ".tsv")):
        nl = norm_buf.find("\n")
        csv_header_end = nl if nl != -1 else len(norm_buf)

    json_key_spans.sort()
    _key_starts = [ks for ks, _ in json_key_spans]

    def _is_key_or_header(s: int, e: int) -> str | None:
        if csv_header_end != -1 and e <= csv_header_end:
            return "header"
        i = bisect.bisect_right(_key_starts, s) - 1
        for j in range(i, max(-1, i - 4), -1):  # key spans are disjoint; check a few neighbours
            ks, ke = json_key_spans[j]
            if ks <= s and e <= ke:
                return "key"
        return None

    exempt_num_spans: list[tuple[int, int]] = []
    for pat in (_UUID_RE, _ISO_DATE_RE, _NUM_DATE_RE, _PSN_OR_ID_INLINE_RE, _SAFE_MEDICAL_EPONYMS_RE):
        for m in pat.finditer(norm_buf):
            exempt_num_spans.append((m.start(), m.end()))

    # B4: typed-ID shapes
    for _id_typ, rx in _B4_PATTERNS:
        for m in rx.finditer(norm_buf):
            s, e = m.start(), m.end()
            if any(xs <= s and e <= xe for xs, xe in exempt_num_spans):
                continue
            in_clin, c_kind = _in_clinical(s, e, clinical_index) if not is_decoded_extra else (False, None)
            if in_clin and c_kind in ("lab", "code", "dose"):
                continue
            hits.append(Hit(out_path=out_path, channel="content", start=s, end=e, layer="B", check="B4"))

    # B2: digit runs >= 6. Exempt spans (dates, UUIDs, PSNs) are masked with NUL first,
    # because whitespace counts as a separator: "17/06/2026 22:00" would otherwise form
    # the group "17/06/2026 22", which reaches past the date and escapes the exemption.
    # Masking keeps offsets and cannot join digits across the masked text.
    b2_buf = list(norm_buf)
    for xs, xe in exempt_num_spans:
        b2_buf[xs:xe] = "\x00" * (xe - xs)
    for m in _DIGIT_GROUP_RE.finditer("".join(b2_buf)):
        s, e = m.start(), m.end()
        if any(xs <= s and e <= xe for xs, xe in xml_markup_spans):
            continue  # XML structure (namespace/OID), not content
        if any(xs <= s and e <= xe for xs, xe in exempt_num_spans):
            continue
        digs = "".join(c for c in m.group(0) if c.isdigit())
        n_digs = len(digs)
        if n_digs < min_digs:
            continue
        in_clin, c_kind = _in_clinical(s, e, clinical_index) if not is_decoded_extra else (False, None)
        before = norm_buf[max(0, s - 24) : s]
        after = norm_buf[e : min(len(norm_buf), e + 16)]
        is_clinical_num = bool(_B2_EXEMPT_BEFORE_RE.search(before) or _B2_EXEMPT_AFTER_RE.search(after))
        if not is_clinical_num and not is_decoded_extra:
            fkey = _field_key_at(norm_buf, s, csv_header_end, out_path)
            if fkey is not None and fkey in _NUMERIC_CLINICAL_KEYS:
                is_clinical_num = True
        if n_digs >= long_digs:
            if in_clin and c_kind in ("lab", "code"):
                continue
            if is_clinical_num and ("rs" in before.lower() or "inr" in before.lower() or "pkg" in before.lower()):
                continue
            hits.append(Hit(out_path=out_path, channel="content", start=s, end=e, layer="B", check="B2"))
        else:
            if in_clin or is_clinical_num:
                continue
            hits.append(Hit(out_path=out_path, channel="content", start=s, end=e, layer="B", check="B2"))

    # B3: values under seed/promoted labels
    if not is_decoded_extra and norm_buf.lstrip().startswith(("{", "[")):
        for jm in re.finditer(r'"([^"\n]{1,40})"\s*:\s*"([^"\n]{1,120})"', norm_buf):
            lbl_norm = jm.group(1).strip().strip("[]").casefold().replace(" ", "_")
            if lbl_norm in all_labels:
                in_clin, _ck = _in_clinical(jm.start(2), jm.end(2), clinical_index)
                if in_clin:
                    continue
                if not _is_valid_b3_value(jm.group(2), lex):
                    hits.append(
                        Hit(
                            out_path=out_path,
                            channel="content",
                            start=jm.start(2),
                            end=jm.end(2),
                            layer="B",
                            check="B3",
                        )
                    )
    for km in _B3_KV_RE.finditer(norm_buf):
        lbl_norm = km.group("label").strip().strip("[]").casefold().replace(" ", "_")
        in_clin, _ck = _in_clinical(km.start("val"), km.end("val"), clinical_index) if not is_decoded_extra else (False, None)
        if not in_clin and lbl_norm in all_labels and not _is_valid_b3_value(km.group("val"), lex):
            hits.append(
                Hit(
                    out_path=out_path,
                    channel="content",
                    start=km.start("val"),
                    end=km.end("val"),
                    layer="B",
                    check="B3",
                )
            )
    if csv_header_end != -1:
        try:
            delim = "\t" if out_path.lower().endswith(".tsv") else ","
            rows = list(csv.reader(io.StringIO(norm_buf), delimiter=delim))
            if len(rows) > 1:
                hdr = [h.strip().strip("[]").casefold().replace(" ", "_") for h in rows[0]]
                for row in rows[1:]:
                    for c_idx, cell in enumerate(row):
                        if c_idx < len(hdr) and hdr[c_idx] in all_labels:
                            if not _is_valid_b3_value(cell, lex):
                                pos = norm_buf.find(cell, csv_header_end)
                                s_off = pos if pos != -1 else 0
                                hits.append(
                                    Hit(
                                        out_path=out_path,
                                        channel="content",
                                        start=s_off,
                                        end=s_off + len(cell),
                                        layer="B",
                                        check="B3",
                                    )
                                )
        except csv.Error:
            pass

    # B1: Default-deny on Title-Case / ALL-CAPS Latin & Devanagari words + lowercase gazetteer hits
    for m in _LATIN_TOK_RE.finditer(norm_buf):
        s, e = m.start(), m.end()
        tok_clean = m.group(0).strip(".'-")
        if len(tok_clean) < 2:
            continue
        if any(xs <= s and e <= xe for xs, xe in exempt_num_spans):
            continue
        if s > 0 and norm_buf[s - 1] == "[" and e < len(norm_buf) and norm_buf[e] == "]":
            continue
        if not is_decoded_extra:
            in_clin, _ck = _in_clinical(s, e, clinical_index)
            if in_clin:
                continue

        low = tok_clean.casefold()
        kh = _is_key_or_header(s, e)
        if kh is not None:
            lbl_norm = low.replace(" ", "_")
            if low in lex.names or lbl_norm in promoted_labels:
                hits.append(Hit(out_path=out_path, channel=kh, start=s, end=e, layer="B", check="B1"))  # type: ignore[arg-type]
            continue

        is_title_or_caps = tok_clean[0].isupper()
        is_gazetteer_lower = tok_clean.islower() and low in lex.names
        if not (is_title_or_caps or is_gazetteer_lower):
            continue
        if tok_clean.upper() in _SAFE_ABBR and tok_clean.isupper():
            continue
        if low in lex.safe_words or ((low in lex.districts or low in lex.states) and low not in lex.names):
            continue  # a place that is also a person name (Sagar, Krishna) is not cleared
        if any(low.endswith(suf) for suf in _CLINICAL_SUFFIXES):
            continue
        if (s > 0 and norm_buf[s - 1].isdigit()) or (e < len(norm_buf) and norm_buf[e].isdigit()):
            continue  # part of an alphanumeric code (SG050A, HbA1c)
        hits.append(Hit(out_path=out_path, channel="content", start=s, end=e, layer="B", check="B1"))

    for m in _DEVA_TOK_RE.finditer(norm_buf):
        s, e = m.start(), m.end()
        tok = m.group(0)
        if not is_decoded_extra:
            in_clin, _ck = _in_clinical(s, e, clinical_index)
            if in_clin:
                continue
        kh = _is_key_or_header(s, e)
        if kh is not None:
            if tok in lex.names:
                hits.append(Hit(out_path=out_path, channel=kh, start=s, end=e, layer="B", check="B1"))  # type: ignore[arg-type]
            continue
        if tok in lex.safe_words or tok in lex.districts or tok in lex.states:
            continue
        hits.append(Hit(out_path=out_path, channel="content", start=s, end=e, layer="B", check="B1"))

    return hits


def scan(
    text: str,
    out_path: str,
    clinical_index: list[tuple[int, int, str]] | None = None,
    policy: dict[str, object] | None = None,
    promoted_labels: set[str] | None = None,
    channel_spans: list[tuple[int, int, str]] | None = None,
) -> list[Hit]:
    """Independent default-deny Layer B re-scan on decoded bytes read back from disk."""
    _ = channel_spans
    lex = _get_lexicons()
    pol = policy or {}
    c_idx = clinical_index or []
    p_labels = promoted_labels or set()

    hits = _scan_single_buffer(text, out_path, c_idx, lex, pol, p_labels, is_decoded_extra=False)

    norm_path = normalize_for_scan(out_path)
    for part in re.split(r"[_.\-/\s]+", norm_path):
        if len(part) < 2:
            continue
        low = part.casefold()
        if low in (
            "sanitized", "doc", "row", "csv", "tsv", "json", "txt", "fhir", "bundle",
            "opd", "lab", "billing", "discharge", "notes", "prescriptions", "registration",
            "clean", "misc",
        ):
            continue
        if low in lex.names or (
            part[0].isupper() and low not in lex.safe_words and low not in lex.districts and low not in lex.states
        ):
            pos = norm_path.find(part)
            hits.append(Hit(out_path=out_path, channel="path", start=max(0, pos), end=max(0, pos) + len(part), layer="B", check="B1"))
        if re.fullmatch(r"\d{6,}", part):
            pos = norm_path.find(part)
            hits.append(Hit(out_path=out_path, channel="path", start=max(0, pos), end=max(0, pos) + len(part), layer="B", check="B2"))

    for es, ee, extra in _decode_hidden_payloads(text):
        extra_hits = _scan_single_buffer(extra, out_path, [], lex, pol, p_labels, is_decoded_extra=True)
        for eh in extra_hits:
            hits.append(Hit(out_path=out_path, channel="content", start=es, end=ee, layer="B", check=eh.check))

    seen: set[tuple[str, str, int, int, str]] = set()
    unique: list[Hit] = []
    for h in sorted(hits, key=lambda x: (x.channel, x.start, x.end, x.check)):
        key = (h.out_path, h.channel, h.start, h.end, h.check)
        if key not in seen:
            seen.add(key)
            unique.append(h)
    return unique
