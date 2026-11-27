"""Layer A scorer-mirror residual scanner (Plan C V8 / C13 / FR-R7).

Compiles harvested identifiers from pipeline ``Span`` objects, ``Evidence``
objects, and validated V7 ``FilenameSignal`` objects into a redacted,
run-scoped ``LeakKeys`` index. Scans rendered output surfaces (content, path,
JSON keys, CSV headers) and bounded decoded wrappers (base64, hex, ``\\u``
escapes, HTML entities, percent escapes) using Layer A normalization, enforcing
the narrowed FR-R7 exemption for clinical and ``doctor_registry`` spans.
"""

from __future__ import annotations

import base64
import binascii
import html
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Mapping

from kavachmed.evidence import Evidence, RecKey, Record, Span
from kavachmed.indic import NameToken, normalize_name
from kavachmed.linkage.filename import FilenameSignal

Channel = Literal["content", "path", "key", "header"]

EXEMPT_KINDS: frozenset[str] = frozenset(
    {
        "drug",
        "dose",
        "lab",
        "vital",
        "code",
        "eponym",
        "ward",
        "department",
        "hospital",
        "district",
        "history",
        "clinical",
        "doctor_registry",
    }
)

_NAME_SPAN_TYPES: frozenset[str] = frozenset(
    {"PATIENT_NAME", "RELATIVE_NAME", "PERSON", "NAME"}
)
_DOB_SPAN_TYPES: frozenset[str] = frozenset({"DOB", "DATE_OF_BIRTH", "BIRTH_DATE"})
_EXACT_SPAN_TYPES: frozenset[str] = frozenset(
    {
        "EMAIL",
        "PAN",
        "VEHICLE_PLATE",
        "HOUSE",
        "LOCALITY",
        "LANDMARK",
        "VILLAGE",
        "STREET",
        "ADDRESS",
        "CANARY",
        "CANARY_MARKER",
        "ADM_KEY",
    }
)
_DIGIT_SPAN_TYPES: frozenset[str] = frozenset(
    {
        "PHONE",
        "MOBILE",
        "AADHAAR",
        "ABHA",
        "MRN",
        "UHID",
        "IPD",
        "OPD",
        "LAB_ID",
        "REG_NO",
        "ID",
    }
)
_NONPATIENT_ROLES: frozenset[str] = frozenset(
    {"doctor", "clinician", "staff", "facility", "nonpatient"}
)

_ZERO_WIDTH_CHARS: frozenset[str] = frozenset(
    {"\u200b", "\u200c", "\u200d", "\u200e", "\u200f", "\ufeff", "\u2060", "\u00ad"}
)

# Explicit confusable / homoglyph fold table (Cyrillic, Greek, fullwidth Latin).
_HOMOGLYPH_MAP: dict[str, str] = {
    "\u0410": "A", "\u0430": "a",  # Cyrillic А/а
    "\u0412": "B", "\u0432": "b",  # Cyrillic В/в
    "\u0415": "E", "\u0435": "e",  # Cyrillic Е/е
    "\u0401": "E", "\u0451": "e",  # Cyrillic Ё/ё
    "\u041a": "K", "\u043a": "k",  # Cyrillic К/к
    "\u041c": "M", "\u043c": "m",  # Cyrillic М/м
    "\u041d": "H", "\u043d": "h",  # Cyrillic Н/н
    "\u041e": "O", "\u043e": "o",  # Cyrillic О/о
    "\u0420": "P", "\u0440": "p",  # Cyrillic Р/р
    "\u0421": "C", "\u0441": "c",  # Cyrillic С/с
    "\u0422": "T", "\u0442": "t",  # Cyrillic Т/т
    "\u0423": "Y", "\u0443": "y",  # Cyrillic У/у
    "\u0425": "X", "\u0445": "x",  # Cyrillic Х/х
    "\u0406": "I", "\u0456": "i",  # Cyrillic І/і
    "\u0408": "J", "\u0458": "j",  # Cyrillic Ј/ј
    "\u0405": "S", "\u0455": "s",  # Cyrillic Ѕ/ѕ
    "\u0391": "A", "\u03b1": "a",  # Greek Α/α
    "\u0392": "B", "\u03b2": "b",  # Greek Β/β
    "\u0395": "E", "\u03b5": "e",  # Greek Ε/ε
    "\u0396": "Z", "\u03b6": "z",  # Greek Ζ/ζ
    "\u0397": "H", "\u03b7": "h",  # Greek Η/η
    "\u0399": "I", "\u03b9": "i",  # Greek Ι/ι
    "\u039a": "K", "\u03ba": "k",  # Greek Κ/κ
    "\u039c": "M", "\u03bc": "m",  # Greek Μ/μ
    "\u039d": "N", "\u03bd": "n",  # Greek Ν/ν
    "\u039f": "O", "\u03bf": "o",  # Greek Ο/ο
    "\u03a1": "P", "\u03c1": "p",  # Greek Ρ/ρ
    "\u03a4": "T", "\u03c4": "t",  # Greek Τ/τ
    "\u03a7": "X", "\u03c7": "x",  # Greek Χ/χ
}

_DIGIT_SEPARATORS: frozenset[str] = frozenset({"-", " ", ".", "/", "_", ":"})
_MIN_DIGIT_CORE_LEN = 5

_B64_RE = re.compile(r"(?<![A-Za-z0-9+/_-])[A-Za-z0-9+/]{16,}={0,2}(?![A-Za-z0-9+/=_-])")
_HEX_RE = re.compile(r"(?<![0-9A-Fa-f])(?:0[xX])?([0-9A-Fa-f]{12,})(?![0-9A-Fa-f])")
_UNICODE_ESC_RE = re.compile(r"(?:\\u[0-9A-Fa-f]{4}|\\U[0-9A-Fa-f]{8}){2,}")
_HTML_ENTITY_RE = re.compile(r"(?:&(?:#\d{2,6}|#x[0-9A-Fa-f]{2,5}|[A-Za-z][A-Za-z0-9]{1,10});)")
_PERCENT_ESC_RE = re.compile(r"(?:%[0-9A-Fa-f]{2}){2,}")


class ResidualScanError(RuntimeError):
    """Text-free error raised when a required residual scan cannot be verified."""

    def __init__(self, code: str) -> None:
        super().__init__(f"residual_a:{code}")
        self.code = code


@dataclass(frozen=True)
class Hit:
    """Residual finding — NEVER carries matched text."""

    out_path: str
    channel: Channel
    start: int
    end: int
    layer: Literal["A", "B"] = "A"
    check: Literal["A", "B1", "B2", "B3", "B4"] = "A"

    def __repr__(self) -> str:
        # Suppress out_path in repr when the path itself may contain a leak.
        return (
            f"Hit(channel={self.channel!r}, start={self.start}, "
            f"end={self.end}, layer={self.layer!r}, check={self.check!r})"
        )


@dataclass(frozen=True)
class ExemptSpan:
    """Output-coordinate protected span for narrowed FR-R7 exemption."""

    out_path: str
    channel: Channel
    start: int
    end: int
    exempt_kind: str | None = None
    protected: bool = True
    decoded_start: int | None = None
    decoded_end: int | None = None


@dataclass(frozen=True)
class ScanOutcome:
    """Result of scanning a rendered file including verification status."""

    hits: tuple[Hit, ...]
    verified: bool = True
    reason: str | None = None

    def __repr__(self) -> str:
        return (
            f"ScanOutcome(hits={len(self.hits)}, "
            f"verified={self.verified}, reason={self.reason!r})"
        )


@dataclass(frozen=True)
class LeakKeys:
    """Run-scoped, redacted index of harvested identifiers for Layer A."""

    name_tokens: tuple[str, ...] = ()
    digit_cores: tuple[str, ...] = ()
    exact_strings: tuple[str, ...] = ()
    dobs: tuple[str, ...] = ()

    def __repr__(self) -> str:
        return (
            f"LeakKeys(<redacted: names={len(self.name_tokens)}, "
            f"digits={len(self.digit_cores)}, "
            f"exact={len(self.exact_strings)}, dobs={len(self.dobs)}>)"
        )

    __str__ = __repr__


def _bump(diagnostics: dict[str, int] | None, key: str, delta: int = 1) -> None:
    if diagnostics is not None:
        diagnostics[key] = diagnostics.get(key, 0) + delta


def _is_devanagari_char(ch: str) -> bool:
    cp = ord(ch)
    return 0x0900 <= cp <= 0x097F or 0xA8E0 <= cp <= 0xA8FF


def _fold_char(ch: str) -> str:
    if ch in _ZERO_WIDTH_CHARS:
        return ""
    if ch in _HOMOGLYPH_MAP:
        ch = _HOMOGLYPH_MAP[ch]
    if unicodedata.category(ch) == "Nd":
        try:
            return str(unicodedata.digit(ch))
        except ValueError:
            pass
    return ch


def normalize_with_spans(text: str) -> tuple[str, list[tuple[int, int]]]:
    """Normalize text per Layer A rules and map each output char to (orig_start, orig_end)."""
    out_chars: list[str] = []
    out_spans: list[tuple[int, int]] = []
    prev_was_space = False

    for orig_idx, raw_ch in enumerate(text):
        orig_end = orig_idx + 1
        nfkc = unicodedata.normalize("NFKC", raw_ch)
        for ch in nfkc:
            folded = _fold_char(ch)
            if not folded:
                continue
            # Decompose to strip Latin accents while retaining Devanagari marks.
            if _is_devanagari_char(folded):
                pieces = [folded]
            else:
                decomposed = unicodedata.normalize("NFD", folded)
                pieces = [
                    p
                    for p in decomposed
                    if unicodedata.category(p) != "Mn" or _is_devanagari_char(p)
                ]
            for piece in pieces:
                for c in piece.casefold():
                    if c.isspace():
                        if not out_chars or prev_was_space:
                            continue
                        out_chars.append(" ")
                        out_spans.append((orig_idx, orig_end))
                        prev_was_space = True
                    else:
                        out_chars.append(c)
                        out_spans.append((orig_idx, orig_end))
                        prev_was_space = False

    while out_chars and out_chars[-1] == " ":
        out_chars.pop()
        out_spans.pop()

    return "".join(out_chars), out_spans


def normalize_text(text: str) -> str:
    """Normalize text per Layer A rules."""
    normed, _ = normalize_with_spans(text)
    return normed


def _is_word_char(ch: str) -> bool:
    return ch.isalnum() or _is_devanagari_char(ch)


_WORD_RUN_RE = re.compile(r"(?:[^\W_]|[\u0900-\u097F])+")


def _has_token_boundaries(norm_text: str, start: int, end: int) -> bool:
    left_ok = start == 0 or not _is_word_char(norm_text[start - 1])
    right_ok = end == len(norm_text) or not _is_word_char(norm_text[end])
    return left_ok and right_ok


def _build_field_text_lookup(
    records: Iterable[Record] | None,
    field_texts: Mapping[tuple[RecKey, int], str] | Mapping[RecKey, Mapping[int, str]] | None,
) -> dict[tuple[RecKey, int], str]:
    lookup: dict[tuple[RecKey, int], str] = {}
    if records is not None:
        for rec in records:
            for fld in rec.fields:
                lookup[(rec.key, fld.idx)] = fld.text
    if field_texts is not None:
        for k, v in field_texts.items():
            if isinstance(v, str):
                lookup[k] = v  # type: ignore[index]
            elif isinstance(v, Mapping):
                for f_idx, txt in v.items():
                    lookup[(k, f_idx)] = txt  # type: ignore[index]
    return lookup


def _resolve_span_text(
    span: Span,
    field_lookup: dict[tuple[RecKey, int], str],
    diagnostics: dict[str, int] | None,
) -> str | None:
    text = field_lookup.get((span.key, span.field_idx))
    if text is None:
        text = getattr(span, "text", None)
    if text is None:
        _bump(diagnostics, "skipped_unresolved_span")
        return None
    if span.start < 0 or span.end <= span.start or span.end > len(text):
        _bump(diagnostics, "skipped_invalid_span")
        return None
    return text[span.start : span.end]


def _add_name_tokens(
    raw_name: str,
    names_set: set[str],
    diagnostics: dict[str, int] | None,
) -> None:
    for tok in normalize_name(raw_name):
        if tok.kind in ("generic", "honorific"):
            _bump(diagnostics, "skipped_generic_name")
            continue
        if tok.kind == "initial":
            _bump(diagnostics, "skipped_initial_name")
            continue
        for candidate in (tok.surface, tok.latin):
            if not candidate:
                continue
            normed = normalize_text(candidate)
            if len(normed) >= 2:
                names_set.add(normed)


def _add_digit_core(
    raw_val: str,
    digit_set: set[str],
    exact_set: set[str],
) -> None:
    digits = "".join(ch for ch in normalize_text(raw_val) if ch.isdigit())
    if len(digits) >= _MIN_DIGIT_CORE_LEN:
        digit_set.add(digits)
        if len(digits) > 10:
            digit_set.add(digits[-10:])
    else:
        normed = normalize_text(raw_val)
        if len(normed) >= 4 and any(c.isalpha() for c in normed) and any(c.isdigit() for c in normed):
            exact_set.add(normed)


def _add_dob_variants(raw_dob: str, dob_set: set[str]) -> None:
    normed = normalize_text(raw_dob)
    if not normed:
        return
    dob_set.add(normed)
    digits = "".join(ch for ch in normed if ch.isdigit())
    if len(digits) == 8:
        dob_set.add(digits)


def compile_keys(
    spans: Iterable[Span],
    evidences: Iterable[Evidence],
    *,
    records: Iterable[Record] | None = None,
    filename_signals: Iterable[FilenameSignal] | None = None,
    field_texts: Mapping[tuple[RecKey, int], str] | Mapping[RecKey, Mapping[int, str]] | None = None,
    diagnostics: dict[str, int] | None = None,
) -> LeakKeys:
    """Compile a deduplicated, redacted ``LeakKeys`` index from harvested PII."""
    field_lookup = _build_field_text_lookup(records, field_texts)
    names_set: set[str] = set()
    digit_set: set[str] = set()
    exact_set: set[str] = set()
    dob_set: set[str] = set()

    # 1. Harvest from Evidence objects (normalized names, kin, IDs, phones, geo, adm_key).
    for ev in evidences:
        for tok in ev.names:
            if isinstance(tok, NameToken):
                if tok.kind == "identifying":
                    for c in (tok.surface, tok.latin):
                        if c:
                            n = normalize_text(c)
                            if len(n) >= 2:
                                names_set.add(n)
                elif tok.kind in ("generic", "honorific"):
                    _bump(diagnostics, "skipped_generic_name")
        for tok in ev.kin:
            if isinstance(tok, NameToken):
                if tok.kind == "identifying":
                    for c in (tok.surface, tok.latin):
                        if c:
                            n = normalize_text(c)
                            if len(n) >= 2:
                                names_set.add(n)
                elif tok.kind in ("generic", "honorific"):
                    _bump(diagnostics, "skipped_generic_name")

        for vals in ev.ids.values():
            for v in vals:
                _add_digit_core(v, digit_set, exact_set)
                normed = normalize_text(v)
                if len(normed) >= 5 and any(c.isalpha() for c in normed):
                    exact_set.add(normed)

        for ph in ev.phones:
            _add_digit_core(ph, digit_set, exact_set)

        if ev.adm_key:
            normed_adm = normalize_text(ev.adm_key)
            if len(normed_adm) >= 4:
                exact_set.add(normed_adm)

        for geo_key in ("house", "locality", "landmark", "village", "street"):
            geo_val = ev.geo.get(geo_key)
            if isinstance(geo_val, str):
                normed_geo = normalize_text(geo_val)
                if len(normed_geo) >= 3:
                    exact_set.add(normed_geo)

    # 2. Harvest from Span objects resolved against original Field.text.
    for sp in spans:
        if sp.protected and sp.exempt_kind in EXEMPT_KINDS:
            continue
        if sp.role in _NONPATIENT_ROLES and sp.typ in _NAME_SPAN_TYPES:
            continue
        raw_val = _resolve_span_text(sp, field_lookup, diagnostics)
        if not raw_val:
            continue
        typ_u = sp.typ.upper()
        if typ_u in _NAME_SPAN_TYPES:
            _add_name_tokens(raw_val, names_set, diagnostics)
        elif typ_u in _DOB_SPAN_TYPES:
            _add_dob_variants(raw_val, dob_set)
        elif typ_u == "DATE":
            # Exclude clinical dates from DOB key set.
            _bump(diagnostics, "skipped_clinical_date")
        elif typ_u in _DIGIT_SPAN_TYPES:
            _add_digit_core(raw_val, digit_set, exact_set)
            normed = normalize_text(raw_val)
            if len(normed) >= 5 and any(c.isalpha() for c in normed):
                exact_set.add(normed)
        elif typ_u in _EXACT_SPAN_TYPES:
            normed = normalize_text(raw_val)
            if len(normed) >= 3:
                exact_set.add(normed)

    # 3. Harvest from validated V7 filename signals even if insufficient for linkage.
    if filename_signals is not None:
        for sig in filename_signals:
            if sig.role.strip().lower() in _NONPATIENT_ROLES:
                _bump(diagnostics, "skipped_nonpatient_filename")
                continue
            if not sig.value or not sig.value.strip():
                continue
            typ_u = sig.typ.strip().upper()
            if typ_u in _NAME_SPAN_TYPES:
                _add_name_tokens(sig.value, names_set, diagnostics)
            elif typ_u in _DOB_SPAN_TYPES:
                _add_dob_variants(sig.value, dob_set)
            elif typ_u in _DIGIT_SPAN_TYPES:
                _add_digit_core(sig.value, digit_set, exact_set)
            else:
                normed = normalize_text(sig.value)
                if len(normed) >= 4:
                    exact_set.add(normed)

    _drop = _non_name_words()
    dropped = {n for n in names_set if n in _drop}
    if dropped:
        _bump(diagnostics, "skipped_non_name_word", len(dropped))
        names_set -= dropped

    keys = LeakKeys(
        name_tokens=tuple(sorted(names_set)),
        digit_cores=tuple(sorted(digit_set)),
        exact_strings=tuple(sorted(exact_set)),
        dobs=tuple(sorted(dob_set)),
    )
    _bump(diagnostics, "compiled_names", len(keys.name_tokens))
    _bump(diagnostics, "compiled_digit_cores", len(keys.digit_cores))
    _bump(diagnostics, "compiled_exact", len(keys.exact_strings))
    _bump(diagnostics, "compiled_dobs", len(keys.dobs))
    return keys



# Words that are never identifying on their own: the union of B's clinical/common safe
# list and D's name-subtracted safe set (both exclude every name-gazetteer token), plus
# record-structure labels. Harvested name keys in this set come from mis-tagged labels
# ("Patient Name") or filename boilerplate and would flag every placeholder and JSON key.
_STRUCTURE_WORDS = frozenset({
    "name", "patient", "pt", "relative", "father", "mother", "husband", "wife", "guardian",
    "attendant", "informant", "mobile", "phone", "contact", "address", "age", "sex", "gender",
    "dob", "uhid", "mrn", "ipd", "opd", "visit", "results", "report", "nursing", "forward",
    "whatsapp", "redacted", "relative_name", "patient_name",
    # FHIR / JSON schema words that also occur as keys
    "city", "state", "district", "country", "line", "text", "display", "given", "family",
    "postalcode", "use", "system", "value", "export", "log", "lab", "labs",
})
_NON_NAME_WORDS: frozenset[str] | None = None


def _non_name_words() -> frozenset[str]:
    global _NON_NAME_WORDS
    if _NON_NAME_WORDS is None:
        import gzip
        from pathlib import Path
        base = Path(__file__).resolve().parent.parent / "config" / "lexicons" / "safe"
        words: set[str] = set(_STRUCTURE_WORDS)
        for f in (base / "safe_words.txt", base / "safe_words.txt.gz", base.parent / "clinical" / "clinical_vocab.txt"):
            try:
                raw = gzip.decompress(f.read_bytes()).decode("utf-8") if f.suffix == ".gz" else f.read_text(encoding="utf-8")
            except OSError:
                continue
            for line in raw.splitlines():
                if line.strip().startswith("#"):
                    continue
                for w in line.split():
                    words.add(normalize_text(w))
        _NON_NAME_WORDS = frozenset(words)
    return _NON_NAME_WORDS


# Our own release tokens: typed placeholders and keyed pseudonyms are never leaks.
_RELEASE_TOKEN_RE = re.compile(r"\[[a-z_]{2,24}\]|psn-[0-9a-z]{8,}", re.IGNORECASE)

def _percent_decode(text: str) -> str:
    """Strict %XX decoder (UTF-8); stdlib urllib is banned by the self-audit network rule."""
    out = bytearray()
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "%" and i + 2 < len(text) and re.fullmatch(r"[0-9A-Fa-f]{2}", text[i + 1:i + 3]):
            out.append(int(text[i + 1:i + 3], 16))
            i += 3
        else:
            out.extend(ch.encode("utf-8"))
            i += 1
    return out.decode("utf-8", errors="strict")


def _is_exempt(
    out_path: str,
    channel: Channel,
    start: int,
    end: int,
    exempt_index: Iterable[ExemptSpan] | None,
    *,
    is_decoded: bool = False,
    decoded_start: int | None = None,
    decoded_end: int | None = None,
) -> bool:
    """Narrowed FR-R7 exemption: only for name matches inside one exempt span."""
    if not exempt_index:
        return False
    for ex in exempt_index:
        if not ex.protected or ex.exempt_kind not in EXEMPT_KINDS:
            continue
        if ex.out_path != out_path or ex.channel != channel:
            continue
        if is_decoded:
            if (
                ex.decoded_start is not None
                and ex.decoded_end is not None
                and decoded_start is not None
                and decoded_end is not None
                and ex.start <= start
                and end <= ex.end
                and ex.decoded_start <= decoded_start
                and decoded_end <= ex.decoded_end
            ):
                return True
        else:
            if ex.start <= start and end <= ex.end:
                return True
    return False


def _scan_normalized_surface(
    raw_text: str,
    out_path: str,
    channel: Channel,
    keys: LeakKeys,
    exempt_index: Iterable[ExemptSpan] | None,
    *,
    is_decoded: bool = False,
    wrapper_range: tuple[int, int] | None = None,
) -> list[Hit]:
    """Scan one surface (or decoded view) and return raw Hits."""
    if not raw_text:
        return []
    norm_text, spans = normalize_with_spans(raw_text)
    if not norm_text:
        return []
    release_ranges = [(m.start(), m.end()) for m in _RELEASE_TOKEN_RE.finditer(norm_text)]

    hits: list[Hit] = []

    def _emit(
        n_start: int,
        n_end: int,
        *,
        is_name: bool = False,
    ) -> None:
        orig_start = spans[n_start][0]
        orig_end = spans[n_end - 1][1]
        if is_decoded and wrapper_range is not None:
            hit_start, hit_end = wrapper_range
            dec_s, dec_e = orig_start, orig_end
        else:
            hit_start, hit_end = orig_start, orig_end
            dec_s, dec_e = None, None
        if is_name and _is_exempt(
            out_path,
            channel,
            hit_start,
            hit_end,
            exempt_index,
            is_decoded=is_decoded,
            decoded_start=dec_s,
            decoded_end=dec_e,
        ):
            return
        if any(rs <= n_start and n_end <= re_ for rs, re_ in release_ranges):
            return
        hits.append(
            Hit(
                out_path=out_path,
                channel=channel,
                start=hit_start,
                end=hit_end,
                layer="A",
                check="A",
            )
        )

    # Single pass over the text's word runs (O(text)), instead of str.find per key
    # (O(keys x text): 200M find calls at 20k records). Same token-boundary semantics:
    # a word run is a maximal run of _is_word_char characters.
    runs = [(m.start(), m.end()) for m in _WORD_RUN_RE.finditer(norm_text)]
    run_index: dict[str, list[int]] = {}
    for rs, re_ in runs:
        run_index.setdefault(norm_text[rs:re_], []).append(rs)

    def _find_all(needle: str, bounded: bool) -> list[int]:
        """Start offsets of token-bounded occurrences (fallback: plain scan for odd needles)."""
        first = _WORD_RUN_RE.match(needle)
        if bounded and first and first.start() == 0:
            head = needle[: first.end()]
            if first.end() == len(needle):
                return list(run_index.get(head, ()))
            return [
                st for st in run_index.get(head, ())
                if norm_text.startswith(needle, st) and _has_token_boundaries(norm_text, st, st + len(needle))
            ]
        out_pos: list[int] = []
        pos = 0
        while True:
            idx = norm_text.find(needle, pos)
            if idx == -1:
                return out_pos
            if not bounded or _has_token_boundaries(norm_text, idx, idx + len(needle)):
                out_pos.append(idx)
            pos = idx + 1

    # 1. Name tokens (token-bounded, subject to FR-R7 exemption).
    for name_tok in keys.name_tokens:
        for idx in _find_all(name_tok, True):
            _emit(idx, idx + len(name_tok), is_name=True)

    # 2. Exact strings (alphanumeric strings require word boundaries).
    for exact in keys.exact_strings:
        require_bounds = _is_word_char(exact[0]) and _is_word_char(exact[-1])
        for idx in _find_all(exact, require_bounds):
            _emit(idx, idx + len(exact), is_name=False)

    # 3. DOBs (never exempt).
    for dob in keys.dobs:
        for idx in _find_all(dob, True):
            _emit(idx, idx + len(dob), is_name=False)

    # 4. Typed digit cores across allowed separators. A core only matches a WHOLE digit run
    # (separators skipped), so a set lookup per run is equivalent to the old scan.
    if keys.digit_cores:
        core_set = set(keys.digit_cores)
        cur_digits: list[str] = []
        cur_norm_indices: list[int] = []

        def _flush() -> None:
            if len(cur_digits) >= _MIN_DIGIT_CORE_LEN and "".join(cur_digits) in core_set:
                _emit(cur_norm_indices[0], cur_norm_indices[-1] + 1, is_name=False)

        for idx, ch in enumerate(norm_text):
            if ch.isdigit():
                cur_digits.append(ch)
                cur_norm_indices.append(idx)
            elif ch in _DIGIT_SEPARATORS and cur_digits:
                continue
            else:
                _flush()
                cur_digits = []
                cur_norm_indices = []
        _flush()

    hits.sort(key=lambda h: (h.start, h.end, h.channel))
    return hits


def _looks_like_text(raw: bytes) -> str | None:
    try:
        decoded = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return None
    if not decoded:
        return None
    printable = sum(1 for ch in decoded if ch.isprintable() or ch.isspace())
    if printable / len(decoded) < 0.85:
        return None
    return decoded


def _extract_decoded_views(
    text: str,
    *,
    max_decode_bytes: int,
    max_wrappers: int,
    diagnostics: dict[str, int] | None,
) -> tuple[list[tuple[int, int, str]], bool]:
    """Extract bounded decoded views from base64, hex, \\u, HTML, and %XX wrappers."""
    views: list[tuple[int, int, str]] = []
    unverified = False
    total_decoded_bytes = 0
    wrapper_count = 0

    def _record_view(start: int, end: int, decoded_str: str) -> bool:
        nonlocal total_decoded_bytes, wrapper_count, unverified
        wrapper_count += 1
        if wrapper_count > max_wrappers:
            _bump(diagnostics, "over_budget_wrappers")
            unverified = True
            return False
        total_decoded_bytes += len(decoded_str.encode("utf-8", errors="ignore"))
        if total_decoded_bytes > max_decode_bytes:
            _bump(diagnostics, "over_budget_wrappers")
            unverified = True
            return False
        views.append((start, end, decoded_str))
        return True

    # Base64 runs
    for m in _B64_RE.finditer(text):
        tok = m.group(0)
        padded = tok + "=" * (-len(tok) % 4)
        try:
            raw = base64.b64decode(padded, validate=True)
        except (binascii.Error, ValueError):
            _bump(diagnostics, "malformed_wrappers")
            unverified = True
            continue
        decoded_str = _looks_like_text(raw)
        if decoded_str is None:
            _bump(diagnostics, "binary_or_unreadable_wrappers")
            unverified = True
            continue
        if not _record_view(m.start(), m.end(), decoded_str):
            break

    # Hex runs
    for m in _HEX_RE.finditer(text):
        hex_body = m.group(1)
        if len(hex_body) % 2 != 0:
            _bump(diagnostics, "malformed_wrappers")
            unverified = True
            continue
        try:
            raw = bytes.fromhex(hex_body)
        except ValueError:
            _bump(diagnostics, "malformed_wrappers")
            unverified = True
            continue
        decoded_str = _looks_like_text(raw)
        if decoded_str is not None:
            if not _record_view(m.start(), m.end(), decoded_str):
                break

    # Unicode escapes (\uXXXX / \UXXXXXXXX)
    for m in _UNICODE_ESC_RE.finditer(text):
        try:
            decoded_str = m.group(0).encode("ascii").decode("unicode_escape")
        except UnicodeDecodeError:
            _bump(diagnostics, "malformed_wrappers")
            unverified = True
            continue
        if not _record_view(m.start(), m.end(), decoded_str):
            break

    # Percent escapes (%XX)
    for m in _PERCENT_ESC_RE.finditer(text):
        decoded_str = _percent_decode(m.group(0))
        if decoded_str != m.group(0):
            if not _record_view(m.start(), m.end(), decoded_str):
                break

    # HTML entities (group contiguous or line-level entity-bearing spans)
    if _HTML_ENTITY_RE.search(text):
        unescaped = html.unescape(text)
        if unescaped != text:
            # Map entity-bearing regions conservatively to full text or individual entities.
            for m in _HTML_ENTITY_RE.finditer(text):
                # Also decode the surrounding token window so split entities like &#65;rif work.
                w_start = m.start()
                while w_start > 0 and not text[w_start - 1].isspace():
                    w_start -= 1
                w_end = m.end()
                while w_end < len(text) and not text[w_end].isspace():
                    w_end += 1
                snippet = html.unescape(text[w_start:w_end])
                if not _record_view(w_start, w_end, snippet):
                    break

    return views, unverified


def _dedup_hits(hits: list[Hit]) -> tuple[Hit, ...]:
    seen: set[tuple[str, Channel, int, int, str, str]] = set()
    unique: list[Hit] = []
    for h in sorted(hits, key=lambda x: (x.out_path, x.channel, x.start, x.end)):
        k = (h.out_path, h.channel, h.start, h.end, h.layer, h.check)
        if k not in seen:
            seen.add(k)
            unique.append(h)
    return tuple(unique)


def scan_with_status(
    text: str,
    path: str,
    keys: LeakKeys,
    exempt_index: Iterable[ExemptSpan] | None = None,
    *,
    structured_surfaces: Mapping[Channel, Iterable[str]] | None = None,
    max_decode_bytes: int = 256 * 1024,
    max_wrappers: int = 256,
    max_decode_depth: int = 2,
    diagnostics: dict[str, int] | None = None,
) -> ScanOutcome:
    """Scan rendered output surfaces and return ``ScanOutcome`` with verification status."""
    if not isinstance(keys, LeakKeys):
        # Callers with nothing harvested (stubs, tests, the platform's {} default) pass a
        # plain mapping; treat it as an empty key set instead of crashing the gate.
        keys = LeakKeys()
    if exempt_index is not None:
        exempt_index = [e for e in exempt_index if isinstance(e, ExemptSpan)]
    all_hits: list[Hit] = []
    unverified = False
    reason: str | None = None

    # 1. Path channel
    if path:
        all_hits.extend(_scan_normalized_surface(path, path, "path", keys, exempt_index))

    # 2. Content channel (original + bounded decoded views up to max_decode_depth)
    if text:
        all_hits.extend(_scan_normalized_surface(text, path, "content", keys, exempt_index))

        frontier: list[tuple[tuple[int, int] | None, str]] = [(None, text)]
        for depth in range(max_decode_depth):
            next_frontier: list[tuple[tuple[int, int] | None, str]] = []
            for outer_range, current_text in frontier:
                views, view_unverified = _extract_decoded_views(
                    current_text,
                    max_decode_bytes=max_decode_bytes,
                    max_wrappers=max_wrappers,
                    diagnostics=diagnostics,
                )
                if view_unverified:
                    unverified = True
                    reason = reason or "unverified_wrapper"
                for w_start, w_end, decoded_str in views:
                    eff_range = outer_range if outer_range is not None else (w_start, w_end)
                    all_hits.extend(
                        _scan_normalized_surface(
                            decoded_str,
                            path,
                            "content",
                            keys,
                            exempt_index,
                            is_decoded=True,
                            wrapper_range=eff_range,
                        )
                    )
                    if depth + 1 < max_decode_depth:
                        next_frontier.append((eff_range, decoded_str))
            if not next_frontier:
                break
            frontier = next_frontier

    # 3. Structured surfaces (CSV headers, JSON keys, etc.)
    if structured_surfaces:
        for ch, surfaces in structured_surfaces.items():
            for surf in surfaces:
                all_hits.extend(_scan_normalized_surface(surf, path, ch, keys, exempt_index))

    deduped = _dedup_hits(all_hits)
    _bump(diagnostics, "hits_total", len(deduped))
    if unverified:
        _bump(diagnostics, "unverified_scans")
    return ScanOutcome(hits=deduped, verified=not unverified, reason=reason)


def scan(
    text: str,
    path: str,
    keys: LeakKeys,
    exempt_index: Iterable[ExemptSpan] | None = None,
    *,
    structured_surfaces: Mapping[Channel, Iterable[str]] | None = None,
    strict: bool = False,
    diagnostics: dict[str, int] | None = None,
) -> list[Hit]:
    """Scan rendered output surfaces and return deduplicated ``Hit`` list."""
    outcome = scan_with_status(
        text,
        path,
        keys,
        exempt_index,
        structured_surfaces=structured_surfaces,
        diagnostics=diagnostics,
    )
    if strict and not outcome.verified:
        raise ResidualScanError(outcome.reason or "unverified_scan")
    return list(outcome.hits)


def scan_gate(
    text: str,
    out_path: str,
    leak_keys: LeakKeys,
    exempt_ranges: Iterable[tuple[int, int, str]] | None = None,
) -> list[Hit]:
    """Adapter for A's read-back gate, which passes ``(start, end, kind)`` content ranges.

    The gate's ``LocMap.clinical_ranges()`` yields plain tuples; ``scan`` expects
    ``ExemptSpan`` objects. Passing tuples straight through raised ``AttributeError``
    and the gate quarantined every file it touched.
    """
    spans = [
        ExemptSpan(out_path=out_path, channel="content", start=s, end=e,
                   exempt_kind=_gate_exempt_kind(text[s:e], k))
        for s, e, k in (exempt_ranges or ())
    ]
    return scan(text, out_path, leak_keys, spans)


_DOCTOR_FULL_NAME_RE = re.compile(
    r"^\s*(?:(?:Consultant|Attending|Practitioner)\s*:\s*|Referred\s+by\s*|ref_doctor\s*)?(?:Dr\.?|डॉ\.?)\s*([A-Zऀ-ॿ][\wऀ-ॿ.'-]*)(?:\s+[A-Zऀ-ॿ][\wऀ-ॿ.'-]*)+\s*$"
)


def _gate_exempt_kind(span_text: str, kind: str) -> str:
    """FR-R7 without a doctor registry: a `Dr.` span naming a full name (>= 2 tokens)
    is treated like a registry hit. A single-token `Dr. Sunita` stays non-exempt, so a
    relative who shares only a first name with a doctor is still caught."""
    if kind == "doctor_context" and _DOCTOR_FULL_NAME_RE.match(span_text):
        return "doctor_registry"
    return kind
