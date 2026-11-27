"""B10: Hidden-channel detection, decoding, and offset-mapped normalization."""

from __future__ import annotations

import base64
import binascii
import html
import re
import unicodedata

from kavachmed.contracts import Context, Field, Record, Span
from kavachmed.detect.context import detect_context
from kavachmed.detect.default_deny import detect_default_deny
from kavachmed.detect.ids import detect_ids
from kavachmed.shield import protect

_DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_ZERO_WIDTH = {"\u200b", "\u200c", "\u200d", "\u200e", "\u200f", "\ufeff", "\u00ad"}
_HOMOGLYPHS = str.maketrans({
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T", "Х": "X",
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y",
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Χ": "X",
})

_RE_B64 = re.compile(r"(?<![A-Za-z0-9+/])([A-Za-z0-9+/]{16,}={0,2})(?![A-Za-z0-9+/=])")
_RE_HEX = re.compile(r"(?<![A-Za-z0-9])(?:0x)?([0-9A-Fa-f]{16,})(?![A-Za-z0-9])")
_RE_PCT = re.compile(r"(?:%[0-9A-Fa-f]{2})+")
_RE_HTML_ENT = re.compile(r"&(?:#\d+|#x[0-9A-Fa-f]+|[A-Za-z]+);")
_RE_ESC = re.compile(r"\\u([0-9A-Fa-f]{4})|\\x([0-9A-Fa-f]{2})")

_PII_KEY_HINTS = {
    "name", "patient_name", "pt_name", "aadhaar", "aadhar", "abha", "phone", "mobile",
    "pan", "email", "address", "dob", "birth", "relative", "father", "husband", "exported_by",
    "source_path", "meta.source", "uhid", "mrn",
}


def normalize_with_offset_map(text: str) -> tuple[str, list[int]]:
    """Normalize text (NFKC, homoglyph fold, zero-width removal, Indic digits) and return offset map."""
    out_chars: list[str] = []
    offset_map: list[int] = []
    for orig_idx, ch in enumerate(text):
        if ch in _ZERO_WIDTH:
            continue
        folded = unicodedata.normalize("NFKC", ch).translate(_HOMOGLYPHS).translate(_DEVA_DIGITS)
        for fc in folded:
            out_chars.append(fc)
            offset_map.append(orig_idx)
    return "".join(out_chars), offset_map


def _decode_pct_with_map(text: str) -> tuple[str, list[tuple[int, int]]]:
    out: list[str] = []
    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(text):
        m = _RE_PCT.match(text, i)
        if m:
            encoded = m.group()
            raw = bytes(int(encoded[j + 1:j + 3], 16) for j in range(0, len(encoded), 3))
            try:
                decoded = raw.decode("utf-8")
            except UnicodeDecodeError:
                for j, ch in enumerate(encoded):
                    out.append(ch)
                    spans.append((i + j, i + j + 1))
                i = m.end()
                continue
            byte_offset = 0
            for ch in decoded:
                byte_count = len(ch.encode("utf-8"))
                out.append(ch)
                spans.append((i + 3 * byte_offset, i + 3 * (byte_offset + byte_count)))
                byte_offset += byte_count
            i = m.end()
        else:
            out.append(text[i])
            spans.append((i, i + 1))
            i += 1
    return "".join(out), spans


def _decode_regex_with_map(text: str, pattern: re.Pattern[str], decode_fn) -> tuple[str, list[tuple[int, int]]]:
    out: list[str] = []
    spans: list[tuple[int, int]] = []
    pos = 0
    for m in pattern.finditer(text):
        for k in range(pos, m.start()):
            out.append(text[k])
            spans.append((k, k + 1))
        rep = decode_fn(m)
        for ch in rep:
            out.append(ch)
            spans.append((m.start(), m.end()))
        pos = m.end()
    for k in range(pos, len(text)):
        out.append(text[k])
        spans.append((k, k + 1))
    return "".join(out), spans


def _project_span_index(sp: Span, offset_map: list[int], orig_len: int) -> Span:
    if not offset_map or sp.start >= len(offset_map):
        return sp
    orig_start = offset_map[sp.start]
    end_idx = min(sp.end - 1, len(offset_map) - 1)
    orig_end = min(orig_len, offset_map[end_idx] + 1)
    return Span(
        key=sp.key,
        field_idx=sp.field_idx,
        start=orig_start,
        end=orig_end,
        typ=sp.typ,
        role=sp.role,
        score=sp.score,
        src=sp.src,
        protected=sp.protected,
        exempt_kind=sp.exempt_kind,
    )


def _project_span_ranges(sp: Span, char_ranges: list[tuple[int, int]], orig_len: int) -> Span:
    if not char_ranges or sp.start >= len(char_ranges):
        return sp
    orig_start = char_ranges[sp.start][0]
    end_idx = min(sp.end - 1, len(char_ranges) - 1)
    orig_end = min(orig_len, char_ranges[end_idx][1])
    return Span(
        key=sp.key,
        field_idx=sp.field_idx,
        start=orig_start,
        end=orig_end,
        typ=sp.typ,
        role=sp.role,
        score=sp.score,
        src=sp.src,
        protected=sp.protected,
        exempt_kind=sp.exempt_kind,
    )


def _run_stack(rec: Record, ctx: Context) -> list[Span]:
    id_sp = detect_ids(rec, ctx)
    cx_sp = detect_context(rec, ctx)
    dd_sp = detect_default_deny(rec, ctx, protect(rec, ctx), id_sp + cx_sp)
    return id_sp + cx_sp + dd_sp


def detect_hidden(rec: Record, ctx: Context) -> list[Span]:
    """Detect PII in paths, keys, headers, metadata, encoded blobs, and normalized text."""
    spans: list[Span] = []

    for fld in rec.fields:
        text = fld.text
        is_meta = fld.vtype in {"path", "key", "header"} or fld.path in {"source_path", "exported_by", "meta.source"}

        if is_meta:
            sub_rec = Record(key=rec.key, fields=[Field(idx=fld.idx, path=fld.path, text=text, vtype="str")], meta=rec.meta)
            spans.extend(_run_stack(sub_rec, ctx))
            # A key/header that merely NAMES a PII field ("uhid", "dob", "address") is schema, not
            # PII; renaming it breaks schema validity and the date-order hints in headers. Only
            # PII values inside keys/headers (UHIDs as keys, "JJH-0411556 Zainab") are flagged,
            # by the detector stack above.

        # 1. Normalized view (NFKC, homoglyphs, zero-width, Indic digits) — full stack incl. default_deny
        norm_text, off_map = normalize_with_offset_map(text)
        if norm_text != text:
            norm_rec = Record(key=rec.key, fields=[Field(idx=fld.idx, path=fld.path, text=norm_text, vtype="str")], meta=rec.meta)
            for sp in _run_stack(norm_rec, ctx):
                spans.append(_project_span_index(sp, off_map, len(text)))

        # 2. Base64 blobs (maps to blob boundary)
        for m in _RE_B64.finditer(text):
            blob = m.group(1)
            if len(blob) % 4 != 0:
                continue
            try:
                raw = base64.b64decode(blob, validate=True)
                dec_txt = raw.decode("utf-8")
                if sum(c.isprintable() or c.isspace() for c in dec_txt) / max(1, len(dec_txt)) <= 0.85:
                    continue
            except (binascii.Error, UnicodeDecodeError):
                continue
            dec_rec = Record(key=rec.key, fields=[Field(idx=fld.idx, path=fld.path, text=dec_txt, vtype="str")], meta=rec.meta)
            for sp in _run_stack(dec_rec, ctx):
                spans.append(Span(rec.key, fld.idx, m.start(1), m.end(1), sp.typ, sp.role, sp.score, sp.src))

        # Hex-encoded UTF-8 is replaced as a whole blob when it contains PII.
        for m in _RE_HEX.finditer(text):
            blob = m.group(1)
            if len(blob) % 2:
                continue
            try:
                dec_txt = bytes.fromhex(blob).decode("utf-8")
            except (ValueError, UnicodeDecodeError):
                continue
            if sum(c.isprintable() or c.isspace() for c in dec_txt) / max(1, len(dec_txt)) <= 0.85:
                continue
            dec_rec = Record(key=rec.key, fields=[Field(idx=fld.idx, path=fld.path, text=dec_txt, vtype="str")], meta=rec.meta)
            for sp in _run_stack(dec_rec, ctx):
                spans.append(Span(rec.key, fld.idx, m.start(), m.end(), sp.typ, sp.role, sp.score, sp.src))

        # 3. Percent-encoding, HTML entities, \uXXXX/\xXX with exact char-range projection
        dec_views: list[tuple[str, list[tuple[int, int]]]] = []
        if _RE_PCT.search(text):
            dec_views.append(_decode_pct_with_map(text))
        if "&" in text and ";" in text:
            dec_views.append(_decode_regex_with_map(text, _RE_HTML_ENT, lambda m: html.unescape(m.group(0))))
        if _RE_ESC.search(text):
            dec_views.append(
                _decode_regex_with_map(
                    text,
                    _RE_ESC,
                    lambda m: chr(int(m.group(1), 16)) if m.group(1) else chr(int(m.group(2), 16)),
                )
            )

        for dec_txt, ranges in dec_views:
            if dec_txt == text:
                continue
            dec_rec = Record(key=rec.key, fields=[Field(idx=fld.idx, path=fld.path, text=dec_txt, vtype="str")], meta=rec.meta)
            for sp in _run_stack(dec_rec, ctx):
                spans.append(_project_span_ranges(sp, ranges, len(text)))

    return spans
