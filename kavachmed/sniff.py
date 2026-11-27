"""Magic-byte sniffing, decode ladder, and content-based format classification."""
from __future__ import annotations

import csv
import io
import re
from pathlib import Path

_BINARY_MAGICS: tuple[bytes, ...] = (
    b"\xff\xd8\xff",          # JPEG
    b"%PDF-",                 # PDF
    b"\x89PNG\r\n\x1a\n",     # PNG
    b"GIF87a",                # GIF87a
    b"GIF89a",                # GIF89a
    b"PK\x03\x04",            # ZIP / OOXML
    b"PK\x05\x06",            # ZIP empty
)

_UTF8_MULTIBYTE_RE = re.compile(
    rb"(?:[\xc2-\xdf][\x80-\xbf]|[\xe0-\xef][\x80-\xbf]{2}|[\xf0-\xf4][\x80-\xbf]{3})"
)


def is_binary_magic(raw: bytes) -> bool:
    """Return True if raw bytes start with a known binary container signature."""
    return any(raw.startswith(sig) for sig in _BINARY_MAGICS)


def _try_bomless_utf16(raw: bytes) -> tuple[str | None, str | None]:
    """Detect BOM-less UTF-16LE/BE via alternating NUL pattern in complete code units."""
    if len(raw) < 4 or len(raw) % 2 != 0:
        return None, None
    even_nuls = sum(1 for i in range(0, len(raw), 2) if raw[i] == 0)
    odd_nuls = sum(1 for i in range(1, len(raw), 2) if raw[i] == 0)
    total_units = len(raw) // 2
    if odd_nuls >= total_units * 0.6 and even_nuls <= total_units * 0.2:
        enc = "utf-16-le"
    elif even_nuls >= total_units * 0.6 and odd_nuls <= total_units * 0.2:
        enc = "utf-16-be"
    else:
        return None, None
    try:
        text = raw.decode(enc, errors="strict")
    except UnicodeDecodeError:
        return None, None
    if "\x00" in text:
        return None, None
    return text, enc


def _has_unexplained_controls(text: str) -> bool:
    for ch in text:
        o = ord(ch)
        if o < 32 and ch not in ("\t", "\n", "\r"):
            return True
    return False


def decode_bytes(raw: bytes) -> tuple[str | None, str | None, str | None]:
    """Decode ladder: BOM -> BOM-less UTF-16 -> strict UTF-8 -> cp1252.

    Returns (text, encoding, error_reason).
    """
    if raw.startswith(b"\xef\xbb\xbf"):
        try:
            text = raw[3:].decode("utf-8", errors="strict")
            if b"\x00" in raw[3:] or _has_unexplained_controls(text):
                return None, None, "undecodable"
            return text, "utf-8-sig", None
        except UnicodeDecodeError:
            return None, None, "undecodable"

    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            text = raw.decode("utf-16", errors="strict")
            if _has_unexplained_controls(text):
                return None, None, "undecodable"
            return text, "utf-16", None
        except UnicodeDecodeError:
            return None, None, "undecodable"

    text_u16, enc_u16 = _try_bomless_utf16(raw)
    if text_u16 is not None:
        return text_u16, enc_u16, None

    if b"\x00" in raw:
        return None, None, "undecodable"

    try:
        text_u8 = raw.decode("utf-8", errors="strict")
        if _has_unexplained_controls(text_u8):
            return None, None, "undecodable"
        return text_u8, "utf-8", None
    except UnicodeDecodeError:
        pass

    # Distinguish positive mixed-encoding evidence (valid multi-byte UTF-8 or stray BOM bytes)
    if b"\xff" in raw or b"\xfe" in raw or _UTF8_MULTIBYTE_RE.search(raw):
        return None, None, "mixed_encoding"

    try:
        text_cp = raw.decode("cp1252", errors="strict")
        if _has_unexplained_controls(text_cp):
            return None, None, "undecodable"
        return text_cp, "cp1252", None
    except UnicodeDecodeError:
        return None, None, "undecodable"


def _delimiter_score(text: str, delim: str) -> int:
    try:
        rows = list(csv.reader(io.StringIO(text, newline=""), delimiter=delim))
    except csv.Error:
        return 0
    if len(rows) < 2:
        return 0
    widths = [len(r) for r in rows[:20] if r]
    if not widths or max(widths) < 2:
        return 0
    first = widths[0]
    consistent = sum(1 for w in widths if w == first)
    return consistent * first


_JSON_ARRAY_START_RE = re.compile(
    r"^\[\s*(?:\{|\[|\"|\]|true\b|false\b|null\b|-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\s*(?:,|\]|$))"
)


def sniff_content(path: Path, text: str) -> str:
    """Classify decoded text by content first, using extension only as tie-breaker."""
    stripped = text.lstrip()
    if stripped.startswith("{") or _JSON_ARRAY_START_RE.match(stripped):
        return "json"
    if stripped.startswith(("<?xml", "<ClinicalDocument")):
        return "xml"

    ext = path.suffix.lower()
    if ext == ".json":
        return "json"
    if ext == ".xml":
        return "xml"

    if ext in (".csv", ".tsv"):
        tab_score = _delimiter_score(text, "\t")
        comma_score = _delimiter_score(text, ",")
        if tab_score > comma_score and tab_score >= 4:
            return "tsv"
        if comma_score > tab_score and comma_score >= 4:
            return "csv"
        return "tsv" if ext == ".tsv" else "csv"

    if ext not in (".txt", ".log", ".md"):
        tab_score = _delimiter_score(text, "\t")
        comma_score = _delimiter_score(text, ",")
        if tab_score > comma_score and tab_score >= 6:
            return "tsv"
        if comma_score > tab_score and comma_score >= 6:
            return "csv"

    return "txt"
