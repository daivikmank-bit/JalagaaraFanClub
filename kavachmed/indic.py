"""Multilingual name normalization, Devanagari transliteration, and consonant skeletons."""

from __future__ import annotations

from dataclasses import dataclass
import unicodedata
from typing import Literal

TokenKind = Literal["identifying", "generic", "honorific", "initial"]


@dataclass(frozen=True, slots=True)
class NameToken:
    """Normalized representation of a single name token."""

    surface: str
    latin: str | None
    kind: TokenKind
    skeleton: str | None


# Honorifics and titles (normalized source-script forms, Latin and Devanagari).
HONORIFICS: frozenset[str] = frozenset(
    {
        "baba",
        "begum",
        "bhai",
        "chacha",
        "dada",
        "didi",
        "dr",
        "janab",
        "jee",
        "ji",
        "kaka",
        "kumari",
        "madam",
        "mama",
        "maulana",
        "mr",
        "mrs",
        "ms",
        "pandit",
        "prof",
        "sahab",
        "saheb",
        "shree",
        "shri",
        "shrimati",
        "sir",
        "smt",
        "कुमारी",
        "जी",
        "डॉ",
        "भाई",
        "बेगम",
        "श्री",
        "श्रीमती",
        "साहब",
        "साहिब",
    }
)

# High-frequency generic name tokens that carry little or no identifying signal.
GENERICS: frozenset[str] = frozenset(
    {
        "ahmad",
        "ahmed",
        "ali",
        "bano",
        "bibi",
        "chand",
        "das",
        "devi",
        "gupta",
        "husain",
        "hussain",
        "iyengar",
        "iyer",
        "kaur",
        "khan",
        "kumar",
        "lal",
        "menon",
        "nair",
        "nath",
        "pal",
        "patel",
        "prasad",
        "rai",
        "ram",
        "rao",
        "reddy",
        "roy",
        "sen",
        "sharma",
        "singh",
        "uddin",
        "verma",
        "कुमार",
        "कौर",
        "खान",
        "दास",
        "देवी",
        "प्रसाद",
        "राम",
        "लाल",
        "सिंह",
    }
)

# Known spelling/script aliases mapped to a canonical Latin token (classified generic).
ALIASES: dict[str, str] = {
    "md": "mhmd",
    "mohd": "mhmd",
    "mohammad": "mhmd",
    "mohammed": "mhmd",
    "muhammad": "mhmd",
    "muhammed": "mhmd",
    "मोहम्मद": "mhmd",
    "मुहम्मद": "mhmd",
}

_DEVANAGARI_CONSONANTS: dict[str, str] = {
    "क": "k",
    "ख": "kh",
    "ग": "g",
    "घ": "gh",
    "ङ": "ng",
    "च": "ch",
    "छ": "chh",
    "ज": "j",
    "झ": "jh",
    "ञ": "ny",
    "ट": "t",
    "ठ": "th",
    "ड": "d",
    "ढ": "dh",
    "ण": "n",
    "त": "t",
    "थ": "th",
    "द": "d",
    "ध": "dh",
    "न": "n",
    "प": "p",
    "फ": "ph",
    "ब": "b",
    "भ": "bh",
    "म": "m",
    "य": "y",
    "र": "r",
    "ल": "l",
    "व": "v",
    "श": "sh",
    "ष": "sh",
    "स": "s",
    "ह": "h",
    "ळ": "l",
    # Precomposed nukta forms (in case NFKC does not decompose or input uses them)
    "क़": "q",
    "ख़": "kh",
    "ग़": "gh",
    "ज़": "z",
    "ड़": "r",
    "ढ़": "rh",
    "फ़": "f",
    "य़": "y",
}

_DEVANAGARI_NUKTA_MAP: dict[str, str] = {
    "क": "q",
    "ख": "kh",
    "ग": "gh",
    "ज": "z",
    "ड": "r",
    "ढ": "rh",
    "फ": "f",
    "य": "y",
}

_DEVANAGARI_INDEPENDENT_VOWELS: dict[str, str] = {
    "अ": "a",
    "आ": "a",
    "इ": "i",
    "ई": "i",
    "उ": "u",
    "ऊ": "u",
    "ऋ": "ri",
    "ए": "e",
    "ऐ": "ai",
    "ओ": "o",
    "औ": "au",
    "ॲ": "a",
    "ऑ": "o",
}

_DEVANAGARI_VOWEL_SIGNS: dict[str, str] = {
    "ा": "a",
    "ि": "i",
    "ी": "i",
    "ु": "u",
    "ू": "u",
    "ृ": "ri",
    "े": "e",
    "ै": "ai",
    "ो": "o",
    "ौ": "au",
    "ॅ": "a",
    "ॉ": "o",
}

_DEVANAGARI_MODIFIERS: dict[str, str] = {
    "ं": "n",
    "ँ": "n",
    "ः": "h",
}

_VIRAMA = "\u094d"
_NUKTA = "\u093c"
_LATIN_VOWELS = frozenset({"a", "e", "i", "o", "u"})


def _is_devanagari(ch: str) -> bool:
    code = ord(ch)
    return 0x0900 <= code <= 0x097F


def _is_ascii_alpha(ch: str) -> bool:
    return "a" <= ch <= "z"


def _normalize_unicode(text: str) -> str:
    """Apply NFKC, remove format chars (Cf), casefold, and strip Latin accents."""
    if not text:
        return ""
    nfkc = unicodedata.normalize("NFKC", text)
    cleaned = "".join(ch for ch in nfkc if unicodedata.category(ch) != "Cf")
    folded = cleaned.casefold()
    decomposed = unicodedata.normalize("NFD", folded)

    out: list[str] = []
    last_base_is_latin = False
    for ch in decomposed:
        cat = unicodedata.category(ch)
        if cat == "Mn" and last_base_is_latin and not _is_devanagari(ch):
            continue
        out.append(ch)
        if not cat.startswith("M"):
            last_base_is_latin = _is_ascii_alpha(ch)
    return "".join(out)


def _tokenize_normalized(norm: str) -> list[str]:
    """Split normalized text into runs of letters and their attached marks."""
    tokens: list[str] = []
    buf: list[str] = []
    for ch in norm:
        cat = unicodedata.category(ch)
        if cat.startswith("L"):
            buf.append(ch)
        elif cat.startswith("M") and buf:
            buf.append(ch)
        else:
            if buf:
                tokens.append("".join(buf))
                buf.clear()
    if buf:
        tokens.append("".join(buf))
    return tokens


def _transliterate_devanagari(token: str) -> str | None:
    """Table-driven Devanagari-to-Latin transliterator with conservative schwa deletion."""
    pieces: list[tuple[str, bool]] = []  # (text, is_implicit_schwa)
    i = 0
    n = len(token)

    while i < n:
        ch = token[i]
        if ch in _DEVANAGARI_INDEPENDENT_VOWELS:
            pieces.append((_DEVANAGARI_INDEPENDENT_VOWELS[ch], False))
            i += 1
            continue

        if ch in _DEVANAGARI_MODIFIERS:
            pieces.append((_DEVANAGARI_MODIFIERS[ch], False))
            i += 1
            continue

        if ch in _DEVANAGARI_CONSONANTS:
            base = _DEVANAGARI_CONSONANTS[ch]
            i += 1
            if i < n and token[i] == _NUKTA:
                base = _DEVANAGARI_NUKTA_MAP.get(ch, base)
                i += 1
            pieces.append((base, False))

            if i < n and token[i] == _VIRAMA:
                i += 1
            elif i < n and token[i] in _DEVANAGARI_VOWEL_SIGNS:
                pieces.append((_DEVANAGARI_VOWEL_SIGNS[token[i]], False))
                i += 1
            else:
                pieces.append(("a", True))
            continue

        return None

    if not pieces:
        return None

    # Drop an implicit final schwa in multisyllabic tokens; preserve explicit vowels.
    has_prior_vowel = any(
        any(c in _LATIN_VOWELS for c in text) for text, _ in pieces[:-1]
    )
    if pieces[-1][1] and has_prior_vowel:
        pieces.pop()

    result = "".join(text for text, _ in pieces)
    return result or None


def _derive_skeleton(latin: str) -> str | None:
    """Derive consonant skeleton for an identifying Latin token."""
    phonetic = latin.replace("w", "v").replace("ph", "f")
    skeleton = "".join(ch for ch in phonetic if ch not in _LATIN_VOWELS)
    return skeleton or None


def _classify_token(surface: str) -> NameToken:
    """Classify and transliterate a single normalized surface token."""
    if surface in ALIASES:
        return NameToken(
            surface=surface,
            latin=ALIASES[surface],
            kind="generic",
            skeleton=None,
        )

    if surface in HONORIFICS:
        latin = surface if surface.isascii() else _transliterate_devanagari(surface)
        return NameToken(
            surface=surface,
            latin=latin,
            kind="honorific",
            skeleton=None,
        )

    if surface in GENERICS:
        latin = surface if surface.isascii() else _transliterate_devanagari(surface)
        return NameToken(
            surface=surface,
            latin=latin,
            kind="generic",
            skeleton=None,
        )

    if all(_is_ascii_alpha(ch) for ch in surface):
        if len(surface) == 1:
            return NameToken(
                surface=surface,
                latin=surface,
                kind="initial",
                skeleton=None,
            )
        return NameToken(
            surface=surface,
            latin=surface,
            kind="identifying",
            skeleton=_derive_skeleton(surface),
        )

    if all(_is_devanagari(ch) for ch in surface):
        latin = _transliterate_devanagari(surface)
        if latin is None:
            return NameToken(
                surface=surface,
                latin=None,
                kind="identifying",
                skeleton=None,
            )
        if latin in ALIASES:
            return NameToken(
                surface=surface,
                latin=ALIASES[latin],
                kind="generic",
                skeleton=None,
            )
        if latin in HONORIFICS:
            return NameToken(
                surface=surface,
                latin=latin,
                kind="honorific",
                skeleton=None,
            )
        if latin in GENERICS:
            return NameToken(
                surface=surface,
                latin=latin,
                kind="generic",
                skeleton=None,
            )
        return NameToken(
            surface=surface,
            latin=latin,
            kind="identifying",
            skeleton=_derive_skeleton(latin),
        )

    return NameToken(
        surface=surface,
        latin=None,
        kind="identifying",
        skeleton=None,
    )


def lexicon_tokens(text: str) -> tuple[str, ...]:
    """Return normalized source-script tokens for safe-word subtraction."""
    if not text:
        return ()
    norm = _normalize_unicode(text)
    if not norm:
        return ()
    return tuple(_tokenize_normalized(norm))


def normalize_name(text: str) -> tuple[NameToken, ...]:
    """Normalize, transliterate, classify, and derive skeletons for a name string."""
    surfaces = lexicon_tokens(text)
    if not surfaces:
        return ()
    return tuple(_classify_token(surface) for surface in surfaces)
