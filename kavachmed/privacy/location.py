"""LGD lexicon loader and location generalization (D4, FR-P4, FR-X1)."""
from __future__ import annotations

import bisect
import hashlib
import re
import struct
import unicodedata
from pathlib import Path
from typing import Literal

from kavachmed.contracts import Evidence, Record

_LEX_ROOT = Path(__file__).resolve().parent.parent.parent / "config" / "lexicons" / "places"

_PIN_RE = re.compile(r"\b([1-9][0-9]{5})\b")
_DEVA_PIN_RE = re.compile(r"(?<![०-९])([१-९][०-९]{5})(?![०-९])")
_DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

_HOUSE_WARD_RE = re.compile(
    r"(?:H\.?\s*No\.?|House\s*No\.?|Flat\s*No\.?|Plot\s*No\.?|Ward\s*No\.?|Mohalla|Colony|Nagar|Gali|Lane|Road|Marg|Near|Opp\.?|Behind|Landmark)\s*[:#-]?\s*[^,\n;]+",
    re.IGNORECASE,
)
_VILLAGE_TEHSIL_RE = re.compile(
    r"\b(?:Vill(?:age)?|Gram|Gaon|Post|PO|Tehsil|Taluka|Block|Sub-?district|ग्राम|गाँव|गांव|तहसील|मोहल्ला|मकान\s*नं\.?)\s*[:#-]?\s*[^,\n;]+",
    re.IGNORECASE,
)
_CO_PREFIX_RE = re.compile(
    r"^(?:C\s*/\s*O|S\s*/\s*O|D\s*/\s*O|W\s*/\s*O)\s+[^,\n;]+(?:,\s*)?",
    re.IGNORECASE,
)


def _norm(s: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", s).casefold().split())


def _hash64(token: str) -> int:
    digest = hashlib.sha256(_norm(token).encode("utf-8")).digest()
    return struct.unpack("<Q", digest[:8])[0]


class LGDIndex:
    """In-memory index of LGD states, districts, sub-districts, and hashed villages."""

    def __init__(self, lex_dir: Path | None = None) -> None:
        base = lex_dir or _LEX_ROOT
        self.states_by_code: dict[str, str] = {}
        self.state_norm_to_canon: dict[str, tuple[str, str]] = {}
        self.district_norm_to_info: dict[str, tuple[str, str, str]] = {}
        self.subdistrict_norm_to_dcode: dict[str, str] = {}
        self.village_hashes: list[int] = []

        st_path = base / "lgd_states.tsv"
        if st_path.is_file():
            for line in st_path.read_text(encoding="utf-8").splitlines()[1:]:
                parts = line.split("\t")
                if len(parts) >= 3:
                    code, en, hi = parts[0].strip(), parts[1].strip(), parts[2].strip()
                    self.states_by_code[code] = en
                    self.state_norm_to_canon[_norm(en)] = (code, en)
                    if hi:
                        self.state_norm_to_canon[_norm(hi)] = (code, en)

        dist_path = base / "lgd_districts.tsv"
        if dist_path.is_file():
            for line in dist_path.read_text(encoding="utf-8").splitlines()[1:]:
                parts = line.split("\t")
                if len(parts) >= 4:
                    dcode, scode, en, hi = parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()
                    self.district_norm_to_info[_norm(en)] = (dcode, en, scode)
                    if hi:
                        self.district_norm_to_info[_norm(hi)] = (dcode, en, scode)

        sd_path = base / "lgd_subdistricts.tsv"
        if sd_path.is_file():
            for line in sd_path.read_text(encoding="utf-8").splitlines()[1:]:
                parts = line.split("\t")
                if len(parts) >= 4:
                    _sdcode, dcode, en, hi = parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()
                    self.subdistrict_norm_to_dcode[_norm(en)] = dcode
                    if hi:
                        self.subdistrict_norm_to_dcode[_norm(hi)] = dcode

        v_path = base / "lgd_villages.bin"
        if v_path.is_file():
            raw = v_path.read_bytes()
            n = len(raw) // 8
            if n > 0:
                self.village_hashes = list(struct.unpack(f"<{n}Q", raw[: n * 8]))

    def is_village_hashed(self, token: str) -> bool:
        if not self.village_hashes or not token.strip():
            return False
        h = _hash64(token)
        idx = bisect.bisect_left(self.village_hashes, h)
        return idx < len(self.village_hashes) and self.village_hashes[idx] == h

    def lookup_district(self, text: str) -> tuple[str, str, str] | None:
        """Return (district_code, canonical_district_name, canonical_state_name) if matched."""
        n = _norm(text)
        if not n:
            return None
        info = self.district_norm_to_info.get(n)
        if info:
            dcode, dname, scode = info
            return (dcode, dname, self.states_by_code.get(scode, "India"))
        pat = self._district_scan_re()
        if pat is None:
            return None
        # Same result as testing every key in insertion order: earliest-inserted matching key wins
        hits = {m.group(0) for m in pat.finditer(n)}
        for k in self._scan_keys:
            if k in hits:
                dcode, dname, scode = self.district_norm_to_info[k]
                return (dcode, dname, self.states_by_code.get(scode, "India"))
        return None

    def _district_scan_re(self) -> re.Pattern[str] | None:
        """One compiled alternation over district keys (len >= 4), built once per index."""
        if not hasattr(self, "_scan_re"):
            self._scan_keys = [k for k in self.district_norm_to_info if len(k) >= 4]
            alt = "|".join(re.escape(k) for k in sorted(self._scan_keys, key=lambda s: (-len(s), s)))
            self._scan_re = re.compile(rf"(?<!\w)(?:{alt})(?!\w)") if alt else None
        return self._scan_re

    def state_for_district(self, district_name: str) -> str | None:
        info = self.lookup_district(district_name)
        return info[2] if info else None

    def is_known_place_token(self, token: str) -> bool:
        n = _norm(token)
        return (
            n in self.district_norm_to_info
            or n in self.state_norm_to_canon
            or n in self.subdistrict_norm_to_dcode
            or self.is_village_hashed(n)
        )


_DEFAULT_INDEX: LGDIndex | None = None


def get_lgd_index() -> LGDIndex:
    global _DEFAULT_INDEX
    if _DEFAULT_INDEX is None:
        _DEFAULT_INDEX = LGDIndex()
    return _DEFAULT_INDEX


def extract_district_from_record(rec: Record | None, lgd: LGDIndex | None = None, ev: Evidence | None = None) -> str | None:
    """Find district from structured district/jila columns, LGD match, or Evidence.geo."""
    idx = lgd or get_lgd_index()
    if rec is not None:
        for fld in rec.fields:
            col = fld.path.split("/")[-1].strip().casefold()
            if col in ("district", "jila", "जिला", "dist", "distt"):
                val = fld.text.strip()
                if val:
                    info = idx.lookup_district(val)
                    return info[1] if info else val
        for fld in rec.fields:
            if fld.vtype in ("key", "header", "path"):
                continue
            info = idx.lookup_district(fld.text)
            if info:
                return info[1]
    if ev and ev.geo:
        raw_d = str(ev.geo.get("district") or ev.geo.get("city") or "").strip()
        if raw_d:
            info = idx.lookup_district(raw_d)
            return info[1] if info else raw_d
    return None


def generalize_pin(pin_text: str, pin_mode: Literal["none", "pin3"]) -> str:
    ascii_p = pin_text.translate(_DEVA_DIGITS).strip()
    m = _PIN_RE.search(ascii_p)
    if not m:
        return "[PIN]"
    if pin_mode == "pin3":
        return m.group(1)[:3] + "***"
    return "[PIN]"


def sanitize_address_text(
    text: str,
    level: Literal["district", "state", "none"] = "district",
    pin_mode: Literal["none", "pin3"] = "none",
    lgd: LGDIndex | None = None,
) -> str:
    """Remove house/ward/mohalla/landmark/village/tehsil/PIN and keep district or state per ladder."""
    idx = lgd or get_lgd_index()
    info = idx.lookup_district(text)
    if level == "none":
        return "[ADDRESS]"
    if level == "state":
        if info:
            return info[2]
        n = _norm(text)
        for sk, (_sc, sname) in idx.state_norm_to_canon.items():
            if sk in n:
                return sname
        return "[STATE]"
    if info:
        dname = info[1]
        if pin_mode == "pin3":
            m = _PIN_RE.search(text.translate(_DEVA_DIGITS))
            if m:
                return f"{dname} ({m.group(1)[:3]}***)"
        return dname
    cleaned = _CO_PREFIX_RE.sub("", text)
    cleaned = _HOUSE_WARD_RE.sub("", cleaned)
    cleaned = _VILLAGE_TEHSIL_RE.sub("", cleaned)
    if pin_mode == "pin3":
        cleaned = _PIN_RE.sub(lambda m: m.group(1)[:3] + "***", cleaned)
    else:
        cleaned = _PIN_RE.sub("", cleaned)
        cleaned = _DEVA_PIN_RE.sub("", cleaned)
    cleaned = re.sub(r"[,\s-]+", " ", cleaned).strip(" ,;-")
    return cleaned if cleaned else "[ADDRESS]"
