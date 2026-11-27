"""Format-aware attacker evidence extraction matching docs/ATTACKS.md (D9)."""
from __future__ import annotations

import base64
import binascii
import csv
import io
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kavachmed.privacy import dx_rollup, location as loc_mod

_DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?([6-9]\d{4}[\s-]?\d{5})(?!\d)")
_MASKED_PHONE_RE = re.compile(r"\b(\d{2})[xX*]{4,6}(\d{2,4})\b")
_MASKED_AADHAAR_RE = re.compile(r"(?:[Xx*]{4}[\s-]?){2}(\d{4})\b")
_EMAIL_RE = re.compile(r"[\w.+-]+@(?!(?:redacted\.invalid)\b)[\w-]+\.[\w.]{2,}")
_AGE_BAND_RE = re.compile(r"\b(\d{1,3})\s*-\s*(\d{1,3})\b|\b(85)\+\b")
_AGE_EXACT_RE = re.compile(r"(?<![A-Za-z0-9-])(\d{1,3})\s*(?:Y(?:rs?|ears?)?|वर्ष|साल)?\s*/\s*([MF])\b", re.I)
# A number followed by "-NN" or "+" is a band (ATTACKS.md: bands `50-54`, `85+`), not an exact age.
_AGE_ANCHOR_RE = re.compile(r"\b(?:Age|Umar|आयु|उम्र)\s*[:=\-]?\s*(\d{1,3})\b(?!\s*(?:-\s*\d|\+))", re.I)
_DURATION_BEFORE_RE = re.compile(r"(?:x|for|since|back|history\s+of|h/o|k/c/o)\s*$", re.I)
_PIN_RE = re.compile(r"\b([1-9]\d{5})\b")
_PIN3_RE = re.compile(r"\b([1-9]\d{2})\*{3}\b")
_PSN_RE = re.compile(r"\b(PSN-[0-9A-HJKMNP-TV-Z]{6,10})\b")
_ISO_DATE_RE = re.compile(r"\b((?:19|20)\d{2})-(\d{2})-(\d{2})\b")
_DMY_DATE_RE = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-]((?:19|20)\d{2})\b")
_REL_RE = re.compile(r"\b(?:S/o|D/o|W/o|C/O|Guardian|Relative)\s*[:.]?\s*([A-Za-z][A-Za-z\s]{1,30}?)(?=[,\n;]|$)", re.I)
_NAME_KV_RE = re.compile(r"(?:Patient\s*Name|Name|Naam)\s*[:=]\s*([A-Za-z][A-Za-z\s]{2,35}?)(?=[,\n;]|$)", re.I)
_HOUSE_RE = re.compile(r"(?:H\.?\s*No\.?|House\s*No\.?)\s*[:#-]?\s*([A-Za-z0-9/-]{1,12})", re.I)
_B64_RE = re.compile(r"[A-Za-z0-9+/]{24,}={0,2}")


_DEVA_TO_LATIN = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo", "ऋ": "ri",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
    "ा": "a", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo", "ृ": "ri",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ं": "n", "ँ": "n", "ः": "h", "्": "",
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "फ़": "f", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    "ज़": "z", "ड़": "r", "ढ़": "rh", "क़": "q", "ख़": "kh", "ग़": "gh",
}


def _translit_deva(s: str) -> str:
    if not any("\u0900" <= c <= "\u097f" for c in s):
        return s
    out: list[str] = []
    for ch in s:
        if ch in _DEVA_TO_LATIN:
            out.append(_DEVA_TO_LATIN[ch])
        else:
            out.append(ch)
    return "".join(out)


def norm(s: str) -> str:
    base = " ".join(unicodedata.normalize("NFKC", s).translate(_DEVA_DIGITS).casefold().split())
    if any("\u0900" <= c <= "\u097f" for c in base):
        return base + " " + _translit_deva(base)
    return base


@dataclass
class ExtractedRecord:
    file_rel: str
    loc: str
    psn: str | None = None
    ref_ids: list[str] = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    relatives: list[str] = field(default_factory=list)
    phones: list[str] = field(default_factory=list)
    masked_phones: list[tuple[str, str]] = field(default_factory=list)
    last4: list[str] = field(default_factory=list)
    emails: list[str] = field(default_factory=list)
    gender: str | None = None
    age_ranges: list[tuple[int, int]] = field(default_factory=list)
    birth_years: list[int] = field(default_factory=list)
    district: str | None = None
    state: str | None = None
    tehsil: str | None = None
    village: str | None = None
    pin: str | None = None
    pin3: str | None = None
    houses: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    hospital: str | None = None
    dates: list[str] = field(default_factory=list)  # ISO YYYY-MM-DD
    los_days: int | None = None
    raw_text: str = ""


def _decode_b64_append(text: str) -> str:
    extra = []
    for m in _B64_RE.finditer(text):
        try:
            dec = base64.b64decode(m.group(0) + "=" * (-len(m.group(0)) % 4)).decode("utf-8", errors="ignore")
            if sum(c.isprintable() for c in dec) > len(dec) * 0.7:
                extra.append(dec)
        except (binascii.Error, ValueError):
            pass
    return text + "\n" + "\n".join(extra) if extra else text


def _infer_hosp(rel: str, text: str) -> str | None:
    comb = (rel + " " + text[:400]).lower()
    for name, toks in (
        ("Sanjeevani", ("sanjeevani", "snj")),
        ("Jeevan Jyoti", ("jeevan", "jjh")),
        ("Arogya", ("arogya", "ass")),
        ("DH Hardoi", ("hardoi", "dhh")),
    ):
        if any(t in comb for t in toks):
            return name
    return None


def _parse_kv_evidence(pairs: list[tuple[str, str]], file_rel: str, loc: str) -> ExtractedRecord:
    ev = ExtractedRecord(file_rel=file_rel, loc=loc)
    lgd = loc_mod.get_lgd_index()
    chunks = []
    for k, v in pairs:
        kl = k.casefold().strip()
        vs = v.strip()
        if not vs:
            continue
        chunks.append(f"{k}: {vs}")
        psn_m = _PSN_RE.search(vs)
        if psn_m and not ev.psn:
            ev.psn = psn_m.group(1)

        if any(x in kl for x in ("patient_name", "patient name", "naam", "firstname", "lastname", "name")) and "dr" not in vs.lower() and "[" not in vs and "PSN-" not in vs:
            if "guardian" in kl or "relative" in kl or "pita" in kl or "mother" in kl:
                ev.relatives.append(norm(vs))
            else:
                ev.names.append(norm(vs))
        elif any(x in kl for x in ("gender", "sex", "ling")):
            vl = vs.casefold()
            if vl.startswith(("m", "पु")):
                ev.gender = "M"
            elif vl.startswith(("f", "म", "स्")):
                ev.gender = "F"
        elif any(x in kl for x in ("age", "umar", "आयु", "उम्र")):
            bm = _AGE_BAND_RE.search(vs)
            if bm:
                if bm.group(3):
                    ev.age_ranges.append((85, 120))
                else:
                    ev.age_ranges.append((int(bm.group(1)), int(bm.group(2))))
            elif em := _AGE_EXACT_RE.search(vs.translate(_DEVA_DIGITS)):
                a = int(em.group(1))
                ev.age_ranges.append((max(0, a - 1), min(120, a + 1)))
                if not ev.gender:
                    ev.gender = em.group(2).upper()
            elif vs.translate(_DEVA_DIGITS).isdigit():
                a = int(vs.translate(_DEVA_DIGITS))
                ev.age_ranges.append((max(0, a - 1), min(120, a + 1)))
        elif any(x in kl for x in ("dob", "birthdate", "janm")):
            ym = re.search(r"\b(19\d{2}|20[0-2]\d)\b", vs)
            if ym:
                ev.birth_years.append(int(ym.group(1)))
        elif any(x in kl for x in ("district", "jila", "जिला")):
            info = lgd.lookup_district(vs)
            ev.district = info[1] if info else vs
        elif any(x in kl for x in ("state", "राज्य")):
            ev.state = vs
        elif any(x in kl for x in ("tehsil", "taluka")):
            if "[" not in vs:
                ev.tehsil = norm(vs)
        elif any(x in kl for x in ("village", "gaon", "गाँव")):
            if "[" not in vs:
                ev.village = norm(vs)
        elif any(x in kl for x in ("pin", "zip", "postal")):
            if m3 := _PIN3_RE.search(vs):
                ev.pin3 = m3.group(1)
            elif mp := _PIN_RE.search(vs):
                ev.pin = mp.group(1)
        elif any(x in kl for x in ("uhid", "patient_ref", "reg no", "patientid")) and "[" not in vs:
            ev.ref_ids.append(norm(vs))
        elif any(x in kl for x in ("diag", "condition", "code.text", "procedure")):
            cid, _icd, _chap = dx_rollup.canonicalize_dx(vs)
            ev.conditions.append(cid)

    full_txt = _decode_b64_append("\n".join(chunks))
    full_ascii = full_txt.translate(_DEVA_DIGITS)
    ev.raw_text = norm(full_txt)
    ev.hospital = _infer_hosp(file_rel, full_txt)

    if not ev.psn:
        pm = _PSN_RE.search(full_txt)
        if pm:
            ev.psn = pm.group(1)

    for m in _PHONE_RE.finditer(full_ascii):
        digs = "".join(c for c in m.group(1) if c.isdigit())
        if len(digs) == 10:
            ev.phones.append(digs)
    for m in _MASKED_PHONE_RE.finditer(full_ascii):
        ev.masked_phones.append((m.group(1), m.group(2)))
    for m in _MASKED_AADHAAR_RE.finditer(full_ascii):
        ev.last4.append(m.group(1))
    for m in _EMAIL_RE.finditer(full_txt):
        ev.emails.append(m.group(0).casefold())
    for m in _AGE_EXACT_RE.finditer(full_ascii):
        if not _DURATION_BEFORE_RE.search(full_ascii[max(0, m.start() - 12) : m.start()]):
            a = int(m.group(1))
            ev.age_ranges.append((max(0, a - 1), min(120, a + 1)))
            if not ev.gender:
                ev.gender = m.group(2).upper()
    for m in _AGE_ANCHOR_RE.finditer(full_ascii):
        if not _DURATION_BEFORE_RE.search(full_ascii[max(0, m.start() - 12) : m.start()]):
            a = int(m.group(1))
            ev.age_ranges.append((max(0, a - 1), min(120, a + 1)))
    for m in _AGE_BAND_RE.finditer(full_ascii):
        if m.group(3):
            ev.age_ranges.append((85, 120))
        else:
            lo, hi = int(m.group(1)), int(m.group(2))
            if 0 <= lo < hi <= 120 and (hi - lo) <= 30:
                ev.age_ranges.append((lo, hi))
    for m in _REL_RE.finditer(full_txt):
        rname = norm(m.group(1))
        if rname and "[" not in rname:
            ev.relatives.append(rname)
    for m in _NAME_KV_RE.finditer(full_txt):
        nm = norm(m.group(1))
        if nm and "[" not in nm and "dr." not in nm:
            ev.names.append(nm)
    for m in _HOUSE_RE.finditer(full_txt):
        ev.houses.append(norm(m.group(1)))

    # Dates & LOS
    iso_dates: list[str] = []
    for m in _ISO_DATE_RE.finditer(full_ascii):
        iso_dates.append(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
    for m in _DMY_DATE_RE.finditer(full_ascii):
        a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= a <= 31 and 1 <= b <= 12:
            iso_dates.append(f"{y:04d}-{b:02d}-{a:02d}")
        elif 1 <= b <= 31 and 1 <= a <= 12:
            iso_dates.append(f"{y:04d}-{a:02d}-{b:02d}")
    ev.dates = sorted(set(iso_dates))
    if len(ev.dates) >= 2:
        import datetime as dt

        try:
            d_objs = [dt.date.fromisoformat(x) for x in ev.dates]
            ev.los_days = (max(d_objs) - min(d_objs)).days
        except ValueError:
            pass
    if not ev.district:
        info = lgd.lookup_district(full_txt)
        if info:
            ev.district = info[1]
    return ev


def _flatten_json_pairs(root: Any) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    stack: list[tuple[str, Any]] = [("", root)]
    while stack:
        prefix, obj = stack.pop()
        if isinstance(obj, dict):
            for k, v in reversed(list(obj.items())):
                np = f"{prefix}.{k}" if prefix else str(k)
                stack.append((np, v))
        elif isinstance(obj, list):
            for i in range(len(obj) - 1, -1, -1):
                stack.append((f"{prefix}[{i}]", obj[i]))
        elif obj is not None:
            pairs.append((prefix or "value", str(obj)))
    return pairs


def extract_dataset_records(root_dir: Path) -> list[ExtractedRecord]:
    """Extract attacker evidence per record from either raw input/ or sanitized/."""
    out: list[ExtractedRecord] = []
    for p in sorted(root_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root_dir).as_posix()
        raw = p.read_bytes()
        text = None
        if raw.startswith(b"\xef\xbb\xbf"):
            text = raw[3:].decode("utf-8", errors="ignore")
        elif raw.startswith((b"\xff\xfe", b"\xfe\xff")):
            text = raw.decode("utf-16", errors="ignore")
        else:
            try:
                text = raw.decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                text = raw.decode("cp1252", errors="ignore")
        ext = p.suffix.lower()
        if ext in (".csv", ".tsv"):
            delim = "\t" if ext == ".tsv" else ","
            try:
                rows = list(csv.reader(io.StringIO(text, newline=""), delimiter=delim))
            except csv.Error:
                continue
            if len(rows) < 2:
                continue
            hdr = rows[0]
            for i, row in enumerate(rows[1:]):
                pairs = [(hdr[c] if c < len(hdr) else f"col_{c}", row[c]) for c in range(len(row))]
                out.append(_parse_kv_evidence(pairs, rel, f"row:{i}"))
        elif ext == ".json":
            try:
                obj = json.loads(text)
            except Exception:
                continue
            if isinstance(obj, dict) and obj.get("resourceType") != "Bundle":
                arr_keys = [k for k, v in obj.items() if isinstance(v, list) and v and all(isinstance(x, dict) for x in v)]
                if arr_keys:
                    for ak in arr_keys:
                        for i, item in enumerate(obj[ak]):
                            pairs = _flatten_json_pairs(item)
                            out.append(_parse_kv_evidence(pairs, rel, f"/{ak}/{i}"))
                    continue
            pairs = _flatten_json_pairs(obj)
            out.append(_parse_kv_evidence(pairs, rel, "doc"))
        elif ext == ".txt":
            out.append(_parse_kv_evidence([("text", text)], rel, "doc"))
    return out
