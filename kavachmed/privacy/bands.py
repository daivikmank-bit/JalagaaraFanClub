"""Age extraction and truthful patient-level age banding (D3, FR-P3)."""
from __future__ import annotations

import re
from typing import Any

from kavachmed.contracts import Evidence, RecKey, Record

_DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

DEFAULT_LATTICE_10 = [0, 5, 10, 18, 30, 40, 50, 60, 70, 85]
LATTICE_20 = [0, 18, 30, 50, 70, 85]

_AGE_STRUCTURED_KEYS = re.compile(
    r"^(?:age|age_yrs|age_years|age_sex|umar|aayu|आयु|उम्र)$",
    re.IGNORECASE,
)
_AGE_SLASH_SEX_RE = re.compile(
    r"(?<![A-Za-z0-9-])(?P<age>[0-9०-९]{1,3})\s*(?:Y(?:rs?|ears?)?|वर्ष|साल)?\s*[/,]\s*(?P<sex>Male|Female|पुरुष|महिला|[MFmf])(?![A-Za-z0-9])"
)
_AGE_ANCHOR_RE = re.compile(
    r"\b(?:Age|Umar|Aayu|आयु|उम्र)\s*[:=\-]?\s*(?P<age>[0-9०-९]{1,3})\s*(?:Y(?:rs?|ears?)?|वर्ष|साल)?\b",
    re.IGNORECASE,
)
_AGE_MONTHS_ANCHOR_RE = re.compile(
    r"\b(?:Age|Umar|Aayu|आयु|उम्र)\s*[:=\-]?\s*(?P<mo>[0-9०-९]{1,2})\s*(?:m(?:onths?|os?)?|माह|महीने)\b",
    re.IGNORECASE,
)
_DURATION_BEFORE_RE = re.compile(
    r"(?:x|for|since|back|history\s+of|h/o|k/c/o|से|तक)\s*$",
    re.IGNORECASE,
)
_DURATION_AFTER_RE = re.compile(
    r"^\s*(?:yrs?\s+back|years?\s+back|yrs?\s+ago|years?\s+ago)\b",
    re.IGNORECASE,
)


def _parse_int(s: str) -> int | None:
    ascii_s = s.translate(_DEVA_DIGITS).strip()
    if ascii_s.isdigit():
        val = int(ascii_s)
        if 0 <= val <= 120:
            return val
    return None


def extract_ages_from_record(rec: Record) -> list[int]:
    """Extract true patient ages from structured fields and free text, ignoring durations."""
    ages: list[int] = []
    for fld in rec.fields:
        if fld.vtype in ("key", "header", "path"):
            continue
        txt = fld.text.strip()
        if not txt:
            continue

        col_name = fld.path.split("/")[-1].strip()
        if _AGE_STRUCTURED_KEYS.match(col_name):
            if re.fullmatch(r"[0-9०-९]{1,3}(?:\s*(?:Y(?:rs?|ears?)?|वर्ष|साल))?", txt, re.IGNORECASE):
                num_m = re.match(r"([0-9०-९]{1,3})", txt)
                if num_m:
                    val = _parse_int(num_m.group(1))
                    if val is not None:
                        ages.append(val)
                        continue
            if re.fullmatch(r"[0-9०-९]{1,2}\s*(?:m(?:onths?|os?)?|माह|महीने)", txt, re.IGNORECASE):
                ages.append(0)
                continue

        for m in _AGE_SLASH_SEX_RE.finditer(txt):
            before = txt[max(0, m.start() - 16) : m.start()]
            if _DURATION_BEFORE_RE.search(before):
                continue
            val = _parse_int(m.group("age"))
            if val is not None:
                ages.append(val)

        for m in _AGE_ANCHOR_RE.finditer(txt):
            before = txt[max(0, m.start() - 16) : m.start()]
            after = txt[m.end() : min(len(txt), m.end() + 16)]
            if _DURATION_BEFORE_RE.search(before) or _DURATION_AFTER_RE.search(after):
                continue
            val = _parse_int(m.group("age"))
            if val is not None:
                ages.append(val)

        for _m in _AGE_MONTHS_ANCHOR_RE.finditer(txt):
            ages.append(0)

    return ages


def _cell_bounds(age: int, lattice: list[int], top_band: str = "85+") -> tuple[int, int | None, str]:
    pts = sorted(lattice)
    top = pts[-1]
    if age >= top:
        return (top, None, top_band)
    for i in range(len(pts) - 1):
        lo, hi = pts[i], pts[i + 1] - 1
        if lo <= age <= hi:
            return (lo, hi, f"{lo}-{hi}")
    return (0, pts[1] - 1, f"0-{pts[1] - 1}")


def band_for_ages(
    ages: list[int],
    lattice: list[int] | None = None,
    top_band: str = "85+",
    coarse_mode: str = "lattice10",
) -> str:
    """Compute the tightest band from lattice containing EVERY true age in `ages` (100% truthful)."""
    if not ages:
        return "Adult"
    if coarse_mode == "suppress_qi":
        mn, mx = min(ages), max(ages)
        if mx < 18:
            return "Child"
        if mn >= 60:
            return "Senior"
        return "Adult"

    lat = LATTICE_20 if coarse_mode == "age20" else (lattice or DEFAULT_LATTICE_10)
    mn, mx = min(ages), max(ages)
    lo_min, hi_min, label_min = _cell_bounds(mn, lat, top_band=top_band)
    lo_max, hi_max, _label_max = _cell_bounds(mx, lat, top_band=top_band)
    if (lo_min, hi_min) == (lo_max, hi_max):
        return label_min
    if hi_max is None:
        return f"{lo_min}+"
    return f"{lo_min}-{hi_max}"


def compute_cluster_bands(
    cluster_of: dict[RecKey, str | None],
    evidences: list[Evidence],
    records_by_key: dict[RecKey, Record],
    policy: dict[str, Any],
    coarse_by_cluster: dict[str, str] | None = None,
    extra_ages_by_key: dict[RecKey, list[int]] | None = None,
) -> dict[str, str]:
    """Compute one truthful patient-level band per cluster across ALL of its records."""
    age_cfg = policy.get("age") or {}
    lattice = list(age_cfg.get("pool_lattice") or DEFAULT_LATTICE_10)
    top_band = str(age_cfg.get("top_band", "85+"))
    ev_by_key = {ev.key: ev for ev in evidences}
    ref_yr = int((policy.get("dates") or {}).get("dataset_ref_year", 2026))

    all_keys = sorted(set(cluster_of) | set(records_by_key) | set(ev_by_key) | set(extra_ages_by_key or {}))
    ages_by_cluster: dict[str, list[int]] = {}
    for rec_key in all_keys:
        cid = cluster_of.get(rec_key) or f"orphan:{rec_key[0]}|{rec_key[1]}"
        rec = records_by_key.get(rec_key)
        ev = ev_by_key.get(rec_key)
        found = extract_ages_from_record(rec) if rec is not None else []
        if extra_ages_by_key and rec_key in extra_ages_by_key:
            found.extend(extra_ages_by_key[rec_key])
        if ev and getattr(ev, "age", None) is not None:
            found.append(int(ev.age))  # type: ignore[arg-type]
        if ev and ev.birth_years:
            by_lo, by_hi = ev.birth_years
            mid_age = max(0, min(120, ref_yr - ((by_lo + by_hi) // 2)))
            found.append(mid_age)
        if found:
            ages_by_cluster.setdefault(cid, []).extend(found)

    out: dict[str, str] = {}
    for cid, age_list in sorted(ages_by_cluster.items()):
        mode = (coarse_by_cluster or {}).get(cid, "lattice10")
        out[cid] = band_for_ages(age_list, lattice=lattice, top_band=top_band, coarse_mode=mode)
    return out


def replace_ages_in_text(text: str, band: str, is_structured_age_field: bool = False) -> str:
    """Replace age mentions in a string with `band` while preserving sex/units and skipping durations."""
    if is_structured_age_field:
        if re.fullmatch(r"\s*[0-9०-९]{1,3}\s*", text):
            return band
        m_only = re.fullmatch(r"\s*[0-9०-९]{1,3}(\s*(?:Y(?:rs?|ears?)?|वर्ष|साल))\s*", text, re.IGNORECASE)
        if m_only:
            return f"{band}{m_only.group(1)}"

    def _sub_slash(m: re.Match[str]) -> str:
        before = text[max(0, m.start() - 16) : m.start()]
        if _DURATION_BEFORE_RE.search(before):
            return m.group(0)
        raw = m.group(0)
        age_tok = m.group("age")
        idx = raw.find(age_tok)
        return raw[:idx] + band + raw[idx + len(age_tok) :]

    out = _AGE_SLASH_SEX_RE.sub(_sub_slash, text)

    def _sub_anchor(m: re.Match[str]) -> str:
        before = out[max(0, m.start() - 16) : m.start()]
        after = out[m.end() : min(len(out), m.end() + 16)]
        if _DURATION_BEFORE_RE.search(before) or _DURATION_AFTER_RE.search(after):
            return m.group(0)
        raw = m.group(0)
        age_tok = m.group("age")
        idx = raw.rfind(age_tok)
        return raw[:idx] + band + raw[idx + len(age_tok) :]

    return _AGE_ANCHOR_RE.sub(_sub_anchor, out)
