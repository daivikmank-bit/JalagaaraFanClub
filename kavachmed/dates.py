"""Date/time finder, per-source date-order oracle, and format-preserving shift (D1, D2)."""
from __future__ import annotations

import datetime as dt
import re
from typing import Any, Literal

from kavachmed.contracts import DateRef, Field, Record

_DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_ASCII_TO_DEVA = str.maketrans("0123456789", "०१२३४५६७८९")

_MONTH_ABBR = [
    "", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]
_MONTH_FULL = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
_MONTH_HINDI = [
    "", "जनवरी", "फरवरी", "मार्च", "अप्रैल", "मई", "जून",
    "जुलाई", "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर",
]

_MONTH_MAP: dict[str, int] = {}
for _i in range(1, 13):
    _MONTH_MAP[_MONTH_ABBR[_i].lower()] = _i
    _MONTH_MAP[_MONTH_FULL[_i].lower()] = _i
    _MONTH_MAP[_MONTH_HINDI[_i]] = _i
_MONTH_MAP["sept"] = 9
_MONTH_MAP["सितम्बर"] = 9
_MONTH_MAP["नवम्बर"] = 11
_MONTH_MAP["दिसम्बर"] = 12
_MONTH_MAP["फ़रवरी"] = 2

_MONTH_NAMES_RE = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?|"
    r"जनवरी|फरवरी|फ़रवरी|मार्च|अप्रैल|मई|जून|जुलाई|अगस्त|सितंबर|सितम्बर|अक्टूबर|नवंबर|नवम्बर|दिसंबर|दिसम्बर)"
)

_TIME_TZ_OPT = (
    r"(?:[T,\s]+(?P<time>\d{1,2}:\d{2}(?::\d{2}(?:\.\d{1,6})?)?(?:\s*(?:am|pm|AM|PM))?)"
    r"(?:\s*(?P<tz>Z|[+-]\d{2}:?\d{2}|IST|UTC|GMT))?)?"
)

_ISO_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(?P<y>(?:19|20)\d{2})-(?P<m>\d{1,2})-(?P<d>\d{1,2})" + _TIME_TZ_OPT + r"(?![A-Za-z0-9_-])"
)
_DMY_NAMED_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?P<d>[0-9०-९]{1,2})[\s-]+(?P<mon>" + _MONTH_NAMES_RE + r")[\s,-]+(?P<y>(?:19|20|१९|२०)?[0-9०-९]{2})(?![A-Za-z0-9_])"
    + _TIME_TZ_OPT,
    re.IGNORECASE,
)
_MDY_NAMED_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?P<mon>" + _MONTH_NAMES_RE + r")\s+(?P<d>[0-9०-९]{1,2}),?\s+(?P<y>(?:19|20|१९|२०)?[0-9०-९]{2})(?![A-Za-z0-9_])"
    + _TIME_TZ_OPT,
    re.IGNORECASE,
)
_NUM_SEP_RE = re.compile(
    r"(?<![A-Za-z0-9_./-])(?P<a>[0-9०-९]{1,2})(?P<sep>[/.-])(?P<b>[0-9०-९]{1,2})(?P=sep)(?P<y>(?:19|20|१९|२०)[0-9०-९]{2}|[0-9०-९]{2})"
    + _TIME_TZ_OPT
    + r"(?![A-Za-z0-9_./-])"
)
_YEARLESS_DM_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?P<d>[0-9०-९]{1,2})\s+(?P<mon>" + _MONTH_NAMES_RE + r")(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_YEARLESS_MD_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?P<mon>" + _MONTH_NAMES_RE + r")\s+(?P<d>[0-9०-९]{1,2})(?![A-Za-z0-9_])",
    re.IGNORECASE,
)

_DOB_HINT_RE = re.compile(r"\b(?:dob|d\.o\.b|birth\s*date|birthdate|janm\s*tithi|born\s*on|जन्म\s*तिथि)\b", re.IGNORECASE)
_MDY_HEADER_RE = re.compile(r"mm\s*/\s*dd|mm\s*-\s*dd", re.IGNORECASE)
_DMY_HEADER_RE = re.compile(r"dd\s*/\s*mm|dd\s*-\s*mm", re.IGNORECASE)
_NEG_CONTEXT_BEFORE_RE = re.compile(r"(?:bp|blood\s*pressure|ratio|ipd|opd|uhid|mrn|reg|adm|pin|loinc|icd)\s*[:#/-]?\s*$", re.IGNORECASE)
_NEG_CONTEXT_AFTER_RE = re.compile(r"^\s*(?:mmhg|/min|mg|mcg|ml|iu|yrs?|years?|wks?|weeks?|months?|days?)\b", re.IGNORECASE)


def _to_ascii_digits(s: str) -> tuple[str, bool]:
    translated = s.translate(_DEVA_DIGITS)
    return translated, translated != s


def _expand_two_digit_year(y2: int, ref_year: int = 2026) -> int:
    pivot = (ref_year % 100) + 10
    return (2000 + y2) if y2 <= pivot else (1900 + y2)


def _valid_ymd(y: int, m: int, d: int) -> bool:
    if not (1900 <= y <= 2100 and 1 <= m <= 12 and 1 <= d <= 31):
        return False
    try:
        dt.date(y, m, d)
        return True
    except ValueError:
        return False


def _hospital_from_record(rec: Record) -> str | None:
    h = rec.meta.get("hospital")
    if h:
        return str(h)
    low = rec.key[0].lower()
    if "arogya" in low or "/ass" in low or "_ass" in low:
        return "ASS"
    if "sanjeevani" in low or "/snj" in low or "_snj" in low:
        return "SNJ"
    if "jeevan" in low or "/jjh" in low or "_jjh" in low:
        return "JJH"
    if "hardoi" in low or "/dhh" in low or "_dhh" in low:
        return "DHH"
    return None


def _policy_from_ctx(ctx: Any) -> dict[str, Any]:
    if ctx is None:
        return {}
    if isinstance(ctx, dict):
        pol = ctx.get("policy")
        return pol if isinstance(pol, dict) else ctx
    pol = getattr(ctx, "policy", None)
    return pol if isinstance(pol, dict) else {}


def infer_date_orders(records: list[Record], policy: dict[str, Any]) -> dict[str, Literal["dmy", "mdy"]]:
    """Infer MM/DD vs DD/MM per source path (with hospital stickiness and header hints)."""
    default_ord: Literal["dmy", "mdy"] = (
        "mdy" if str((policy.get("dates") or {}).get("default_order", "dmy")).lower() == "mdy" else "dmy"
    )
    by_file: dict[str, dict[str, Any]] = {}
    hosp_votes: dict[str, list[int]] = {}

    for rec in records:
        rel = rec.key[0]
        st = by_file.setdefault(rel, {"dmy": 0, "mdy": 0, "hint": None, "hospital": _hospital_from_record(rec)})
        for h in rec.meta.get("header") or []:
            hs = str(h)
            if _MDY_HEADER_RE.search(hs):
                st["hint"] = "mdy"
            elif _DMY_HEADER_RE.search(hs):
                st["hint"] = "dmy"
        for fld in rec.fields:
            if fld.vtype in ("header", "key", "path"):
                if _MDY_HEADER_RE.search(fld.text):
                    st["hint"] = "mdy"
                elif _DMY_HEADER_RE.search(fld.text):
                    st["hint"] = "dmy"
                continue
            if _MDY_HEADER_RE.search(fld.path):
                st["hint"] = "mdy"
            elif _DMY_HEADER_RE.search(fld.path):
                st["hint"] = "dmy"
            for m in _NUM_SEP_RE.finditer(fld.text):
                a_s, _ = _to_ascii_digits(m.group("a"))
                b_s, _ = _to_ascii_digits(m.group("b"))
                a, b = int(a_s), int(b_s)
                if a > 12 and 1 <= b <= 12 and a <= 31:
                    st["dmy"] += 1
                    if st["hospital"]:
                        hosp_votes.setdefault(st["hospital"], [0, 0])[0] += 1
                elif b > 12 and 1 <= a <= 12 and b <= 31:
                    st["mdy"] += 1
                    if st["hospital"]:
                        hosp_votes.setdefault(st["hospital"], [0, 0])[1] += 1

    orders: dict[str, Literal["dmy", "mdy"]] = {}
    for rel, st in by_file.items():
        if st["hint"] in ("dmy", "mdy"):
            orders[rel] = st["hint"]
        elif st["mdy"] > st["dmy"]:
            orders[rel] = "mdy"
        elif st["dmy"] > st["mdy"]:
            orders[rel] = "dmy"
        elif st["hospital"] == "ASS":
            orders[rel] = "mdy"
        elif st["hospital"] in ("SNJ", "JJH", "DHH"):
            orders[rel] = "dmy"
        elif st["hospital"] and st["hospital"] in hosp_votes:
            dv, mv = hosp_votes[st["hospital"]]
            orders[rel] = "mdy" if mv > dv else "dmy"
        else:
            orders[rel] = default_ord
    return orders


def _is_dob_context(path: str, text: str, start: int) -> bool:
    if _DOB_HINT_RE.search(path):
        return True
    window = text[max(0, start - 32) : start]
    return bool(_DOB_HINT_RE.search(window))


def _precision_from_time(time_part: str, tz_part: str) -> Literal["date", "datetime", "instant"]:
    if tz_part and time_part:
        return "instant"
    if time_part:
        return "datetime"
    return "date"


def find(
    rec: Record,
    ctx: Any = None,
    order_override: Literal["dmy", "mdy"] | None = None,
    ref_year: int | None = None,
) -> list[DateRef]:
    """Find all date/datetime values inside a Record without false-positives on clinical numerics."""
    pol = _policy_from_ctx(ctx)
    dates_cfg = pol.get("dates") or {}
    default_ord = "mdy" if str(dates_cfg.get("default_order", "dmy")).lower() == "mdy" else "dmy"
    if order_override:
        order = order_override
    elif rec.meta.get("date_order") in ("dmy", "mdy"):
        order = rec.meta["date_order"]
    elif _hospital_from_record(rec) == "ASS":
        order = "mdy"
    else:
        order = default_ord

    assumed_year = int(ref_year or rec.meta.get("assumed_year") or dates_cfg.get("dataset_ref_year", 2026))
    found: list[DateRef] = []

    for fld in rec.fields:
        if fld.vtype in ("key", "header", "path"):
            continue
        text = fld.text
        if not text:
            continue
        occupied: list[tuple[int, int]] = []

        def _overlaps(s: int, e: int) -> bool:
            return any(os < e and s < oe for os, oe in occupied)

        # 1. ISO YYYY-MM-DD[T...]
        for m in _ISO_RE.finditer(text):
            s, e = m.start(), m.end()
            if _overlaps(s, e):
                continue
            y, mo, d = int(m.group("y")), int(m.group("m")), int(m.group("d"))
            iso_fmt = "iso"
            if not _valid_ymd(y, mo, d):
                # unseen-format sources write YYYY-DD-MM (middle part > 12)
                if mo > 12 and _valid_ymd(y, d, mo):
                    mo, d, iso_fmt = d, mo, "ydm"
                else:
                    continue
            tp = m.group("time") or ""
            tz = m.group("tz") or ""
            date_end = m.end("d")
            t_sep = text[date_end : m.start("time")] if tp else ""
            tz_sep = text[m.end("time") : m.start("tz")] if (tp and tz) else ""
            occupied.append((s, e))
            found.append(
                DateRef(
                    key=rec.key, field_idx=fld.idx, start=s, end=e, raw=text[s:e],
                    year=y, month=mo, day=d, has_year=True,
                    precision=_precision_from_time(tp, tz), fmt=iso_fmt, sep="-",
                    time_part=(t_sep + tp) if tp else "", tz_part=(tz_sep + tz) if tz else "",
                    is_dob=_is_dob_context(fld.path, text, s),
                )
            )

        # 2. Named month with year: 12 Apr 1977 / 12-Apr-1977
        for m in _DMY_NAMED_RE.finditer(text):
            s, e = m.start(), m.end()
            if _overlaps(s, e):
                continue
            d_s, deva_d = _to_ascii_digits(m.group("d"))
            y_s, deva_y = _to_ascii_digits(m.group("y"))
            mon_raw = m.group("mon")
            mo = _MONTH_MAP.get(mon_raw.lower()) or _MONTH_MAP.get(mon_raw)
            if not mo:
                continue
            two_dig = len(y_s) == 2
            y = _expand_two_digit_year(int(y_s), assumed_year) if two_dig else int(y_s)
            d = int(d_s)
            if not _valid_ymd(y, mo, d):
                continue
            tp = m.group("time") or ""
            tz = m.group("tz") or ""
            t_sep = text[m.end("y") : m.start("time")] if tp else ""
            tz_sep = text[m.end("time") : m.start("tz")] if (tp and tz) else ""
            sep = "-" if "-" in text[m.end("d") : m.start("y")] else " "
            m_style = "hindi" if any("\u0900" <= c <= "\u097f" for c in mon_raw) else ("full" if len(mon_raw) > 4 else "abbr")
            occupied.append((s, e))
            found.append(
                DateRef(
                    key=rec.key, field_idx=fld.idx, start=s, end=e, raw=text[s:e],
                    year=y, month=mo, day=d, has_year=True,
                    precision=_precision_from_time(tp, tz), fmt="dmy_named", sep=sep,
                    two_digit_year=two_dig, month_name_style=m_style,
                    time_part=(t_sep + tp) if tp else "", tz_part=(tz_sep + tz) if tz else "",
                    devanagari=deva_d or deva_y or m_style == "hindi",
                    is_dob=_is_dob_context(fld.path, text, s),
                )
            )

        # 3. Named month MDY: Apr 12, 1977
        for m in _MDY_NAMED_RE.finditer(text):
            s, e = m.start(), m.end()
            if _overlaps(s, e):
                continue
            d_s, deva_d = _to_ascii_digits(m.group("d"))
            y_s, deva_y = _to_ascii_digits(m.group("y"))
            mon_raw = m.group("mon")
            mo = _MONTH_MAP.get(mon_raw.lower()) or _MONTH_MAP.get(mon_raw)
            if not mo:
                continue
            two_dig = len(y_s) == 2
            y = _expand_two_digit_year(int(y_s), assumed_year) if two_dig else int(y_s)
            d = int(d_s)
            if not _valid_ymd(y, mo, d):
                continue
            tp = m.group("time") or ""
            tz = m.group("tz") or ""
            t_sep = text[m.end("y") : m.start("time")] if tp else ""
            tz_sep = text[m.end("time") : m.start("tz")] if (tp and tz) else ""
            m_style = "hindi" if any("\u0900" <= c <= "\u097f" for c in mon_raw) else ("full" if len(mon_raw) > 4 else "abbr")
            has_comma = "," in text[m.end("d") : m.start("y")]
            occupied.append((s, e))
            found.append(
                DateRef(
                    key=rec.key, field_idx=fld.idx, start=s, end=e, raw=text[s:e],
                    year=y, month=mo, day=d, has_year=True,
                    precision=_precision_from_time(tp, tz), fmt="mdy_named",
                    sep="," if has_comma else " ", two_digit_year=two_dig,
                    month_name_style=m_style,
                    time_part=(t_sep + tp) if tp else "", tz_part=(tz_sep + tz) if tz else "",
                    devanagari=deva_d or deva_y or m_style == "hindi",
                    is_dob=_is_dob_context(fld.path, text, s),
                )
            )

        # 4. Numeric separated DD/MM/YYYY, MM/DD/YYYY, DD-MM-YYYY, DD.MM.YY, Devanagari
        for m in _NUM_SEP_RE.finditer(text):
            s, e = m.start(), m.end()
            if _overlaps(s, e):
                continue
            before = text[max(0, s - 24) : s]
            after = text[e : min(len(text), e + 16)]
            if _NEG_CONTEXT_BEFORE_RE.search(before) or _NEG_CONTEXT_AFTER_RE.search(after):
                continue
            a_s, deva_a = _to_ascii_digits(m.group("a"))
            b_s, deva_b = _to_ascii_digits(m.group("b"))
            y_s, deva_y = _to_ascii_digits(m.group("y"))
            a, b = int(a_s), int(b_s)
            two_dig = len(y_s) == 2
            if len(m.group("y")) == 1:
                continue
            if m.group("sep") == "-" and two_dig and a <= 3 and b <= 3 and int(y_s) <= 3:
                continue
            y = _expand_two_digit_year(int(y_s), assumed_year) if two_dig else int(y_s)
            if a > 12 and 1 <= b <= 12:
                d, mo, fmt = a, b, "dmy_num"
            elif b > 12 and 1 <= a <= 12:
                mo, d, fmt = a, b, "mdy_num"
            elif order == "mdy":
                mo, d, fmt = a, b, "mdy_num"
            else:
                d, mo, fmt = a, b, "dmy_num"
            if not _valid_ymd(y, mo, d):
                continue
            tp = m.group("time") or ""
            tz = m.group("tz") or ""
            t_sep = text[m.end("y") : m.start("time")] if tp else ""
            tz_sep = text[m.end("time") : m.start("tz")] if (tp and tz) else ""
            padded = len(a_s) == 2 and len(b_s) == 2
            occupied.append((s, e))
            found.append(
                DateRef(
                    key=rec.key, field_idx=fld.idx, start=s, end=e, raw=text[s:e],
                    year=y, month=mo, day=d, has_year=True,
                    precision=_precision_from_time(tp, tz),
                    fmt=fmt + ("_pad" if padded else "_nopad"),
                    sep=m.group("sep"), two_digit_year=two_dig,
                    time_part=(t_sep + tp) if tp else "", tz_part=(tz_sep + tz) if tz else "",
                    devanagari=deva_a or deva_b or deva_y,
                    is_dob=_is_dob_context(fld.path, text, s),
                )
            )

        # 5. Year-less dates: 5 September / ५ सितंबर / September 5
        for pat, fmt_name in ((_YEARLESS_DM_RE, "d_mon"), (_YEARLESS_MD_RE, "mon_d")):
            for m in pat.finditer(text):
                s, e = m.start(), m.end()
                if _overlaps(s, e):
                    continue
                d_s, deva_d = _to_ascii_digits(m.group("d"))
                mon_raw = m.group("mon")
                mo = _MONTH_MAP.get(mon_raw.lower()) or _MONTH_MAP.get(mon_raw)
                if not mo:
                    continue
                d = int(d_s)
                if not _valid_ymd(assumed_year, mo, d):
                    continue
                m_style = "hindi" if any("\u0900" <= c <= "\u097f" for c in mon_raw) else ("full" if len(mon_raw) > 4 else "abbr")
                occupied.append((s, e))
                found.append(
                    DateRef(
                        key=rec.key, field_idx=fld.idx, start=s, end=e, raw=text[s:e],
                        year=assumed_year, month=mo, day=d, has_year=False,
                        precision="year_less", fmt=fmt_name, sep=" ",
                        month_name_style=m_style,
                        devanagari=deva_d or m_style == "hindi",
                        is_dob=_is_dob_context(fld.path, text, s),
                    )
                )

    return sorted(found, key=lambda r: (r.field_idx, r.start, r.end))


def _format_month(month: int, style: str) -> str:
    if style == "hindi":
        return _MONTH_HINDI[month]
    if style == "full":
        return _MONTH_FULL[month]
    return _MONTH_ABBR[month]


def shift(ref: DateRef | str, days: int, resolved_year: int | None = None) -> str:
    """Shift a DateRef OR raw date string by `days`, preserving format, script, time, tz."""
    if isinstance(ref, str):
        tmp = Record(key=("_", "doc"), fields=[Field(idx=0, path="date", text=ref)])
        refs = find(tmp)
        if not refs:
            return ""
        ref = refs[0]

    base_y = resolved_year if (not ref.has_year and resolved_year) else ref.year
    try:
        base_dt = dt.date(base_y, ref.month, ref.day)
    except ValueError:
        base_dt = dt.date(base_y, ref.month, min(ref.day, 28))
    shifted = base_dt + dt.timedelta(days=days)

    if ref.fmt == "iso":
        return f"{shifted.year:04d}-{shifted.month:02d}-{shifted.day:02d}{ref.time_part}{ref.tz_part}"
    if ref.fmt == "ydm":
        return f"{shifted.year:04d}-{shifted.day:02d}-{shifted.month:02d}{ref.time_part}{ref.tz_part}"

    if ref.fmt == "dmy_named":
        mon = _format_month(shifted.month, ref.month_name_style)
        y_str = f"{shifted.year % 100:02d}" if ref.two_digit_year else f"{shifted.year:04d}"
        d_str = f"{shifted.day}"
        if ref.devanagari:
            d_str = d_str.translate(_ASCII_TO_DEVA)
            y_str = y_str.translate(_ASCII_TO_DEVA)
        return f"{d_str}{ref.sep}{mon}{ref.sep}{y_str}{ref.time_part}{ref.tz_part}"

    if ref.fmt == "mdy_named":
        mon = _format_month(shifted.month, ref.month_name_style)
        y_str = f"{shifted.year % 100:02d}" if ref.two_digit_year else f"{shifted.year:04d}"
        d_str = f"{shifted.day}"
        if ref.devanagari:
            d_str = d_str.translate(_ASCII_TO_DEVA)
            y_str = y_str.translate(_ASCII_TO_DEVA)
        mid = ", " if ref.sep == "," else " "
        return f"{mon} {d_str}{mid}{y_str}{ref.time_part}{ref.tz_part}"

    if ref.fmt in ("d_mon", "mon_d"):
        mon = _format_month(shifted.month, ref.month_name_style)
        d_str = f"{shifted.day}"
        if ref.devanagari:
            d_str = d_str.translate(_ASCII_TO_DEVA)
        return f"{d_str} {mon}" if ref.fmt == "d_mon" else f"{mon} {d_str}"

    padded = ref.fmt.endswith("_pad")
    d_s = f"{shifted.day:02d}" if padded else f"{shifted.day}"
    m_s = f"{shifted.month:02d}" if padded else f"{shifted.month}"
    y_s = f"{shifted.year % 100:02d}" if ref.two_digit_year else f"{shifted.year:04d}"
    if ref.devanagari:
        d_s = d_s.translate(_ASCII_TO_DEVA)
        m_s = m_s.translate(_ASCII_TO_DEVA)
        y_s = y_s.translate(_ASCII_TO_DEVA)
    if ref.fmt.startswith("mdy_num"):
        return f"{m_s}{ref.sep}{d_s}{ref.sep}{y_s}{ref.time_part}{ref.tz_part}"
    return f"{d_s}{ref.sep}{m_s}{ref.sep}{y_s}{ref.time_part}{ref.tz_part}"


def make_shifter(order: Literal["dmy", "mdy"], ref_year: int | None = None) -> Any:
    """Return ``fn(text, days) -> str`` that parses `text` in the file's date order.

    ``shift(str)`` alone re-parses with the default (DD/MM) order, which would swap
    day and month on MM/DD sources (Arogya). The edit builder passes raw strings,
    so the pipeline hands it this per-file closure instead.
    """
    def _shift_text(text: str, days: int) -> str:
        tmp = Record(key=("_", "doc"), fields=[Field(idx=0, path="date", text=text)])
        refs = find(tmp, order_override=order, ref_year=ref_year)
        if not refs:
            return ""
        return shift(refs[0], days, resolved_year=ref_year)
    return _shift_text


def resolve_year_less_year(ref: DateRef, family_years: list[int], default_ref_year: int = 2026) -> int:
    """Choose nearest year from family's dated events, else dataset_ref_year (D2)."""
    if ref.has_year:
        return ref.year
    if not family_years:
        return default_ref_year
    counts: dict[int, int] = {}
    for y in family_years:
        counts[y] = counts.get(y, 0) + 1
    best_y = sorted(counts.items(), key=lambda kv: (-kv[1], abs(kv[0] - default_ref_year), kv[0]))[0][0]
    return best_y
