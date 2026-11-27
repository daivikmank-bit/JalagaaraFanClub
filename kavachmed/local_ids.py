"""Whole-value issuer-local record number grammars, independent of linkage trust."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_FLAGS = re.ASCII | re.IGNORECASE
_EPOCH = (
    r"(?P<prefix>[A-Z]{2,8})(?:(?P<sep1>[-./])(?P<series>[A-Z]{1,8}))?"
    r"(?P<sep2>[-./])(?P<epoch>[0-9]{2,4})(?P<sep3>[-./])(?P<serial>[0-9]{3,16})"
)
_EPOCH_ID = re.compile(_EPOCH, _FLAGS)
_ATTACHED_EPOCH_ID = re.compile(r"([A-Z]{2,8})([0-9]{4})([-/])([0-9]{4,16})", _FLAGS)
_COMPACT_ID = re.compile(r"([A-Z]{2,8})([-/]?)([0-9]{4,16})", _FLAGS)
LABELED_LOCAL = re.compile(
    r"(?:\b(?:UHID|MRN|patient\s*(?:id|no)|ref\.?\s*(?:no\.?)?|reg\.?\s*(?:no\.?)?)|पंजीकरण\s*(?:सं\.?|संख्या)?)\s*[:#]?\s*"
    r"(?P<value>" + _EPOCH + r"|[A-Z]{2,8}[-/]?[0-9]{4,16})(?![A-Z0-9])",
    _FLAGS,
)
_COMPONENT_ID = re.compile(
    r"(?<![A-Z0-9])(?:" + _EPOCH + r"|[A-Z]{2,8}[-]?[0-9]{5,16}|"
    r"(?:adm|ipd|opd|mrn|uhid)[-.][0-9]{2,4}[-.][0-9]{3,16}|"
    r"[A-Z]{2,8}[-.](?:ip|op|adm)[-.][0-9]{3,16})(?![A-Z0-9])",
    _FLAGS,
)
GENERIC_PREFIXES = frozenset({
    "uhid", "mrn", "id", "regno", "reg", "crno", "cr", "no", "pid", "patid",
})
PATIENT_ID_HEADERS = frozenset({
    "mrn", "mrno", "mrnumber", "medicalrecordnumber", "uhid", "uhidno", "patientid",
    "patientidentifier", "patientno", "patientnumber", "hospitalnumber", "hospno", "patient.identifier",
    # Common short aliases used in HMIS exports
    "ref", "refno", "regid", "regno", "reg_no", "patref", "pid", "ptid", "ptno",
    "hospid", "hosp_id", "hosp_no", "hospno",
})
_NONPATIENT_PREFIXES = frozenset({
    "ipd", "opd", "adm", "clm", "claim", "pol", "pmjay", "lab", "pkg", "icd", "loinc",
})


@dataclass(frozen=True, slots=True)
class LocalID:
    prefix: str
    core: str
    # A format is an issuer prefix, series, separators, and digit widths. Explicit
    # declarations for one format do not authorize another prefix or series.
    format: tuple[str, ...]

    @property
    def issuer_bearing(self) -> bool:
        return self.prefix not in GENERIC_PREFIXES | _NONPATIENT_PREFIXES


def parse_local_id(value: str) -> LocalID | None:
    if len(value) > 80:
        return None
    folded = unicodedata.normalize("NFKC", value).strip()
    if not folded.isascii():
        folded = "".join(str(unicodedata.decimal(c)) if c.isdecimal() else c for c in folded)
    m = _EPOCH_ID.fullmatch(folded)
    if m:
        prefix, sep1, series, sep2, epoch, sep3, serial = m.groups()
        prefix = prefix.casefold()
        # Preserve series, separators, epoch and leading zeros: equality between
        # two representations needs an explicit issuer rule.
        core = folded[len(prefix):]
        fmt = (prefix, sep1 or "", series or "", sep2, str(len(epoch)), sep3, str(len(serial)))
        return LocalID(prefix, core, fmt)
    m = _ATTACHED_EPOCH_ID.fullmatch(folded)
    if m:
        prefix, epoch, sep, serial = m.groups()
        prefix = prefix.casefold()
        return LocalID(prefix, epoch + sep + serial, (prefix, "attached", sep, str(len(serial))))
    m = _COMPACT_ID.fullmatch(folded)
    if m:
        prefix, sep, serial = m.groups()
        prefix = prefix.casefold()
        return LocalID(prefix, serial, (prefix, sep, str(len(serial))))
    return None


def patient_id_header(path: str) -> bool:
    name = path.rpartition("/")[2].casefold()
    return re.sub(r"[\s_-]", "", name) in PATIENT_ID_HEADERS


def contains_local_id(component: str) -> bool:
    return _COMPONENT_ID.search(component) is not None
