"""Frozen interface contracts for KavachMed (Phase 1 / M1)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

RecKey = tuple[str, str]
# (input_relative_path, loc)
# loc = "row:12" | "/visits/3" | "doc" | "@filename" | "@header" | "@keys" | "@meta"


@dataclass(frozen=True)
class Field:
    """One text-bearing value inside a Record."""
    idx: int
    path: str
    text: str
    vtype: Literal["str", "int", "float", "key", "header", "path", "b64"] = "str"


@dataclass
class Record:
    """Universal Record Model (URM) unit."""
    key: RecKey
    fields: list[Field]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedFile:
    """Result of parsing one input file into URM records + projector."""
    rel_path: str
    sha256: str
    fmt: Literal["csv", "tsv", "json", "fhir", "text", "xml"]
    encoding: str
    newline: str
    records: list[Record]
    raw_bytes: bytes
    # projector(edits, psn_map) -> (output_bytes, LocMap)
    projector: Callable[..., tuple[bytes, "LocMap"]] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FileEntry:
    """Inventory entry produced by inventory.scan()."""
    abs_path: Path
    rel_path: str
    sha256: str
    raw_bytes: bytes
    sniff_fmt: str
    encoding: str | None
    text: str | None
    quarantine_reason: str | None = None


@dataclass(frozen=True)
class Span:
    """A detected or protected character span inside a Record Field."""
    key: RecKey
    field_idx: int
    start: int
    end: int
    typ: str
    role: Literal["patient", "relative", "third_party", "sender", "doctor", "unknown", "none"]
    score: float
    src: Literal[
        "ids", "context", "gazetteer", "default_deny",
        "labels", "crf", "propagate", "shield",
    ]
    protected: bool = False
    exempt_kind: str | None = None
    # exempt_kind in {"drug","dose","lab","code","eponym","ward","history","doctor_registry"}


@dataclass
class Evidence:
    """Normalized identity evidence extracted from one Record."""
    key: RecKey
    names: list[str] = field(default_factory=list)
    kin: list[str] = field(default_factory=list)
    kin_type: str | None = None
    sex: str | None = None
    birth_years: tuple[int, int] | None = None
    ids: dict[str, list[str]] = field(default_factory=dict)
    phones: list[str] = field(default_factory=list)
    phone_shared: bool = False
    geo: dict[str, str] = field(default_factory=dict)
    hospital: str | None = None
    adm_key: str | None = None
    dates: list[str] = field(default_factory=list)
    # Additive optional fields for Owner D privacy controller & utility
    age: int | None = None
    dx_terms: list[str] = field(default_factory=list)
    los_days: int | None = None
    procedure_terms: list[str] = field(default_factory=list)


@dataclass
class Resolution:
    """Output of linkage.resolve()."""
    patient_of: dict[RecKey, str | None]   # "PSN-XXXXXXXX" or None (abstain)
    cluster_of: dict[RecKey, str | None]   # internal cluster id, never emitted
    family_of: dict[RecKey, str]           # temporal family key, always set
    shift_days: dict[str, int]             # family key -> ±[30, 365], never |Δ| < 30


@dataclass
class PrivacyPlan:
    """Output of privacy.plan()."""
    age_band: dict[str, str] = field(default_factory=dict)
    location_level: dict[RecKey, Literal["district", "state", "none"]] = field(default_factory=dict)
    dx_rollup: dict[RecKey, str] = field(default_factory=dict)
    generalized: list[RecKey] = field(default_factory=list)
    qi_kinds: dict[str, int] = field(default_factory=dict)
    # Additive optional metadata for FHIR/controller & risk reporting
    pin_level: dict[RecKey, Literal["none", "pin3"]] = field(default_factory=dict)
    suppress_qi: set[RecKey] = field(default_factory=set)
    risk_summary: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DateRef:
    """Parsed date/time occurrence inside a Record field (D1, D2)."""
    key: RecKey
    field_idx: int
    start: int
    end: int
    raw: str
    year: int
    month: int
    day: int
    has_year: bool
    precision: Literal["date", "datetime", "instant", "year_less"]
    fmt: str
    sep: str = "/"
    two_digit_year: bool = False
    month_name_style: str = ""
    time_part: str = ""
    tz_part: str = ""
    devanagari: bool = False
    is_dob: bool = False


@dataclass(frozen=True)
class Edit:
    """Atomic edit instruction consumed by a format projector."""
    key: RecKey
    field_idx: int
    start: int
    end: int
    kind: Literal["REPLACE", "DROP", "PSN", "SHIFT_DATE", "BAND_AGE", "NUM_SURROGATE", "PLACEHOLDER"]
    text: str


@dataclass(frozen=True)
class LocEntry:
    """Maps an output byte/char range [out_start, out_end) back to an input Field."""
    out_start: int
    out_end: int
    key: RecKey
    field_idx: int
    in_start: int
    in_end: int
    exempt_kind: str | None = None


class LocMap:
    """Coordinate index built by projector.apply() for gate.py & residual layers."""
    def __init__(self, entries: list[LocEntry] | None = None) -> None:
        self.entries: list[LocEntry] = sorted(entries or [], key=lambda e: (e.out_start, e.out_end))
        # O(log n) lookup: starts for bisect + running max of ends (entries may overlap)
        self._starts = [e.out_start for e in self.entries]
        self._max_end: list[int] = []
        m = -1
        for e in self.entries:
            m = max(m, e.out_end)
            self._max_end.append(m)

    def lookup(self, out_start: int, out_end: int) -> list[LocEntry]:
        import bisect
        hi = bisect.bisect_left(self._starts, out_end)
        out: list[LocEntry] = []
        i = hi - 1
        while i >= 0 and self._max_end[i] > out_start:
            e = self.entries[i]
            if e.out_end > out_start and e.out_start < out_end:
                out.append(e)
            i -= 1
        out.reverse()
        return out

    def clinical_ranges(self) -> list[tuple[int, int, str]]:
        return [
            (e.out_start, e.out_end, e.exempt_kind)
            for e in self.entries
            if e.exempt_kind is not None
        ]


@dataclass(frozen=True)
class Hit:
    """Residual finding from Layer A or Layer B — NEVER carries matched text."""
    out_path: str
    channel: Literal["content", "path", "key", "header"]
    start: int
    end: int
    layer: Literal["A", "B"]
    check: Literal["A", "B1", "B2", "B3", "B4"]


@dataclass
class Context:
    """Execution context passed to detection and clinical shield stages (Owner B)."""
    policy: dict[str, Any] = field(default_factory=dict)
    lexicons: dict[str, Any] = field(default_factory=dict)
    coltypes: dict[str, str] = field(default_factory=dict)
    doctor_registry: set[str] = field(default_factory=set)
    keys: Any = None

