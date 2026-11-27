"""Dev-only truth oracle loader and Record/Span adapters.

Enforces strict path containment so truth annotations are only ever loaded
when ``input_dir`` resolves inside ``<KIT_DIR>/dev`` and ``truth.json``
resolves to ``<KIT_DIR>/dev/truth/truth.json``.

No raw PII surfaces, patient labels, or sensitive paths are included in
exception messages, coverage summaries, or consumer views.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from kavachmed.contracts import Field, RecKey, Record, Span
except ImportError:  # pragma: no cover - fallback before Plan A scaffolds contracts.py
    RecKey = tuple[str, str]

    @dataclass(frozen=True)
    class Field:
        idx: int
        path: str
        text: str
        vtype: str = "str"

    @dataclass
    class Record:
        key: RecKey
        fields: list[Field]
        meta: dict[str, Any] = field(default_factory=dict)

    @dataclass(frozen=True)
    class Span:
        key: RecKey
        field_idx: int
        start: int
        end: int
        typ: str
        role: str
        score: float
        src: str
        protected: bool = False
        exempt_kind: str | None = None


_VALID_ROLES = {"patient", "relative", "third_party", "sender", "doctor", "unknown", "none"}
_CHANNEL_TO_FIELD_PREFIX = {
    "filename": "@filename",
    "header": "@header",
    "key": "@keys",
    "metadata": "@meta",
}


class OracleBoundaryError(PermissionError):
    """Raised when oracle input or truth paths violate the dev-only boundary."""


class OracleConfigError(ValueError):
    """Raised when truth.json is missing, malformed, or has duplicate keys."""


@dataclass
class OracleCoverage:
    """Aggregate, text-free coverage counts across oracle lookups."""

    units_queried: int = 0
    units_missing: int = 0
    spans_emitted: int = 0
    spans_ambiguous: int = 0
    spans_unresolved: int = 0
    spans_unsupported_channel: int = 0
    dates_emitted: int = 0
    fields_emitted: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "units_queried": self.units_queried,
            "units_missing": self.units_missing,
            "spans_emitted": self.spans_emitted,
            "spans_ambiguous": self.spans_ambiguous,
            "spans_unresolved": self.spans_unresolved,
            "spans_unsupported_channel": self.spans_unsupported_channel,
            "dates_emitted": self.dates_emitted,
            "fields_emitted": self.fields_emitted,
        }


def is_oracle_enabled(env: dict[str, str] | None = None) -> bool:
    """Return True only when KM_ORACLE=1 is explicitly set."""
    source = os.environ if env is None else env
    return source.get("KM_ORACLE", "").strip() == "1"


def _resolve_dev_paths(
    input_dir: str | Path,
    kit_dir: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    """Validate and resolve (dev_root, resolved_input, resolved_truth) before reading."""
    raw_kit = kit_dir if kit_dir is not None else os.environ.get("KIT_DIR", "")
    if not raw_kit or not str(raw_kit).strip():
        raise OracleBoundaryError("oracle_kit_dir_unset")

    kit_root = Path(raw_kit).resolve(strict=True)
    dev_root = (kit_root / "dev").resolve(strict=True)
    if not dev_root.is_dir():
        raise OracleBoundaryError("oracle_dev_root_missing")

    resolved_input = Path(input_dir).resolve(strict=True)
    if not resolved_input.is_relative_to(dev_root):
        raise OracleBoundaryError("oracle_input_outside_dev_root")

    truth_path = (dev_root / "truth" / "truth.json").resolve(strict=False)
    if not truth_path.is_relative_to(dev_root):
        raise OracleBoundaryError("oracle_truth_outside_dev_root")
    if not truth_path.is_file():
        raise OracleConfigError("oracle_truth_missing")

    return dev_root, resolved_input, truth_path


def _normalize_role(role: Any) -> str:
    if isinstance(role, str) and role in _VALID_ROLES:
        return role
    return "unknown"


def _eligible_fields(record: Record, channel: str) -> list[Field]:
    """Return fields eligible for a given annotation channel."""
    if channel == "content":
        return [
            f
            for f in record.fields
            if f.vtype in {"str", "int", "float", "b64"}
            and not f.path.startswith("@")
        ]
    prefix = _CHANNEL_TO_FIELD_PREFIX.get(channel)
    if prefix is not None:
        return [
            f
            for f in record.fields
            if f.path == prefix or f.path.startswith(prefix + "/") or f.vtype in {"path", "header", "key"}
        ]
    return []


def _find_all_occurrences(haystack: str, needle: str) -> list[tuple[int, int]]:
    if not needle:
        return []
    hits: list[tuple[int, int]] = []
    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx == -1:
            break
        hits.append((idx, idx + len(needle)))
        start = idx + 1
    return hits


class Oracle:
    """Consumer view of dev truth annotations (no patient IDs exposed)."""

    def __init__(self, units: dict[RecKey, dict[str, Any]]) -> None:
        self._units = units
        self.coverage = OracleCoverage()

    def has_record(self, key: RecKey) -> bool:
        return key in self._units

    def oracle_spans(self, record: Record) -> list[Span]:
        """Adapt truth entities for ``record`` into ``Span`` objects."""
        self.coverage.units_queried += 1
        unit = self._units.get(record.key)
        if unit is None:
            self.coverage.units_missing += 1
            return []

        spans: list[Span] = []
        entities = unit.get("entities") or []
        for ent in entities:
            if not isinstance(ent, dict):
                continue
            channel = str(ent.get("channel") or "content")
            surface = ent.get("surface")
            if not isinstance(surface, str) or not surface:
                self.coverage.spans_unresolved += 1
                continue

            eligible = _eligible_fields(record, channel)
            if not eligible:
                if channel not in {"content", *_CHANNEL_TO_FIELD_PREFIX}:
                    self.coverage.spans_unsupported_channel += 1
                else:
                    self.coverage.spans_unresolved += 1
                continue

            # Prefer explicit coordinates when provided and valid.
            explicit_fidx = ent.get("field_idx")
            explicit_start = ent.get("start")
            explicit_end = ent.get("end")
            if (
                isinstance(explicit_fidx, int)
                and isinstance(explicit_start, int)
                and isinstance(explicit_end, int)
            ):
                matched_field = next((f for f in eligible if f.idx == explicit_fidx), None)
                if (
                    matched_field is not None
                    and 0 <= explicit_start < explicit_end <= len(matched_field.text)
                    and matched_field.text[explicit_start:explicit_end] == surface
                ):
                    spans.append(
                        Span(
                            key=record.key,
                            field_idx=matched_field.idx,
                            start=explicit_start,
                            end=explicit_end,
                            typ=str(ent.get("type") or "UNKNOWN"),
                            role=_normalize_role(ent.get("role")),
                            score=1.0,
                            src="ids",
                        )
                    )
                    self.coverage.spans_emitted += 1
                    continue

            # Otherwise match surface within eligible fields; abstain if ambiguous.
            candidates: list[tuple[int, int, int]] = []
            for f in eligible:
                for s, e in _find_all_occurrences(f.text, surface):
                    candidates.append((f.idx, s, e))

            if len(candidates) == 1:
                f_idx, s, e = candidates[0]
                spans.append(
                    Span(
                        key=record.key,
                        field_idx=f_idx,
                        start=s,
                        end=e,
                        typ=str(ent.get("type") or "UNKNOWN"),
                        role=_normalize_role(ent.get("role")),
                        score=1.0,
                        src="ids",
                    )
                )
                self.coverage.spans_emitted += 1
            elif len(candidates) > 1:
                self.coverage.spans_ambiguous += 1
            else:
                self.coverage.spans_unresolved += 1

        return spans

    def oracle_dates(self, record: Record) -> list[str]:
        """Return annotated date strings for ``record`` (no patient labels)."""
        unit = self._units.get(record.key)
        if unit is None:
            return []
        dates = [str(d) for d in (unit.get("dates") or []) if d is not None]
        self.coverage.dates_emitted += len(dates)
        return dates

    def oracle_fields(self, record: Record) -> dict[str, Any]:
        """Return a copy of annotated quasi-identifier fields for ``record``."""
        unit = self._units.get(record.key)
        if unit is None:
            return {}
        fields_dict = unit.get("fields")
        if not isinstance(fields_dict, dict):
            return {}
        out = dict(fields_dict)
        self.coverage.fields_emitted += len(out)
        return out


def load_oracle(
    input_dir: str | Path,
    kit_dir: str | Path | None = None,
) -> Oracle:
    """Validate dev containment and load ``truth.json`` into an ``Oracle``."""
    try:
        _dev_root, _resolved_input, truth_path = _resolve_dev_paths(input_dir, kit_dir)
    except OSError as exc:
        raise OracleBoundaryError("oracle_path_resolution_failed") from exc

    try:
        raw_text = truth_path.read_text(encoding="utf-8")
        payload = json.loads(raw_text)
    except (OSError, ValueError) as exc:
        raise OracleConfigError("oracle_truth_malformed") from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("files"), dict):
        raise OracleConfigError("oracle_truth_schema_invalid")

    units: dict[RecKey, dict[str, Any]] = {}
    seen_counts: Counter[RecKey] = Counter()

    for rel_path, file_entry in payload["files"].items():
        if not isinstance(rel_path, str) or not isinstance(file_entry, dict):
            raise OracleConfigError("oracle_truth_file_entry_invalid")
        file_units = file_entry.get("units")
        if not isinstance(file_units, dict):
            raise OracleConfigError("oracle_truth_units_invalid")
        for loc, unit_entry in file_units.items():
            if not isinstance(loc, str) or not isinstance(unit_entry, dict):
                raise OracleConfigError("oracle_truth_unit_invalid")
            rec_key: RecKey = (rel_path, loc)
            seen_counts[rec_key] += 1
            if seen_counts[rec_key] > 1:
                raise OracleConfigError("oracle_duplicate_record_key")
            # Strip patient / link from consumer unit view.
            units[rec_key] = {
                "entities": list(unit_entry.get("entities") or []),
                "dates": list(unit_entry.get("dates") or []),
                "fields": dict(unit_entry.get("fields") or {})
                if isinstance(unit_entry.get("fields"), dict)
                else {},
            }

    return Oracle(units)
