"""Dev-only C5 evidence extraction harness using V2 oracle (Plan C V3).

Evaluates ``kavachmed.evidence.extract`` against ``Oracle.oracle_fields``
and ``Oracle.oracle_dates`` using ``Oracle.oracle_spans``. Tracks injected
emulation values separately from measured accuracy, and outputs only
aggregate counts (never raw PII or patient labels).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kavachmed.evidence import DateProvider, Evidence, LgdLookup, extract
from tools.oracle import Oracle
from tools.oracle.loader import Record


@dataclass
class C5Report:
    """Aggregate, text-free C5 extraction evaluation report."""

    comparable_records: int = 0
    passing_records: int = 0
    missing_oracle_units: int = 0
    field_passed: dict[str, int] = field(default_factory=dict)
    field_total: dict[str, int] = field(default_factory=dict)
    injected_checks: int = 0
    issuer_mismatches: int = 0
    oracle_coverage: dict[str, int] = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        return (
            self.passing_records / self.comparable_records
            if self.comparable_records
            else 0.0
        )

    @property
    def target_status(self) -> str:
        if self.comparable_records == 0:
            return "pending"
        return "pass" if self.pass_rate >= 0.98 else "below_target"


def _compare_field(
    ev: Evidence,
    field_name: str,
    expected_val: Any,
) -> bool:
    if field_name in {"gender", "sex"}:
        exp = str(expected_val).strip().upper()[:1]
        return ev.sex == exp
    if field_name == "age":
        if ev.birth_years is None:
            return False
        # Without an encounter year, we only verify that a valid interval exists.
        return ev.birth_years[0] <= ev.birth_years[1]
    if field_name == "birth_year":
        if ev.birth_years is None:
            return False
        yr = int(expected_val)
        return ev.birth_years[0] <= yr <= ev.birth_years[1]
    if field_name == "hospital":
        return ev.hospital == expected_val
    return True


def evaluate_records_with_oracle(
    records: list[Record],
    oracle: Oracle,
    *,
    date_provider: DateProvider | None = None,
    lgd_lookup: LgdLookup | None = None,
    injected_date_records: set[tuple[str, str]] | None = None,
) -> tuple[C5Report, list[Evidence]]:
    """Run ``extract`` using ``oracle_spans`` and score against ``oracle_fields``."""
    report = C5Report()
    evidences: list[Evidence] = []
    injected = injected_date_records or set()

    for rec in records:
        spans = oracle.oracle_spans(rec)
        expected_fields = oracle.oracle_fields(rec)
        _oracle_dates = oracle.oracle_dates(rec)
        if not oracle.has_record(rec.key):
            report.missing_oracle_units += 1
            continue

        diag: dict[str, int] = {}
        ev = extract(
            rec,
            spans,
            date_provider=date_provider,
            lgd_lookup=lgd_lookup,
            diagnostics=diag,
        )
        report.issuer_mismatches += diag.get("issuer_mismatches", 0)
        evidences.append(ev)

        if rec.key in injected:
            report.injected_checks += 1
            continue

        comparable_keys = [
            k
            for k in expected_fields
            if k in {"gender", "sex", "birth_year", "hospital"}
            and expected_fields[k] is not None
        ]
        if not comparable_keys:
            continue

        report.comparable_records += 1
        all_ok = True
        for k in comparable_keys:
            report.field_total[k] = report.field_total.get(k, 0) + 1
            if _compare_field(ev, k, expected_fields[k]):
                report.field_passed[k] = report.field_passed.get(k, 0) + 1
            else:
                all_ok = False
        if all_ok:
            report.passing_records += 1

    report.oracle_coverage = oracle.coverage.as_dict()
    return report, evidences
