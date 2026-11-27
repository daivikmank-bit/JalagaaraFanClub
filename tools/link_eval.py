#!/usr/bin/env python3
"""Linkage evaluator with kit-parity metrics, B3, and trap diagnostics (Plan C V2).

Usage:
    python3 tools/link_eval.py --truth dev/truth/truth.json --output my_output/
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def record_for(loc: str, recs: dict[str, str | None]) -> str | None:
    """Mirror check_dev.py locator resolution: exact, JSON-pointer ancestor, or doc.

    An explicit ``None`` at the exact locator takes precedence over fallbacks.
    """
    if loc in recs:
        return recs[loc]
    if loc.startswith("/"):
        parts = loc.split("/")
        for i in range(len(parts) - 1, 1, -1):
            anc = "/".join(parts[:i])
            if anc in recs:
                return recs[anc]
    return recs.get("doc")


@dataclass(frozen=True)
class ParityMetrics:
    evaluated_records: int
    linked_records: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f05: float
    merged_clusters: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "evaluated_records": self.evaluated_records,
            "linked_records": self.linked_records,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f05": self.f05,
            "merged_clusters": self.merged_clusters,
        }


@dataclass(frozen=True)
class B3Metrics:
    evaluated_records: int
    precision: float
    recall: float
    f1: float
    all_null: bool
    documented_identity_score_zero: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "evaluated_records": self.evaluated_records,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "all_null": self.all_null,
            "documented_identity_score_zero": self.documented_identity_score_zero,
        }


@dataclass(frozen=True)
class CohortMetric:
    status: str  # "ok" | "unavailable"
    numerator: int
    denominator: int
    value: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "value": self.value,
        }


def compute_parity_metrics(units: list[tuple[str, str | None]]) -> ParityMetrics:
    """Compute pairwise TP/FP/FN, precision, recall, F0.5, and merged clusters."""
    by_true: dict[str, list[int]] = defaultdict(list)
    by_pred: dict[str, list[int]] = defaultdict(list)

    for idx, (true_pt, pred_psn) in enumerate(units):
        by_true[true_pt].append(idx)
        if pred_psn:
            by_pred[pred_psn].append(idx)

    true_pairs = sum(len(v) * (len(v) - 1) // 2 for v in by_true.values())
    pred_pairs = sum(len(v) * (len(v) - 1) // 2 for v in by_pred.values())

    tp = 0
    for members in by_pred.values():
        counts_by_true: dict[str, int] = defaultdict(int)
        for idx in members:
            counts_by_true[units[idx][0]] += 1
        for cnt in counts_by_true.values():
            tp += cnt * (cnt - 1) // 2

    fp = pred_pairs - tp
    fn = true_pairs - tp
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f05 = (1.25 * prec * rec / (0.25 * prec + rec)) if (prec + rec) else 0.0
    merged_clusters = sum(
        1 for members in by_pred.values() if len({units[i][0] for i in members}) > 1
    )

    return ParityMetrics(
        evaluated_records=len(units),
        linked_records=sum(1 for _, p in units if p),
        tp=tp,
        fp=fp,
        fn=fn,
        precision=prec,
        recall=rec,
        f05=f05,
        merged_clusters=merged_clusters,
    )


def compute_b3_metrics(units: list[tuple[str, str | None]]) -> B3Metrics:
    """Compute B-cubed precision, recall, and F1 treating unlinked records as singletons."""
    n = len(units)
    if n == 0:
        return B3Metrics(
            evaluated_records=0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            all_null=True,
            documented_identity_score_zero=True,
        )

    by_true: dict[str, set[int]] = defaultdict(set)
    by_pred: dict[str, set[int]] = defaultdict(set)
    any_linked = False

    for idx, (true_pt, pred_psn) in enumerate(units):
        by_true[true_pt].add(idx)
        if pred_psn:
            any_linked = True
            by_pred[pred_psn].add(idx)
        else:
            by_pred[f"__singleton__:{idx}"].add(idx)

    total_p = 0.0
    total_r = 0.0
    for idx, (true_pt, pred_psn) in enumerate(units):
        t_set = by_true[true_pt]
        p_key = pred_psn if pred_psn else f"__singleton__:{idx}"
        p_set = by_pred[p_key]
        overlap = len(t_set & p_set)
        total_p += overlap / len(p_set)
        total_r += overlap / len(t_set)

    prec = total_p / n
    rec = total_r / n
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

    return B3Metrics(
        evaluated_records=n,
        precision=prec,
        recall=rec,
        f1=f1,
        all_null=not any_linked,
        documented_identity_score_zero=not any_linked,
    )


def _pair_recall_for_patients(
    units: list[tuple[str, str | None]],
    cohort_patients: set[str],
) -> CohortMetric:
    if not cohort_patients:
        return CohortMetric(status="unavailable", numerator=0, denominator=0, value=None)
    filtered = [(t, p) for (t, p) in units if t in cohort_patients]
    if not filtered:
        return CohortMetric(status="unavailable", numerator=0, denominator=0, value=None)
    m = compute_parity_metrics(filtered)
    denom = m.tp + m.fn
    if denom == 0:
        return CohortMetric(status="unavailable", numerator=0, denominator=0, value=None)
    return CohortMetric(status="ok", numerator=m.tp, denominator=denom, value=m.tp / denom)


def _homonym_separation(
    units: list[tuple[str, str | None]],
    homonym_patients: set[str],
) -> CohortMetric:
    if len(homonym_patients) < 2:
        return CohortMetric(status="unavailable", numerator=0, denominator=0, value=None)
    filtered = [(t, p) for (t, p) in units if t in homonym_patients]
    distinct_patients = sorted({t for t, _ in filtered})
    if len(distinct_patients) < 2:
        return CohortMetric(status="unavailable", numerator=0, denominator=0, value=None)

    pred_by_patient: dict[str, set[str]] = defaultdict(set)
    for t, p in filtered:
        if p:
            pred_by_patient[t].add(p)

    separated = 0
    total_pairs = 0
    for p1, p2 in itertools.combinations(distinct_patients, 2):
        total_pairs += 1
        if not (pred_by_patient[p1] & pred_by_patient[p2]):
            separated += 1

    if total_pairs == 0:
        return CohortMetric(status="unavailable", numerator=0, denominator=0, value=None)
    return CohortMetric(
        status="ok",
        numerator=separated,
        denominator=total_pairs,
        value=separated / total_pairs,
    )


def _extract_cohorts(truth: dict[str, Any]) -> dict[str, set[str]]:
    traps = truth.get("traps") or {}
    if not isinstance(traps, dict):
        return {"language": set(), "fragmented": set(), "homonym": set()}

    def _as_set(keys: tuple[str, ...]) -> set[str]:
        out: set[str] = set()
        for k in keys:
            val = traps.get(k)
            if isinstance(val, list):
                out.update(str(x) for x in val)
            elif isinstance(val, dict):
                out.update(str(x) for x in val.keys())
        return out

    return {
        "language": _as_set(("language", "language_trap", "multilingual")),
        "fragmented": _as_set(("fragmented", "fragmented_notes", "no_id")),
        "homonym": _as_set(("homonym", "homonyms", "same_name")),
    }


def evaluate_output(truth_path: Path, output_dir: Path) -> dict[str, Any]:
    """Load truth.json + output/{report.json,records.jsonl} and compute all metrics."""
    truth = json.loads(truth_path.read_text(encoding="utf-8"))

    try:
        report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        report = {"files": []}
    renamed = {
        e.get("input_sha256"): e.get("output")
        for e in (report.get("files") or [])
        if isinstance(e, dict)
    }

    recs: dict[str, dict[str, str | None]] = defaultdict(dict)
    records_path = output_dir / "records.jsonl"
    if records_path.is_file():
        for line in records_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            out_file = str(r["file"]).replace("sanitized/", "", 1)
            recs[out_file][r["loc"]] = r.get("patient")

    units: list[tuple[str, str | None]] = []
    ambiguous_merges = 0
    ambiguous_checked = 0

    for rel, f in (truth.get("files") or {}).items():
        out_rel = rel
        declared = renamed.get(f.get("sha256"))
        if declared:
            out_rel = str(declared).replace("sanitized/", "", 1)
        file_recs = recs.get(out_rel, {})
        for loc, u in (f.get("units") or {}).items():
            link_type = u.get("link")
            patient = u.get("patient")
            pred = record_for(loc, file_recs)
            if link_type == "required" and patient:
                units.append((str(patient), pred))
            elif link_type == "ambiguous":
                ambiguous_checked += 1
                if pred:
                    ambiguous_merges += 1

    parity = compute_parity_metrics(units)
    b3 = compute_b3_metrics(units)
    cohorts = _extract_cohorts(truth)
    diag = {
        "language_pair_recall": _pair_recall_for_patients(units, cohorts["language"]).as_dict(),
        "fragmented_pair_recall": _pair_recall_for_patients(units, cohorts["fragmented"]).as_dict(),
        "homonym_separation": _homonym_separation(units, cohorts["homonym"]).as_dict(),
        "ambiguous_unit_merges": {
            "count": ambiguous_merges,
            "checked": ambiguous_checked,
        },
    }

    return {
        "parity": parity.as_dict(),
        "b3": b3.as_dict(),
        "diagnostics": diag,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate linkage output against dev truth.")
    ap.add_argument("--truth", required=True, help="Path to truth.json")
    ap.add_argument("--output", required=True, help="Path to output directory")
    ap.add_argument("--parity-only", action="store_true", help="Print kit parity metrics only")
    ap.add_argument("--json", action="store_true", help="Emit JSON summary")
    args = ap.parse_args(argv)

    result = evaluate_output(Path(args.truth), Path(args.output))
    if args.json:
        payload = {"parity": result["parity"]} if args.parity_only else result
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    p = result["parity"]
    print(
        f"PARITY: evaluated={p['evaluated_records']} linked={p['linked_records']} "
        f"TP={p['tp']} FP={p['fp']} FN={p['fn']} "
        f"precision={p['precision']:.4f} recall={p['recall']:.4f} F0.5={p['f05']:.4f} "
        f"merged_clusters={p['merged_clusters']}"
    )
    if not args.parity_only:
        b = result["b3"]
        d = result["diagnostics"]
        print(
            f"B3: precision={b['precision']:.4f} recall={b['recall']:.4f} F1={b['f1']:.4f} "
            f"all_null={b['all_null']}"
        )
        print(
            f"TRAPS: language={d['language_pair_recall']['status']} "
            f"fragmented={d['fragmented_pair_recall']['status']} "
            f"homonym={d['homonym_separation']['status']} "
            f"ambiguous_merges={d['ambiguous_unit_merges']['count']}/"
            f"{d['ambiguous_unit_merges']['checked']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
