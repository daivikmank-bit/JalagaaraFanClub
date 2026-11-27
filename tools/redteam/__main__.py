"""CLI entrypoint for red-team evaluation and Expert-Determination summary (D9, D15)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .attacks import eval_a6_pseudonym_inversion, eval_a7_cross_release, run_attacks, uniqueness_vs_attributes_curve


def _render_expert_summary(summary: dict) -> str:
    raw = summary["raw"]
    san = summary["sanitized"]
    lines = [
        "# Expert-Determination-Style Risk Summary (KavachMed Red Team)",
        "",
        "> **Scope:** Quantitative evaluation against the published A1–A7 registry & known-facts adversary (`docs/ATTACKS.md`). Not a legal certification.",
        "",
        "## 1. Methods & Attacker Assumptions",
        "- Population registry (`registry.csv`, 4–5x patient count) with demographic + leaked-contact overlay (60%).",
        "- Known-facts auxiliary knowledge (`known_facts.jsonl`) with visit dates, hospital, rare condition, and decoys.",
        "- Attacker attempts A1 (direct), A2/A3 (filtered Sweeney linkage), A4 (cross-file profile pooling), A5 (targeted + global-shift recovery), A6 (PSN inversion), and A7 (cross-release comparison).",
        "",
        "## 2. Measured Attack Exposure (Raw vs Sanitized)",
        "",
        "| Attack | Raw Input | Sanitized Release | Target |",
        "|:--|:--:|:--:|:--:|",
        f"| A1 Direct Exposure | {raw['A1_direct_exposure']*100:.1f}% | {san['A1_direct_exposure']*100:.1f}% | 0% |",
        f"| A3 Record Linkage (mean prob) | {raw['A3_record_linkage']*100:.1f}% | {san['A3_record_linkage']*100:.1f}% | low |",
        f"| A4 Profile Re-ID (>=0.5) | {raw['A4_profile_reid']*100:.1f}% | {san['A4_profile_reid']*100:.1f}% | <=5% |",
        f"| A5 Targeted Attack | {raw['A5_targeted_success']*100:.1f}% | {san['A5_targeted_success']*100:.1f}% | <=15% |",
        f"| A5 Decoy False Positives | {raw['A5_decoy_false_positives']} | {san['A5_decoy_false_positives']} | 0 |",
        f"| A6 Pseudonym Inversion | — | {summary['A6']['A6_inversion_rate']*100:.1f}% | 0% |",
        "",
        "## 3. Uniqueness vs Attribute Count Curve",
        "",
        "| Released Quasi-Identifiers | Registry Unique | k < 5 Share |",
        "|:--|:--:|:--:|",
    ]
    for c in summary["uniqueness_curve"]:
        lines.append(f"| {c['attributes']} | {c['unique_share']*100:.1f}% | {c['k_lt_5_share']*100:.1f}% |")
    lines.extend([
        "",
        f"**Determination Verdict:** `{summary['expert_determination']['verdict']}`",
        "",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="KavachMed Red-Team Evaluator (A1-A7)")
    ap.add_argument("--input", required=True, help="Raw input directory")
    ap.add_argument("--output", required=True, help="Pipeline output directory (with sanitized/)")
    ap.add_argument("--second-output", default=None, help="Optional second output directory under another secret for A7")
    ap.add_argument("--registry", required=True, help="Attacker registry.csv")
    ap.add_argument("--facts", required=True, help="Attacker known_facts.jsonl")
    ap.add_argument("--truth", required=True, help="Truth truth.json (eval only)")
    ap.add_argument("--json-out", default=None, help="Optional JSON report path")
    ap.add_argument("--summary-out", default=None, help="Optional Expert Determination Markdown path")
    args = ap.parse_args(argv)

    raw_metrics = run_attacks(Path(args.input), Path(args.registry), Path(args.facts), Path(args.truth))
    san_dir = Path(args.output) / "sanitized"
    san_metrics = run_attacks(
        san_dir,
        Path(args.registry),
        Path(args.facts),
        Path(args.truth),
        report_path=Path(args.output) / "report.json",
    )
    a6 = eval_a6_pseudonym_inversion(Path(args.output) / "records.jsonl", Path(args.registry))
    a7 = eval_a7_cross_release(Path(args.output), Path(args.second_output)) if args.second_output else None
    curve = uniqueness_vs_attributes_curve(Path(args.registry))

    a3_rel = san_metrics["A3_record_linkage"] / max(1e-6, raw_metrics["A3_record_linkage"])
    a4_rel = san_metrics["A4_profile_reid"] / max(1e-6, raw_metrics["A4_profile_reid"])
    a5_rel = san_metrics["A5_targeted_success"] / max(1e-6, raw_metrics["A5_targeted_success"])

    summary = {
        "raw": raw_metrics,
        "sanitized": san_metrics,
        "A6": a6,
        "A7": a7,
        "relative_to_raw": {
            "A3_ratio": round(a3_rel, 4),
            "A4_ratio": round(a4_rel, 4),
            "A5_ratio": round(a5_rel, 4),
        },
        "uniqueness_curve": curve,
        "expert_determination": {
            "method": "Registry linkage A1-A7 + k-anonymity amplification + Rocher attribute budget",
            "verdict": "very_small_risk" if (san_metrics["A4_profile_reid"] <= 0.05 and san_metrics["A5_targeted_success"] <= 0.15) else "escalate_recommended",
        },
    }

    print("=== Red Team (A1-A7) ===")
    print(f"Raw       : A1={raw_metrics['A1_direct_exposure']:.3f}  A3={raw_metrics['A3_record_linkage']:.3f}  A4={raw_metrics['A4_profile_reid']:.3f}  A5={raw_metrics['A5_targeted_success']:.3f}")
    print(f"Sanitized : A1={san_metrics['A1_direct_exposure']:.3f}  A3={san_metrics['A3_record_linkage']:.3f}  A4={san_metrics['A4_profile_reid']:.3f}  A5={san_metrics['A5_targeted_success']:.3f}")
    print(f"A6/A7     : A6_inv={a6['A6_inversion_rate']:.3f}" + (f"  A7_risk={a7['A7_cross_release_risk']:.3f}" if a7 else ""))
    print(f"Rel/Raw   : A3={a3_rel:.3f}  A4={a4_rel:.3f}  A5={a5_rel:.3f}")
    print("Uniqueness curve:", ", ".join(f"{c['attributes']}={c['unique_share']:.2f}" for c in curve))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.summary_out:
        Path(args.summary_out).write_text(_render_expert_summary(summary), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
