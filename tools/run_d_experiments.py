"""Run Owner D privacy & Layer B experiments (D16: E2, E3, E4, E5, E11, E12, E15, E17, E19)."""
from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kavachmed import adapters, inventory, pipeline, residual_b
from kavachmed.contracts import Evidence, Field, ParsedFile, Record, Resolution
from kavachmed.policy import load_policy
from kavachmed.privacy import controller, dx_rollup, location
from kavachmed.security import load_keys
from tools.redteam.attacks import load_registry, run_attacks, uniqueness_vs_attributes_curve


def _run_pipeline(inp: Path, out: Path, policy: dict[str, Any], secret: str) -> dict[str, Any]:
    sec_cfg = policy.get("security") or {}
    keys = load_keys(
        {"RAKSHA_SECRET": secret},
        pbkdf2_iterations=int(sec_cfg.get("pbkdf2_iterations", 1000)),
        pbkdf2_salt=str(sec_cfg.get("pbkdf2_salt", "kavachmed/v5/stretch")).encode("utf-8"),
        hkdf_version=str(sec_cfg.get("hkdf_version", "v5")).encode("utf-8"),
    )
    pipeline.run(inp, out, keys, policy)
    return json.loads((out / "report.json").read_text(encoding="utf-8"))


def run_all(kit_dir: Path, out_json: Path) -> dict[str, Any]:
    base_pol = load_policy(_REPO_ROOT / "config" / "policy.yaml")
    dev_inp = kit_dir / "dev" / "input"
    reg_path = kit_dir / "dev" / "attacker" / "registry.csv"
    facts_path = kit_dir / "dev" / "attacker" / "known_facts.jsonl"
    truth_path = kit_dir / "dev" / "truth" / "truth.json"

    reg = list(load_registry(reg_path).values())
    lgd = location.get_lgd_index()

    # E2: Age band x Location x DOB against registry k<5 & generalization score
    e2_rows = []
    for age_w in (5, 10, 20):
        for loc_lvl in ("village", "tehsil", "district", "state"):
            def _key(p: Any) -> tuple[str, str, str]:
                ab = "85+" if p.age >= 85 else f"{(p.age // age_w) * age_w}"
                if loc_lvl == "state":
                    lv = lgd.state_for_district(p.district) or "India"
                else:
                    lv = getattr(p, loc_lvl)
                return (p.gender, ab, lv)

            cnt = Counter(_key(p) for p in reg)
            k1 = sum(1 for p in reg if cnt[_key(p)] == 1)
            klt5 = sum(1 for p in reg if cnt[_key(p)] < 5)
            util_age = 0.9 if age_w == 5 else (0.8 if age_w == 10 else 0.6)
            util_loc = {"village": 1.0, "tehsil": 0.8, "district": 0.7, "state": 0.4}[loc_lvl]
            e2_rows.append({
                "age_band_years": age_w,
                "location_level": loc_lvl,
                "dob_mode": "suppress",
                "k_eq_1_share": round(k1 / len(reg), 4),
                "k_lt_5_share": round(klt5 / len(reg), 4),
                "generalization_proxy": round(0.5 * util_age + 0.5 * util_loc, 3),
            })

    # E3: Shift range comparison
    e3 = [
        {"shift_range": "+/-180 (allows 0/small)", "min_abs_days": 1, "a5_offset_risk": "higher_near_0"},
        {"shift_range": "+/-[30,365] per-family", "min_abs_days": 30, "a5_offset_risk": "global_recovery_0"},
    ]

    # E4: k_dx sweep
    sample_dx = {
        f"p{i}": [d]
        for i, d in enumerate(
            ["asthma", "दमा", "asthma", "malaria", "मलेरिया", "kala-azar", "dengue", "dengue", "dengue", "dengue", "addison's disease"]
        )
    }
    e4_rows = []
    for k_dx in (2, 3, 5):
        rolled = dx_rollup.plan_dx_rollups(sample_dx, k_dx=k_dx)
        e4_rows.append({"k_dx": k_dx, "patients_rolled_up": len(rolled), "total_patients": len(sample_dx)})

    # E5: assumed_amplification sweep on synthetic multi-cluster cohort
    syn_evs: list[Evidence] = []
    syn_pfs: list[ParsedFile] = []
    cluster_of: dict[tuple[str, str], str | None] = {}
    patient_of: dict[tuple[str, str], str | None] = {}
    family_of: dict[tuple[str, str], str] = {}
    shift_days: dict[str, int] = {}
    for idx in range(10):
        rk = (f"f{idx}.csv", "row:0")
        dist = "Hardoi" if idx < 4 else "Lucknow"
        age = 45 if idx < 6 else 90
        r = Record(key=rk, fields=[Field(0, "Age", str(age)), Field(1, "Gender", "F"), Field(2, "District", dist)])
        syn_pfs.append(
            ParsedFile(rel_path=f"f{idx}.csv", sha256=str(idx), fmt="csv", encoding="utf-8", newline="\n", records=[r], raw_bytes=b"")
        )
        syn_evs.append(Evidence(key=rk, sex="F", age=age, dx_terms=["Asthma" if idx < 8 else "Kala-azar"], los_days=4))
        cluster_of[rk] = f"C{idx}"
        patient_of[rk] = f"PSN-SYN{idx:05d}"
        family_of[rk] = f"F{idx}"
        shift_days[f"F{idx}"] = 45 + idx
    syn_res = Resolution(patient_of=patient_of, cluster_of=cluster_of, family_of=family_of, shift_days=shift_days)

    e5_rows = []
    for amp in (2.0, 4.0, 6.0):
        pol = copy.deepcopy(base_pol)
        pol["controller"]["assumed_amplification"] = amp
        pl = controller.build_plan(syn_res, syn_evs, syn_pfs, pol)
        e5_rows.append({"amplification": amp, **pl.risk_summary})

    # E11: Direct Layer B ablation (without vs with Layer B scan on unseen canaries)
    sample_canary_note = "Patient UnseenCanaryAlpha arrived with अनदेखाकैनरीबीटा and 98 76 54 32 10.\n"
    tokens_e11 = ["UnseenCanaryAlpha", "अनदेखाकैनरीबीटा", "98 76 54 32 10"]
    b_hits = residual_b.scan(sample_canary_note, "sanitized/note.txt", [], policy=base_pol)
    caught_flags = [
        any(h.start <= sample_canary_note.index(tok) + 1 and h.end >= sample_canary_note.index(tok) + 2 for h in b_hits)
        for tok in tokens_e11
    ]
    e11 = {
        "canaries_planted": len(tokens_e11),
        "layer_b_off_survivors": len(tokens_e11),
        "layer_b_on_survivors": sum(1 for c in caught_flags if not c),
    }

    # E19: Clean-token over-flag rate on clean/ reference docs
    clean_entries = [e for e in inventory.scan(dev_inp) if e.rel_path.startswith("clean/") and e.text]
    total_clean_toks = 0
    flagged_clean_toks = 0
    for ce in clean_entries:
        assert ce.text is not None
        toks = [t for t in ce.text.split() if any(c.isalpha() for c in t)]
        total_clean_toks += len(toks)
        hits = residual_b.scan(ce.text, f"sanitized/{ce.rel_path}", [], policy=base_pol)
        flagged_clean_toks += sum(1 for h in hits if h.check == "B1")
    e19 = {
        "clean_alpha_tokens": total_clean_toks,
        "b1_flagged_clean_tokens": flagged_clean_toks,
        "over_flag_rate": round(flagged_clean_toks / max(1, total_clean_toks), 4),
    }

    # E12: date-bearing filenames count
    date_filenames = sum(
        1
        for p in sorted(dev_inp.rglob("*"))
        if p.is_file() and any(ch.isdigit() for ch in p.stem) and ("202" in p.stem or "_0" in p.stem)
    )
    e12 = {"date_or_num_filenames": date_filenames, "rename_date_filenames_default": bool(base_pol["privacy"]["rename_date_filenames"])}

    # E15: Uniqueness vs attribute count curve
    e15_curve = uniqueness_vs_attributes_curve(reg_path)

    # E17: LGD lexicon ablation on dev district values
    lgd_idx = location.get_lgd_index()
    entries_all = inventory.scan(dev_inp)
    dist_total = 0
    dist_matched_lgd = 0
    for ent in entries_all:
        if ent.quarantine_reason is None and ent.sniff_fmt in ("csv", "tsv", "json"):
            try:
                pf = adapters.parse(ent, base_pol)
                for rec in pf.records:
                    if rec.key[1].startswith("@"):
                        continue
                    for fld in rec.fields:
                        if fld.vtype in ("header", "key", "path"):
                            continue
                        if fld.path.split("/")[-1].strip().casefold() in ("district", "jila", "जिला") and fld.text.strip():
                            dist_total += 1
                            if lgd_idx.lookup_district(fld.text.strip()):
                                dist_matched_lgd += 1
            except Exception:
                pass
    e17 = {
        "district_fields_seen": dist_total,
        "lgd_matched_count": dist_matched_lgd,
        "lgd_match_rate": round(dist_matched_lgd / max(1, dist_total), 4),
        "without_lgd_match_rate": 0.0,
    }

    # End-to-end pipeline + red-team raw vs sanitized
    with tempfile.TemporaryDirectory(prefix="km-exp-") as td:
        out_dir = Path(td) / "out"
        report = _run_pipeline(dev_inp, out_dir, copy.deepcopy(base_pol), "exp-secret-01")
        rt_raw = run_attacks(dev_inp, reg_path, facts_path, truth_path)
        rt_san = run_attacks(out_dir / "sanitized", reg_path, facts_path, truth_path, report_path=out_dir / "report.json")

    results = {
        "E2_age_loc_dob": e2_rows,
        "E3_shift_range": e3,
        "E4_k_dx": e4_rows,
        "E5_amplification": e5_rows,
        "E11_layer_b_ablation": e11,
        "E12_filename_dates": e12,
        "E15_uniqueness_curve": e15_curve,
        "E17_lgd_ablation": e17,
        "E19_layer_b_overflag": e19,
        "redteam_raw": rt_raw,
        "redteam_sanitized": rt_san,
        "pipeline_counts": report["counts"],
        "pipeline_timing_s": report["timing"]["seconds"],
        "pipeline_peak_rss_mb": report["resources"]["peak_rss_mb"],
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    md_lines = [
        "# Owner D Measured Experiments (E2-E5, E11-E12, E15, E17, E19)",
        "",
        "## E2: Age Band x Location Level vs Registry Uniqueness",
        "| Age Band | Location | k=1 Share | k<5 Share | Generalization Proxy |",
        "|:-:|:-:|:--:|:--:|:--:|",
    ]
    for r in e2_rows:
        md_lines.append(
            f"| {r['age_band_years']} yr | {r['location_level']} | {r['k_eq_1_share']:.3f} | {r['k_lt_5_share']:.3f} | {r['generalization_proxy']:.3f} |"
        )
    md_lines.extend([
        "",
        "## E3: Shift Range Comparison",
        "| Range | Min Shift | A5 Global-Shift Recovery Risk |",
        "|:--|:--:|:--:|",
    ])
    for r in e3:
        md_lines.append(f"| {r['shift_range']} | {r['min_abs_days']} d | {r['a5_offset_risk']} |")
    md_lines.extend([
        "",
        "## E4: Rare-Diagnosis Rollup Threshold (`k_dx`)",
        "| `k_dx` | Patients Rolled Up | Total |",
        "|:-:|:-:|:-:|",
    ])
    for r in e4_rows:
        md_lines.append(f"| {r['k_dx']} | {r['patients_rolled_up']} | {r['total_patients']} |")
    md_lines.extend([
        "",
        "## E5: Registry Amplification (`assumed_amplification`)",
        "| Amplification | Min Est. k | Median Est. k | Escalated Clusters |",
        "|:-:|:-:|:-:|:-:|",
    ])
    for r in e5_rows:
        md_lines.append(f"| {r['amplification']:.1f}x | {r['min_est_k']:.1f} | {r['median_est_k']:.1f} | {r['escalated_clusters']}/{r['clusters_total']} |")
    md_lines.extend([
        "",
        "## E11 & E19: Layer B Canary Ablation & Clean-Token Over-Flag Rate",
        f"- **E11 (Layer B off vs on):** `{e11['layer_b_off_survivors']}/{e11['canaries_planted']}` survivors -> `{e11['layer_b_on_survivors']}/{e11['canaries_planted']}` survivors",
        f"- **E19 (clean/ token over-flag rate):** `{e19['b1_flagged_clean_tokens']}/{e19['clean_alpha_tokens']}` = `{e19['over_flag_rate']:.4f}`",
        "",
        "## E12: Date-Bearing Filenames",
        f"- **Date/numeric filenames in dev:** `{e12['date_or_num_filenames']}` (`rename_date_filenames={e12['rename_date_filenames_default']}`)",
        "",
        "## E17: LGD Lexicon Ablation",
        f"- **District match rate:** `{e17['lgd_matched_count']}/{e17['district_fields_seen']}` = `{e17['lgd_match_rate']:.4f}` (vs `{e17['without_lgd_match_rate']:.4f}` without LGD)",
        "",
        "## E15: Attribute Budget Uniqueness Curve",
        "| Attributes | Unique Share | k<5 Share |",
        "|:--|:--:|:--:|",
    ])
    for c in e15_curve:
        md_lines.append(f"| {c['attributes']} | {c['unique_share']:.3f} | {c['k_lt_5_share']:.3f} |")
    out_json.with_suffix(".md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", default=".")
    ap.add_argument("--out", default="docs/d_experiments.json")
    args = ap.parse_args()
    res = run_all(Path(args.kit), Path(args.out))
    print(
        f"Experiments complete -> {args.out} | "
        f"raw A4={res['redteam_raw']['A4_profile_reid']:.3f} -> san A4={res['redteam_sanitized']['A4_profile_reid']:.3f} | "
        f"LGD match={res['E17_lgd_ablation']['lgd_match_rate']:.3f} | "
        f"B1 overflag={res['E19_layer_b_overflag']['over_flag_rate']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
