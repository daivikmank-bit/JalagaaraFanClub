"""Phase 3 experiments owned by B: E8, E14, E18, E20 (corrected controls & denominators)."""

from __future__ import annotations

import json
from pathlib import Path

from kavachmed.contracts import Context, Field, Record, Span
from kavachmed.detect.arbiter import arbitrate, detect
from kavachmed.detect.context import detect_context
from kavachmed.detect.default_deny import detect_default_deny
from kavachmed.detect.ids import detect_ids
from kavachmed.detect.labels import detect_labels, induce_labels
from kavachmed.shield import protect


def _load_lexicons() -> dict[str, set[str]]:
    root = Path(__file__).resolve().parents[1]
    safe = {
        l.strip().lower()
        for l in (root / "config" / "lexicons" / "safe" / "safe_words.txt").read_text(encoding="utf-8").splitlines()
        if l.strip()
    }
    names = {
        l.strip().lower()
        for l in (root / "config" / "lexicons" / "names" / "generic_names_seed.txt").read_text(encoding="utf-8").splitlines()
        if l.strip()
    }
    drugs = {
        l.strip().lower()
        for l in (root / "config" / "lexicons" / "clinical" / "drugs.txt").read_text(encoding="utf-8").splitlines()
        if l.strip()
    }
    clinical = set(drugs)
    clinical |= {
        l.strip().lower()
        for l in (root / "config" / "lexicons" / "clinical" / "symptoms_conditions.txt").read_text(encoding="utf-8").splitlines()
        if l.strip()
    }
    return {"safe_words": safe, "names": names, "drugs": drugs, "clinical_raw": clinical, "places": set()}


def _name_fully_covered(expected_name: str, txt: str, spans: list[Span]) -> bool:
    idx = txt.find(expected_name)
    if idx == -1:
        return False
    end = idx + len(expected_name)
    covered = [False] * len(txt)
    for sp in spans:
        for i in range(max(0, sp.start), min(len(txt), sp.end)):
            covered[i] = True
    alnum = [i for i in range(idx, end) if txt[i].isalnum()]
    return bool(alnum) and all(covered[i] for i in alnum)


def _clinical_untouched(expected_med: str, txt: str, spans: list[Span]) -> bool:
    idx = txt.find(expected_med)
    if idx == -1:
        return True
    end = idx + len(expected_med)
    return not any(not (sp.end <= idx or sp.start >= end) for sp in spans)


def run_all_experiments() -> dict[str, object]:
    """Run corrected E8, E14, E18, and E20 experiments."""
    lex = _load_lexicons()
    base_pol = {
        "detect": {
            "stack": "recall_first",
            "recall_threshold": 0.35,
            "precision_threshold": 0.55,
            "vehicle_plates": True,
            "doctor_reg_remove_digits": True,
            "default_deny": {"score": 0.45},
            "crf_weight": 0.25,
        }
    }

    # Genuinely unlabelled chaos prose (E18) — no labelled canary in denominator!
    unlabelled_cases = [
        ("Zoravar Singhania presented with fever and cough", "Zoravar Singhania", "fever"),
        ("देवेंद्र राठौड़ को बुखार है", "देवेंद्र राठौड़", "बुखार"),
        ("Mrinalini Deshpande advised Tab Dolo 650 mg BD", "Mrinalini Deshpande", "Dolo 650"),
        ("K/C/O DM2 x 9 yrs. Tushar Kulkarni reviewed.", "Tushar Kulkarni", "DM2"),
        ("Nikhilesh Banerjee had BP 120/80", "Nikhilesh Banerjee", "BP 120/80"),
    ]
    # Labelled + variant cases (for E8 & E14)
    labelled_cases = [
        ("Pt. Name: Kavita Joshi\nC/O fever", "Kavita Joshi"),
        ("नाम: सुरेश वर्मा\nदवा जारी", "सुरेश वर्मा"),
        ("Patiant_Naam: CanaryZeta123", "CanaryZeta123"),
    ]

    all_recs = [
        Record(key=(f"chaos/{i}.txt", "doc"), fields=[Field(0, "doc", txt)])
        for i, (txt, _, _) in enumerate(unlabelled_cases)
    ] + [
        Record(key=(f"lbl/{i}.txt", "doc"), fields=[Field(0, "doc", txt)])
        for i, (txt, _) in enumerate(labelled_cases)
    ]
    ctx_base = Context(policy=base_pol, lexicons=lex)
    promoted = induce_labels(all_recs, ctx_base)
    ctx_ind = Context(policy=base_pol, lexicons={**lex, "promoted_labels": promoted})

    # E18: Default-deny vs label-led on unlabelled chaos prose
    dd_hits = 0
    lbl_hits = 0
    med_untouched = 0
    for i, (txt, exp_name, exp_med) in enumerate(unlabelled_cases):
        rec = all_recs[i]
        sh = protect(rec, ctx_ind)
        full_sp = detect(rec, ctx_ind, sh)
        if _name_fully_covered(exp_name, txt, full_sp):
            dd_hits += 1
        if _clinical_untouched(exp_med, txt, full_sp):
            med_untouched += 1
        base_only = arbitrate(detect_ids(rec, ctx_ind) + detect_context(rec, ctx_ind), sh, ctx_ind)
        if _name_fully_covered(exp_name, txt, base_only):
            lbl_hits += 1

    e18 = {
        "default_deny_hits": dd_hits,
        "label_led_hits": lbl_hits,
        "total": len(unlabelled_cases),
        "default_deny_recall": dd_hits / len(unlabelled_cases),
        "label_led_recall": lbl_hits / len(unlabelled_cases),
        "medical_untouched": med_untouched,
        "medical_total": len(unlabelled_cases),
        "medical_kept": med_untouched / len(unlabelled_cases),
    }

    # E8: Independent ablation of default_deny and label induction against baseline (ids + context)
    def _score_mode(use_dd: bool, use_ind: bool) -> float:
        hits = 0
        targets = [(all_recs[i], unlabelled_cases[i][1]) for i in range(len(unlabelled_cases))] + [
            (all_recs[len(unlabelled_cases) + j], labelled_cases[j][1]) for j in range(len(labelled_cases))
        ]
        for rec, exp in targets:
            sh = protect(rec, ctx_ind)
            cands = detect_ids(rec, ctx_ind) + detect_context(rec, ctx_ind)
            if use_ind:
                cands += detect_labels(rec, ctx_ind)
            if use_dd:
                cands += detect_default_deny(rec, ctx_ind, sh, cands)
            sp = arbitrate(cands, sh, ctx_ind)
            if _name_fully_covered(exp, rec.fields[0].text, sp):
                hits += 1
        return hits / len(targets)

    e8 = {
        "baseline_ids_context": _score_mode(False, False),
        "plus_induction_only": _score_mode(False, True),
        "plus_default_deny_only": _score_mode(True, False),
        "plus_both": _score_mode(True, True),
        "crf_comparison": "not_run (python-crfsuite not installed; B15 cut per fallback)",
    }

    # E14: recall_first vs precision_first on identical corpus
    ctx_pf = Context(policy={"detect": {**base_pol["detect"], "stack": "precision_first"}}, lexicons={**lex, "promoted_labels": promoted})
    targets = [(all_recs[i], unlabelled_cases[i][1]) for i in range(len(unlabelled_cases))] + [
        (all_recs[len(unlabelled_cases) + j], labelled_cases[j][1]) for j in range(len(labelled_cases))
    ]
    rf_hits = sum(1 for rec, exp in targets if _name_fully_covered(exp, rec.fields[0].text, detect(rec, ctx_ind)))
    pf_hits = sum(1 for rec, exp in targets if _name_fully_covered(exp, rec.fields[0].text, detect(rec, ctx_pf)))
    e14 = {
        "total": len(targets),
        "recall_first_hits": rf_hits,
        "precision_first_hits": pf_hits,
        "recall_first": rf_hits / len(targets),
        "precision_first": pf_hits / len(targets),
    }

    # E20: Raw clinical set S vs S \ N on actual colliding words + non-colliding clinical controls
    collisions = ["Asha", "Kumari"]  # genuinely in S (symptoms_conditions.txt) AND in N!
    controls = ["fever", "hypertension"]  # clinical terms in S that must NOT be flagged
    ctx_raw_s = Context(policy=base_pol, lexicons={**lex, "safe_words": lex["clinical_raw"]})

    s_minus_n_hits = sum(
        1 for i, nm in enumerate(collisions)
        if _name_fully_covered(nm, f"{nm} visited", detect_default_deny(Record(key=(f"e20/{i}", "doc"), fields=[Field(0, "doc", f"{nm} visited")]), ctx_ind, []))
    )
    raw_s_hits = sum(
        1 for i, nm in enumerate(collisions)
        if _name_fully_covered(nm, f"{nm} visited", detect_default_deny(Record(key=(f"e20/{i}", "doc"), fields=[Field(0, "doc", f"{nm} visited")]), ctx_raw_s, []))
    )
    ctrl_kept = sum(
        1 for i, term in enumerate(controls)
        if _clinical_untouched(
            term,
            f"patient has {term}",
            detect_default_deny(Record(key=(f"e20c/{i}", "doc"), fields=[Field(0, "doc", f"patient has {term}")]), ctx_ind, []),
        )
    )
    e20 = {
        "collisions_tested": collisions,
        "with_name_subtraction_S_minus_N": s_minus_n_hits / len(collisions),
        "without_name_subtraction_raw_S": raw_s_hits / len(collisions),
        "clinical_controls_kept": ctrl_kept / len(controls),
    }

    return {"E8": e8, "E14": e14, "E18": e18, "E20": e20}


if __name__ == "__main__":
    print(json.dumps(run_all_experiments(), indent=2, ensure_ascii=False))
