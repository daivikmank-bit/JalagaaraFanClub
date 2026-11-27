"""B13: Script and community fairness audit on synthetic multilingual fixtures."""

from __future__ import annotations

from kavachmed.contracts import Context, Field, Record
from kavachmed.detect.arbiter import detect
from kavachmed.shield import protect


def run_fairness_audit(ctx: Context) -> dict[str, object]:
    """Evaluate synthetic recall across scripts (Latin, Devanagari, Hinglish) and communities."""
    cases = [
        # (script, community, text, expected_name)
        ("latin", "hindu", "Pt. Name: Aarav Kulkarni\nC/O fever for 2 days", "Aarav Kulkarni"),
        ("latin", "muslim", "Pt. Name: Farhan Siddiqui\nTab Dolo 650 mg BD", "Farhan Siddiqui"),
        ("latin", "sikh", "Pt. Name: Bhavya Kaur\nK/C/O asthma", "Bhavya Kaur"),
        ("latin", "christian", "Pt. Name: Mary Thomas\nBP 120/80", "Mary Thomas"),
        ("devanagari", "hindu", "नाम: राजेश तिवारी\nबुखार 3 दिन से", "राजेश तिवारी"),
        ("devanagari", "muslim", "नाम: इमरान अंसारी\nजांच सामान्य", "इमरान अंसारी"),
        ("devanagari", "sikh", "नाम: हरप्रीत सिंह\nदवा जारी", "हरप्रीत सिंह"),
        ("devanagari", "christian", "नाम: मैरी थॉमस\nसलाह आराम", "मैरी थॉमस"),
        ("hinglish", "hindu", "Vikram ko 4 din se bukhar hai", "Vikram"),
        ("hinglish", "muslim", "Zoya ka BP 130/80 hai", "Zoya"),
        ("hinglish", "sikh", "Jaspreet ji ko khansi hai", "Jaspreet"),
        ("hinglish", "christian", "Thomas ka report normal hai", "Thomas"),
    ]

    by_script: dict[str, list[bool]] = {}
    by_community: dict[str, list[bool]] = {}
    for idx, (scr, comm, text, expected) in enumerate(cases):
        rec = Record(key=(f"fairness/{scr}_{comm}_{idx}.txt", "doc"), fields=[Field(idx=0, path="doc", text=text)])
        sh = protect(rec, ctx)
        spans = detect(rec, ctx, sh)
        detected_texts = [text[s.start : s.end] for s in spans]
        hit = any(expected in dt for dt in detected_texts)
        by_script.setdefault(scr, []).append(hit)
        by_community.setdefault(comm, []).append(hit)

    script_recalls = {k: {"hits": sum(v), "total": len(v), "recall": sum(v) / len(v)} for k, v in sorted(by_script.items())}
    comm_recalls = {k: {"hits": sum(v), "total": len(v), "recall": sum(v) / len(v)} for k, v in sorted(by_community.items())}

    s_vals = [v["recall"] for v in script_recalls.values()]
    c_vals = [v["recall"] for v in comm_recalls.values()]
    script_gap = (max(s_vals) - min(s_vals)) if s_vals else 0.0
    comm_gap = (max(c_vals) - min(c_vals)) if c_vals else 0.0
    max_gap = max(script_gap, comm_gap)

    return {
        "by_script": script_recalls,
        "by_community": comm_recalls,
        "script_gap": script_gap,
        "community_gap": comm_gap,
        "max_gap": max_gap,
        "passed": max_gap <= 0.02,
    }
