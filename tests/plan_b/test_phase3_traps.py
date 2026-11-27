"""Unit tests for Plan B Phase 3 (B14, B15, E8, E14, E18, E20) using synthetic data only."""

from __future__ import annotations

import json
from pathlib import Path

from kavachmed.contracts import Context, Field, Record
from kavachmed.detect.arbiter import detect
from kavachmed.detect.crf import detect_crf
from kavachmed.detect.labels import detect_labels, induce_labels, write_promoted_labels
from tools.run_b_experiments import run_all_experiments


def _ctx() -> Context:
    safe_path = Path(__file__).resolve().parents[2] / "config" / "lexicons" / "safe" / "safe_words.txt"
    safe_words = {l.strip().lower() for l in safe_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    return Context(
        policy={
            "detect": {
                "stack": "recall_first",
                "recall_threshold": 0.35,
                "precision_threshold": 0.55,
                "default_deny": {"score": 0.45},
                "crf_weight": 0.25,
            }
        },
        lexicons={"safe_words": safe_words},
    )


def test_b14_label_induction_and_artefact(tmp_path: Path):
    ctx = _ctx()
    records = [
        Record(key=("syn/1.txt", "doc"), fields=[Field(0, "doc", "CustomGuardian: Aarav Mehta\nPatiant_Naam: CanaryAlpha")]),
        Record(key=("syn/2.txt", "doc"), fields=[Field(0, "doc", "CustomGuardian: Priya Nair")]),
        Record(key=("syn/3.txt", "doc"), fields=[Field(0, "doc", "CustomGuardian: Rohan Joshi")]),
        Record(key=("syn/4.txt", "doc"), fields=[Field(0, "doc", "CustomGuardian: CanaryToken99")]),
    ]
    promoted = induce_labels(records, ctx)
    assert "customguardian" in promoted
    assert "patiant naam" in promoted

    artefact = tmp_path / "promoted_labels.json"
    write_promoted_labels(promoted, artefact)
    data = json.loads(artefact.read_text(encoding="utf-8"))
    assert data["schema"] == "kavachmed-promoted-labels/1"
    assert "customguardian" in data["labels"]

    # Canary under promoted label is caught even though it's alphanumeric
    spans = detect_labels(records[3], ctx, promoted)
    assert any("CanaryToken99" in records[3].fields[0].text[s.start : s.end] for s in spans)


def test_b15_crf_abstains_gracefully_when_cut():
    ctx = _ctx()
    rec = Record(key=("syn/crf.txt", "doc"), fields=[Field(0, "doc", "Kavita Joshi visited")])
    assert detect_crf(rec, ctx, model_path=None) == []


def test_e8_e14_e18_e20_experiments():
    res = run_all_experiments()
    assert res["E18"]["default_deny_recall"] >= 0.95
    assert res["E18"]["medical_kept"] >= 0.99
    assert res["E14"]["recall_first"] >= res["E14"]["precision_first"]
    assert res["E20"]["with_name_subtraction_S_minus_N"] == 1.0
    assert res["E20"]["without_name_subtraction_raw_S"] < 1.0
    assert res["E20"]["clinical_controls_kept"] == 1.0
