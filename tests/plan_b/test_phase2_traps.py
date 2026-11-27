"""Unit tests for Plan B Phase 2 (B7, B10, B11, B12, B13) using synthetic data only."""

from __future__ import annotations

import base64
from pathlib import Path

from kavachmed.contracts import Context, Field, Record, Span
from kavachmed.detect.arbiter import arbitrate, detect
from kavachmed.detect.gazetteer import detect_gazetteer, is_place_token
from kavachmed.detect.hidden import detect_hidden, normalize_with_offset_map
from kavachmed.propagate import sweep_corpus
from kavachmed.shield import protect
from tools.detect_report import compare_profiles
from tools.fairness_audit import run_fairness_audit


def _ctx(**extra_lex) -> Context:
    safe_path = Path(__file__).resolve().parents[2] / "config" / "lexicons" / "safe" / "safe_words.txt"
    names_path = Path(__file__).resolve().parents[2] / "config" / "lexicons" / "names" / "generic_names_seed.txt"
    safe_words = {l.strip().lower() for l in safe_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    names = {l.strip().lower() for l in names_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    lex = {"safe_words": safe_words, "names": names, "places": {"sitapur", "lucknow"}, **extra_lex}
    return Context(
        policy={
            "detect": {
                "stack": "recall_first",
                "recall_threshold": 0.35,
                "precision_threshold": 0.55,
                "vehicle_plates": True,
                "doctor_reg_remove_digits": True,
                "default_deny": {"score": 0.45},
            }
        },
        lexicons=lex,
        doctor_registry={"Ramesh Srivastava"},
    )


def test_b7_gazetteer_feature_never_a_gate():
    ctx = _ctx()
    assert is_place_token("Sitapur", ctx)
    rec = Record(key=("syn/g.txt", "doc"), fields=[Field(0, "doc", "Visited Sitapur with Aarav Mehta")])
    gaz_spans = detect_gazetteer(rec, ctx)
    texts = [rec.fields[0].text[s.start : s.end] for s in gaz_spans]
    assert any("Aarav Mehta" in t for t in texts)
    assert not any("Sitapur" in t for t in texts)

    # Unseen name still caught by full detect() via default-deny
    unseen = Record(key=("syn/g2.txt", "doc"), fields=[Field(0, "doc", "Xylophar Zindabad visited clinic")])
    all_spans = detect(unseen, ctx)
    assert any("Xylophar Zindabad" in unseen.fields[0].text[s.start : s.end] for s in all_spans)


def test_b10_hidden_channels_b64_and_homoglyph_offset_map():
    ctx = _ctx()
    b64_payload = base64.b64encode(b"Patient Name: Kavita Joshi Phone: 9876543210").decode("ascii")
    rec = Record(
        key=("syn/Aarav_Mehta_9876543210.csv", "row:0"),
        fields=[
            Field(0, "path", "syn/Aarav_Mehta_9876543210.csv", vtype="path"),
            Field(1, "header", "patient_name", vtype="header"),
            Field(2, "consent", f"blob={b64_payload}", vtype="str"),
            Field(3, "note", "Р\u200bАN: ABCDE1234F", vtype="str"),  # Cyrillic Р, А + zero-width space
        ],
    )
    norm, omap = normalize_with_offset_map("Р\u200bАN")
    assert norm == "PAN"
    assert len(omap) == 3

    spans = detect_hidden(rec, ctx)
    assert any(s.field_idx == 0 and s.typ == "PHONE" for s in spans)
    # Integration contract: a header that only NAMES a PII field is schema and stays intact
    assert not any(s.field_idx == 1 and not s.protected for s in spans)
    # ...but a header that CONTAINS PII values is still caught
    rec_h = Record(key=("syn/w.csv", "@header"), fields=[Field(0, "header", "SNJ-0099123 Bhoomika", vtype="header")])
    hspans = detect_hidden(rec_h, ctx)
    assert any(s.typ == "MRN" for s in hspans) and any("NAME" in s.typ for s in hspans)
    assert any(s.field_idx == 2 and s.typ == "PHONE" for s in spans)
    assert any(s.field_idx == 3 and s.typ == "PAN" for s in spans)


def test_b11_arbiter_precedence_and_profiles():
    ctx_rec = _ctx()
    rec = Record(
        key=("syn/arb.txt", "doc"),
        fields=[Field(0, "doc", "Dr. Ramesh Srivastava prescribed Dolo 650. Zoravar Singhania called 9876543210.")],
    )
    sh = protect(rec, ctx_rec)
    spans_rf = detect(rec, ctx_rec, sh)
    texts_rf = [rec.fields[0].text[s.start : s.end] for s in spans_rf]
    assert any("Zoravar Singhania" in t for t in texts_rf)
    assert any("9876543210" in t for t in texts_rf)
    assert not any("Ramesh Srivastava" in t for t in texts_rf)

    # precision_first drops default-deny
    comp = compare_profiles([rec], ctx_rec)
    assert comp["recall_first"]["by_src"].get("default_deny", 0) >= 1
    assert comp["precision_first"]["by_src"].get("default_deny", 0) == 0


def test_b12_cluster_propagation_keeps_dr_ramesh_sweeps_relative():
    ctx = _ctx()
    r1 = Record(
        key=("syn/f1.txt", "doc"),
        fields=[Field(0, "doc", "UHID: ABC/29/847291. Relative: Ramesh Patel.")],
    )
    r2 = Record(
        key=("syn/f2.txt", "doc"),
        fields=[Field(0, "doc", "Ref 847291. Dr. Ramesh Srivastava reviewed. Attendant Ramesh arrived.")],
    )
    sh1, sh2 = protect(r1, ctx), protect(r2, ctx)
    sp1, sp2 = detect(r1, ctx, sh1), detect(r2, ctx, sh2)

    cluster_of = {r1.key: "C1", r2.key: "C1"}
    swept = sweep_corpus([r1, r2], cluster_of, {r1.key: sp1, r2.key: sp2}, {r1.key: sh1, r2.key: sh2})

    r2_texts = [(r2.fields[0].text[s.start : s.end], s.typ) for s in swept[r2.key]]
    assert any("847291" in t and typ == "MRN" for t, typ in r2_texts)
    # Attendant Ramesh is swept, Dr. Ramesh Srivastava is protected!
    assert any(t == "Ramesh" for t, _ in r2_texts)
    assert not any("Srivastava" in t for t, _ in r2_texts)


def test_b13_fairness_audit_gap_under_2_pts():
    res = run_fairness_audit(_ctx())
    assert res["passed"], f"Fairness gap too high: {res}"
