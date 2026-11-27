"""End-to-end acceptance tests for Plan B Phase 1, Phase 2, and Phase 3."""

from __future__ import annotations

import base64
import json
from pathlib import Path

from kavachmed.contracts import Context, Field, Record
from kavachmed.detect.arbiter import detect
from kavachmed.detect.crf import detect_crf
from kavachmed.detect.labels import induce_labels, write_promoted_labels
from kavachmed.propagate import build_index, sweep
from kavachmed.shield import protect
from kavachmed.typing import type_columns
from tools.detect_report import compare_profiles
from tools.fairness_audit import run_fairness_audit
from tools.run_b_experiments import run_all_experiments


def _load_ctx(stack: str = "recall_first", promoted: dict[str, str] | None = None) -> Context:
    root = Path(__file__).resolve().parents[2]
    safe = {l.strip().lower() for l in (root / "config" / "lexicons" / "safe" / "safe_words.txt").read_text(encoding="utf-8").splitlines() if l.strip()}
    names = {l.strip().lower() for l in (root / "config" / "lexicons" / "names" / "generic_names_seed.txt").read_text(encoding="utf-8").splitlines() if l.strip()}
    drugs = {l.strip().lower() for l in (root / "config" / "lexicons" / "clinical" / "drugs.txt").read_text(encoding="utf-8").splitlines() if l.strip()}
    return Context(
        policy={
            "detect": {
                "stack": stack,
                "recall_threshold": 0.35,
                "precision_threshold": 0.55,
                "vehicle_plates": True,
                "doctor_reg_remove_digits": True,
                "default_deny": {"score": 0.45},
                "crf_weight": 0.25,
            }
        },
        lexicons={
            "safe_words": safe,
            "names": names,
            "drugs": drugs,
            "places": {"nashik", "sitapur", "पुणे"},
            "promoted_labels": promoted or {},
        },
        doctor_registry={"Ramesh Srivastava", "Nevin"},
    )


def test_phase1_acceptance():
    """Phase 1 (B1-B6, B8-B9): typing, IDs, shield, context, default-deny, address."""
    ctx = _load_ctx()
    records = [
        Record(
            key=("syn/p1.csv", "row:0"),
            fields=[
                Field(0, "uhid", "412345"),
                Field(1, "address", "Ward 12, Village Belora, District Nashik, PIN 422001"),
                Field(
                    2,
                    "note",
                    "Dr. Nevin\nVelora Neris arrived. BP: phone 9876543210. ID AB/26/412345. "
                    "device serial: zx9q77. IP 2001:db8::42. Dr. Varen Neth. Reg. No. 58371. "
                    "H/O asthma reviewed by patient Kaelen Vora. C/O fever for 3 days. "
                    "Zarela ko 4 din se bukhar hai. Kamal Bhaiya: send report. "
                    "PSN-ABCD1234 [NAME] dose ७५ INR 412345.",
                ),
            ],
        )
    ]
    coltypes = type_columns(records)
    assert coltypes["uhid"] == "id"
    assert coltypes["address"] == "address"

    sp = detect(records[0], ctx)
    f0 = records[0].fields[0].text
    f1 = records[0].fields[1].text
    f2 = records[0].fields[2].text

    assert any(s.field_idx == 0 and s.typ == "MRN" for s in sp)
    assert any(s.field_idx == 1 and s.typ == "LOCALITY" and "Ward 12" in f1[s.start : s.end] for s in sp)
    assert any(s.field_idx == 1 and s.typ == "VILLAGE" for s in sp)
    assert any(s.field_idx == 1 and s.typ == "PIN" for s in sp)
    assert not any(s.field_idx == 1 and "Nashik" in f1[s.start : s.end] for s in sp)

    got_f2 = {(f2[s.start : s.end], s.typ) for s in sp if s.field_idx == 2}
    assert ("9876543210", "PHONE") in got_f2
    assert ("AB/26/412345", "MRN") in got_f2
    assert ("zx9q77", "DEVICE_SERIAL") in got_f2
    assert ("2001:db8::42", "IP_ADDRESS") in got_f2
    assert ("58371", "DOCTOR_REG") in got_f2
    covered_f2 = " ".join(t for t, _ in got_f2)
    assert "Velora Neris" in covered_f2
    assert "Kaelen Vora" in covered_f2
    assert "Zarela" in covered_f2
    for token in ("Kamal", "Bhaiya"):
        assert token in covered_f2
    for kept in ("Nevin", "fever", "PSN-ABCD1234", "[NAME]", "७५", "INR 412345"):
        start = f2.index(kept)
        end = start + len(kept)
        assert not any(s.field_idx == 2 and s.start < end and start < s.end for s in sp)


def test_phase2_acceptance():
    """Phase 2 (B7, B10-B13): hidden channels, arbitration, propagation, fairness."""
    ctx = _load_ctx()
    b64_blob = base64.b64encode(b"Phone: 9123456789").decode("ascii")
    r1 = Record(
        key=("syn/p2_a.txt", "doc"),
        fields=[
            Field(
                0,
                "doc",
                f"BP 120/80; contact audit%40example.org\nＺｅｌｏｒａ\nblob={b64_blob}\n"
                "UHID: ABC/29/847291. Relative: Velora Neris. Phone 9876543210.",
            )
        ],
    )
    r2 = Record(
        key=("syn/p2_b.txt", "doc"),
        fields=[
            Field(
                0,
                "doc",
                "Ref 847291. Call (987) 654.3210. velora neris arrived. "
                "Dr. Ramesh Srivastava reviewed. LabSeq 9123456789 mmol/L.",
            )
        ],
    )

    sh1, sh2 = protect(r1, ctx), protect(r2, ctx)
    sp1, sp2 = detect(r1, ctx, sh1), detect(r2, ctx, sh2)
    f1 = r1.fields[0].text
    f1_spans = [(f1[s.start : s.end], s.typ) for s in sp1]
    assert ("audit%40example.org", "EMAIL") in f1_spans
    assert not any("BP 120/80" in t for t, _ in f1_spans)
    assert any("Ｚｅｌｏｒａ" in t for t, _ in f1_spans)

    class _Res:
        cluster_of = {r1.key: "C1", r2.key: "C1"}
        family_of = {}

    res = _Res()
    build_index([r1, r2], res, {r1.key: sp1, r2.key: sp2})
    swept2 = sweep(r2, res, sp2, protected_spans=sh2)
    f2 = r2.fields[0].text
    prop2 = [(f2[s.start : s.end], s.typ) for s in swept2 if s.src == "propagate"]
    assert any("847291" in t and typ == "MRN" for t, typ in prop2)
    assert any("654.3210" in t and typ == "PHONE" for t, typ in prop2)
    assert any("velora neris" in t.lower() for t, _ in prop2)
    assert not any("Srivastava" in f2[s.start : s.end] for s in swept2)
    assert not any("9123456789" in f2[s.start : s.end] for s in swept2)

    prof = compare_profiles([r1], ctx)
    assert prof["recall_first"]["by_src"].get("default_deny", 0) >= 1
    assert prof["precision_first"]["by_src"].get("default_deny", 0) == 0

    fair = run_fairness_audit(ctx)
    assert fair["passed"] and fair["max_gap"] <= 0.02


def test_phase3_acceptance(tmp_path: Path):
    """Phase 3 (B14-B15, E8/E14/E18/E20): label induction, CRF cut, experiments."""
    base_ctx = _load_ctx()
    records = [
        Record(key=("syn/p3_1.txt", "doc"), fields=[Field(0, "doc", "CustomKin: Aarav Mehta\nPatiant_Naam: ze99 / BP 120/80")]),
        Record(key=("syn/p3_2.txt", "doc"), fields=[Field(0, "doc", "CustomKin: Priya Nair")]),
        Record(key=("syn/p3_3.txt", "doc"), fields=[Field(0, "doc", "CustomKin: Rohan Joshi")]),
        Record(key=("syn/p3_4.txt", "doc"), fields=[Field(0, "doc", "Phone\nName: zq9912")]),
    ]
    promoted = induce_labels(records, base_ctx)
    assert "customkin" in promoted
    assert "patiant naam" in promoted

    artefact = tmp_path / "promoted_labels.json"
    write_promoted_labels(promoted, artefact)
    loaded = json.loads(artefact.read_text(encoding="utf-8"))
    assert loaded["schema"] == "kavachmed-promoted-labels/1"

    ctx = _load_ctx(promoted=promoted)
    sp0 = detect(records[0], ctx)
    t0 = records[0].fields[0].text
    assert any("ze99" in t0[s.start : s.end] for s in sp0)
    assert not any("BP 120/80" in t0[s.start : s.end] for s in sp0)

    sp3 = detect(records[3], ctx)
    t3 = records[3].fields[0].text
    assert any("zq9912" in t3[s.start : s.end] for s in sp3)

    assert detect_crf(records[0], ctx, model_path=None) == []

    exp = run_all_experiments()
    assert exp["E8"]["plus_both"] >= exp["E8"]["baseline_ids_context"]
    assert exp["E14"]["recall_first"] >= exp["E14"]["precision_first"]
    assert exp["E18"]["default_deny_recall"] >= 0.95
    assert exp["E18"]["medical_kept"] >= 0.99
    assert exp["E20"]["with_name_subtraction_S_minus_N"] == 1.0
    assert exp["E20"]["without_name_subtraction_raw_S"] == 0.0
    assert exp["E20"]["clinical_controls_kept"] == 1.0
