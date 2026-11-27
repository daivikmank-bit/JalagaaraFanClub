"""Regression tests for every defect in plan-b-walkthrough-audit.md & plan-B-remediation.md."""

from __future__ import annotations

from pathlib import Path

from kavachmed.contracts import Context, Field, Record
from kavachmed.detect.arbiter import detect
from kavachmed.detect.labels import induce_labels
from kavachmed.propagate import build_index, sweep
from kavachmed.shield import protect
from kavachmed.typing import type_columns


def _ctx(places: set[str] | None = None, doctor_registry: set[str] | None = None, promoted: dict[str, str] | None = None) -> Context:
    root = Path(__file__).resolve().parents[2]
    safe = {l.strip().lower() for l in (root / "config" / "lexicons" / "safe" / "safe_words.txt").read_text(encoding="utf-8").splitlines() if l.strip()}
    names = {l.strip().lower() for l in (root / "config" / "lexicons" / "names" / "generic_names_seed.txt").read_text(encoding="utf-8").splitlines() if l.strip()}
    drugs = {l.strip().lower() for l in (root / "config" / "lexicons" / "clinical" / "drugs.txt").read_text(encoding="utf-8").splitlines() if l.strip()}
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
        lexicons={
            "safe_words": safe,
            "names": names,
            "drugs": drugs,
            "places": places or {"nashik", "पुणे"},
            "promoted_labels": promoted or {},
        },
        doctor_registry=doctor_registry or set(),
    )


def test_b1_uhid_header_and_tie_fallback():
    r1 = Record(key=("syn/t1.csv", "row:0"), fields=[Field(0, "uhid", "412345"), Field(1, "mix", "9876543210")])
    r2 = Record(key=("syn/t1.csv", "row:1"), fields=[Field(0, "uhid", "412346"), Field(1, "mix", "Velora Neris")])
    coltypes = type_columns([r1, r2])
    assert coltypes["uhid"] == "id"
    assert coltypes["mix"] == "free_text"

    ctx = _ctx()
    sp = detect(r1, ctx)
    assert any(s.field_idx == 0 and s.typ == "MRN" for s in sp)


def test_b2_b3_structural_id_mixed_phone_ipv6_serial_doc_reg():
    ctx = _ctx()
    rec = Record(
        key=("syn/ids.txt", "doc"),
        fields=[
            Field(0, "doc", "BP: phone 9876543210. ID AB/26/412345. device serial: zx9q77. IP 2001:db8::42. Dr. Varen Neth. Reg. No. 58371")
        ],
    )
    txt = rec.fields[0].text
    sp = detect(rec, ctx)
    got = {(txt[s.start : s.end], s.typ) for s in sp}
    assert ("9876543210", "PHONE") in got
    assert ("AB/26/412345", "MRN") in got
    assert ("zx9q77", "DEVICE_SERIAL") in got
    assert ("2001:db8::42", "IP_ADDRESS") in got
    assert ("58371", "DOCTOR_REG") in got


def test_b4_blank_registry_multiline_doctor_history_and_co():
    ctx = _ctx(doctor_registry={"", "   ", "Nevin"})
    rec = Record(
        key=("syn/sh.txt", "doc"),
        fields=[
            Field(0, "doc", "Dr. Nevin\nVelora Neris arrived. H/O asthma reviewed by patient Kaelen Vora. C/O velora neris.")
        ],
    )
    txt = rec.fields[0].text
    sh = protect(rec, ctx)
    sp = detect(rec, ctx, sh)
    covered = " ".join(txt[s.start : s.end] for s in sp)
    assert "Velora Neris" in covered
    assert "Kaelen Vora" in covered
    assert "Nevin" not in covered


def test_b6_roles_initials_timestamps_and_hinglish_duration():
    ctx = _ctx()
    rec = Record(
        key=("syn/ctx.txt", "doc"),
        fields=[
            Field(0, "doc", "Kamal Bhaiya: send report\n[10:32] Velora Neris: arrived\nPatient Name: K Rao\nZarela ko 4 din se bukhar hai")
        ],
    )
    txt = rec.fields[0].text
    sp = detect(rec, ctx)
    items = [(txt[s.start : s.end], s.typ, s.role) for s in sp]
    assert any("Kamal Bhaiya" in t and r == "relative" for t, _, r in items)
    assert any("Velora Neris" in t and r == "sender" for t, _, r in items)
    assert any(t == "K Rao" and typ == "PATIENT_NAME" and r == "patient" for t, typ, r in items)
    assert any(t == "Zarela" and r == "patient" for t, _, r in items)


def test_b8_b9_psn_placeholders_deva_digits_districts_inr_and_ward():
    ctx = _ctx()
    rec1 = Record(key=("syn/p1.txt", "doc"), fields=[Field(0, "doc", "PSN-ABCD1234 [NAME] dose ७५ INR 412345")])
    assert detect(rec1, ctx) == []

    rec2 = Record(key=("syn/p2.txt", "doc"), fields=[Field(0, "doc", "Village Belora, District Nashik")])
    sp2 = detect(rec2, ctx)
    assert any(rec2.fields[0].text[s.start : s.end] == "Village Belora" and s.typ == "VILLAGE" for s in sp2)
    assert not any("Nashik" in rec2.fields[0].text[s.start : s.end] for s in sp2)

    rec3 = Record(key=("syn/p3.txt", "doc"), fields=[Field(0, "doc", "जिला पुणे")])
    assert detect(rec3, ctx) == []

    rec4 = Record(key=("syn/p4.csv", "row:0"), fields=[Field(0, "address", "Ward 12")])
    type_columns([rec4])
    sp4 = detect(rec4, ctx)
    assert any(s.typ == "LOCALITY" and rec4.fields[0].text[s.start : s.end] == "Ward 12" for s in sp4)


def test_b10_b11_b14_hidden_offsets_fullwidth_and_trimmed_labels():
    ctx = _ctx(promoted={"name": "PATIENT_NAME", "father name": "RELATIVE_NAME"})
    rec = Record(
        key=("syn/h.txt", "doc"),
        fields=[
            Field(0, "doc", "BP 120/80; contact audit%40example.org\nＺｅｌｏｒａ\nName: ze99 / BP 120/80\nPhone\nName: zq9912"),
            Field(1, "father name", "Mohanlal Vora"),
        ],
    )
    sp = detect(rec, ctx)
    f0 = rec.fields[0].text
    f0_spans = [(f0[s.start : s.end], s.typ) for s in sp if s.field_idx == 0]
    assert ("audit%40example.org", "EMAIL") in f0_spans
    assert not any("BP 120/80" in t for t, _ in f0_spans)
    assert any("Ｚｅｌｏｒａ" in t for t, _ in f0_spans)
    assert any("ze99" in t for t, _ in f0_spans)
    assert any("zq9912" in t for t, _ in f0_spans)

    f1_spans = [s for s in sp if s.field_idx == 1]
    assert any(s.typ == "RELATIVE_NAME" and s.role == "relative" for s in f1_spans)


def test_b12_sweep_entrypoint_cross_record_and_lab_guard():
    ctx = _ctx()
    r1 = Record(key=("syn/a.txt", "doc"), fields=[Field(0, "doc", "Phone 9876543210. Relative: Velora Neris. Aadhaar: 2345 6789 0124.")])
    r2 = Record(key=("syn/b.txt", "doc"), fields=[Field(0, "doc", "Call (987) 654.3210. velora neris came. LabSeq 234567890124 mmol/L.")])
    r3 = Record(key=("syn/c.txt", "doc"), fields=[Field(0, "doc", "velora neris unrelated cluster.")])

    sh1, sh2, sh3 = protect(r1, ctx), protect(r2, ctx), protect(r3, ctx)
    sp1, sp2, sp3 = detect(r1, ctx, sh1), detect(r2, ctx, sh2), detect(r3, ctx, sh3)

    class _Res:
        cluster_of = {r1.key: "C1", r2.key: "C1", r3.key: "C2"}
        family_of = {}

    res = _Res()
    build_index([r1, r2, r3], res, {r1.key: sp1, r2.key: sp2, r3.key: sp3})

    # Call the planned single-record entry point sweep(rec, resolution, spans)
    swept2 = sweep(r2, res, sp2, protected_spans=sh2)
    f2 = r2.fields[0].text
    prop_spans2 = [(f2[s.start : s.end], s.typ, s.src) for s in swept2 if s.src == "propagate"]
    assert any("654.3210" in t and typ == "PHONE" for t, typ, _ in prop_spans2)
    assert any("velora neris" in t.lower() for t, _, _ in prop_spans2)
    # Protected LabSeq must NOT be overwritten by Aadhaar digit sweep!
    assert not any("234567890124" in f2[s.start : s.end] for s in swept2)

    # Unrelated cluster C2 must NOT get C1's name sweep
    swept3 = sweep(r3, res, sp3, protected_spans=sh3)
    assert not any(s.src == "propagate" for s in swept3)
