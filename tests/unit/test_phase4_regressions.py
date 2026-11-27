"""Phase 4 §0: Regressions for rename collisions, filename signals, age/date, patient_ref, shield."""
from __future__ import annotations

import json
from pathlib import Path

from kavachmed import adapters, edits, evidence, inventory, rename, shield
from kavachmed.contracts import Edit, Field, PrivacyPlan, RecKey, Record, Resolution, Span
from kavachmed.security import load_keys


def test_destinations_are_unique_after_extension_selection() -> None:
    keys = load_keys({"RAKSHA_SECRET": "fixture"}, pbkdf2_iterations=1000)
    mapping = rename.allocate_output_paths(["report.bin", "report.txt"], keys)
    assert len({value.casefold() for value in mapping.values()}) == 2


def test_evidence_receives_filename_signals() -> None:
    key: RecKey = ("lab/7582322248_report.json", "/0")
    rec = Record(key=key, fields=[Field(idx=0, path="/0/note", text="ok")])
    sig = rename.extract_filename_signals(key[0])
    ev = evidence.extract(rec, [], filename_signals=sig)
    assert "7582322248" in ev.phones


def test_age_band_and_date_shifter_applied_before_numeric_surrogate() -> None:
    keys = load_keys({"RAKSHA_SECRET": "fixture"}, pbkdf2_iterations=1000)
    key: RecKey = ("p.json", "doc")
    rec = Record(key=key, fields=[
        Field(idx=0, path="/age", text="48", vtype="int"),
        Field(idx=1, path="/visit_date", text="2025-02-01", vtype="str"),
    ])
    res = Resolution(
        patient_of={key: "PSN-12345678"},
        cluster_of={key: "c1"},
        family_of={key: "fam1"},
        shift_days={"fam1": 35},
    )
    plan = PrivacyPlan(age_band={"c1": "40-49"})
    spans = [
        Span(key, 0, 0, 2, "AGE", "patient", 0.9, "context"),
        Span(key, 1, 0, 10, "DATE", "patient", 0.9, "context"),
    ]
    ed = edits.build(
        rec,
        spans,
        res,
        plan,
        keys=keys,
        date_shifter=lambda d, delta: f"SHIFTED({d},{delta})",
    )
    assert ed[0].kind == "BAND_AGE" and ed[0].text == "40-49"
    assert ed[1].kind == "SHIFT_DATE" and ed[1].text == "SHIFTED(2025-02-01,35)"


def test_json_existing_patient_ref_not_duplicated(tmp_path: Path) -> None:
    (tmp_path / "p.json").write_text('{"visits":[{"patient_ref":"OLD","v":1}]}', encoding="utf-8")
    parsed = adapters.parse(inventory.scan(tmp_path)[0], {})
    rk: RecKey = ("p.json", "/visits/0")
    rec = next(r for r in parsed.records if r.key == rk)
    val_f = next(f for f in rec.fields if f.text == "OLD")
    ed = [Edit(rk, val_f.idx, 0, 3, "PSN", "PSN-12345678")]
    assert parsed.projector is not None
    out_bytes, _ = parsed.projector(ed, {rk: "PSN-12345678"})
    pairs: list[tuple[str, object]] = []
    json.loads(out_bytes.decode("utf-8"), object_pairs_hook=lambda p: pairs.extend(p) or dict(p))
    keys_in_visit = [k for k, _ in pairs if k == "patient_ref"]
    assert keys_in_visit == ["patient_ref"]


def test_shield_doctor_registry_requires_ctx_evidence() -> None:
    key: RecKey = ("n.txt", "doc")
    rec = Record(key=key, fields=[Field(idx=0, path="doc", text="Dr. Alpha Beta reviewed")])
    no_reg = shield.protect(rec, {})
    assert not any(s.exempt_kind == "doctor_registry" for s in no_reg)

    with_reg = shield.protect(rec, {"doctor_registry": {"alpha beta"}})
    assert any(s.exempt_kind == "doctor_registry" for s in with_reg)

