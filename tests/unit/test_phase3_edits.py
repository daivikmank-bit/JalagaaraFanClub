"""Phase 3 Steps 2-4 (A8): Span resolution, PSN injection, surrogates, and clinical LocMap."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from kavachmed import adapters, edits, inventory
from kavachmed.contracts import PrivacyPlan, RecKey, Record, Field, Resolution, Span
from kavachmed.security import StageError, load_keys


def test_span_priority_permutation_invariance_and_protected_carve() -> None:
    key: RecKey = ("note.txt", "doc")
    rec = Record(key=key, fields=[Field(idx=0, path="doc", text="xx 9876543210 1-0-1 yy")])
    res = Resolution(patient_of={key: None}, cluster_of={key: None}, family_of={key: "note.txt"}, shift_days={"note.txt": 45})
    plan = PrivacyPlan()

    shield = Span(key, 0, 14, 19, "CLINICAL", "none", 1.0, "shield", protected=True, exempt_kind="dose")
    wide_dd = Span(key, 0, 3, 22, "REDACTED", "unknown", 0.5, "default_deny")
    phone_id = Span(key, 0, 3, 13, "PHONE", "patient", 0.95, "ids")

    e1 = edits.build(rec, [shield, wide_dd, phone_id], res, plan)
    e2 = edits.build(rec, [phone_id, shield, wide_dd], res, plan)
    assert e1 == e2

    # Protected 1-0-1 (14..19) must never be intersected by any edit
    for ed in e1:
        assert not (ed.start < 19 and 14 < ed.end)
    assert any(ed.start == 3 and ed.end == 13 and ed.text == "[PHONE]" for ed in e1)


def test_linked_psn_vs_relative_and_unlinked() -> None:
    key_linked: RecKey = ("a.txt", "doc")
    key_unlinked: RecKey = ("b.txt", "doc")
    rec_a = Record(key=key_linked, fields=[Field(idx=0, path="doc", text="Pt: Alpha S/o Beta")])
    rec_b = Record(key=key_unlinked, fields=[Field(idx=0, path="doc", text="Pt: Gamma")])

    res = Resolution(
        patient_of={key_linked: "PSN-12345678", key_unlinked: None},
        cluster_of={key_linked: "c1", key_unlinked: None},
        family_of={key_linked: "a.txt", key_unlinked: "b.txt"},
        shift_days={"a.txt": 40, "b.txt": -40},
    )
    plan = PrivacyPlan()

    spans_a = [
        Span(key_linked, 0, 4, 9, "PATIENT_NAME", "patient", 0.9, "context"),
        Span(key_linked, 0, 14, 18, "RELATIVE_NAME", "relative", 0.9, "context"),
    ]
    spans_b = [Span(key_unlinked, 0, 4, 9, "PATIENT_NAME", "patient", 0.9, "context")]

    ea = edits.build(rec_a, spans_a, res, plan)
    eb = edits.build(rec_b, spans_b, res, plan)
    assert [e.text for e in ea] == ["PSN-12345678", "[RELATIVE]"]
    assert [e.text for e in eb] == ["[NAME]"]


def test_surrogate_integer_deterministic_and_distinct() -> None:
    keys = load_keys({"RAKSHA_SECRET": "practice"}, pbkdf2_iterations=1000)
    s1 = edits.surrogate_integer(keys, "aadhar_no", "123456789012")
    s2 = edits.surrogate_integer(keys, "aadhar_no", "123456789012")
    assert s1 == s2
    assert s1 != "123456789012"
    assert len(s1) == 12 and s1.isdecimal() and not s1.startswith("0")

    with pytest.raises(StageError):
        edits.surrogate_integer(keys, "aadhar_no", "0123")


def test_psn_injection_fallbacks_across_formats(tmp_path: Path) -> None:
    # 1. CSV fallback: adds patient_ref column when no PSN edit exists
    (tmp_path / "t.csv").write_bytes(b"code,val\n1,ok\n2,ok\n")
    p_csv = adapters.parse(next(e for e in inventory.scan(tmp_path) if e.rel_path == "t.csv"), {})
    psn_map = {("t.csv", "row:0"): "PSN-AAAAAAAA", ("t.csv", "row:1"): None}
    assert p_csv.projector is not None
    out_csv, _ = p_csv.projector([], psn_map)
    assert out_csv.decode("utf-8") == "code,val,patient_ref\n1,ok,PSN-AAAAAAAA\n2,ok,\n"

    # 2. JSON fallback: inserts patient_ref into empty {} and non-empty {}
    (tmp_path / "j.json").write_text('{"visits":[{},{"v":1}]}', encoding="utf-8")
    p_json = adapters.parse(next(e for e in inventory.scan(tmp_path) if e.rel_path == "j.json"), {})
    psn_json = {("j.json", "/visits/0"): "PSN-BBBBBBBB", ("j.json", "/visits/1"): None}
    assert p_json.projector is not None
    out_json, _ = p_json.projector([], psn_json)
    assert json.loads(out_json) == {"visits": [{"patient_ref": "PSN-BBBBBBBB"}, {"v": 1}]}

    # 3. Text fallback: prepends Patient: PSN-... preserving CRLF
    (tmp_path / "n.txt").write_bytes(b"Fever 1-0-1\r\n")
    p_txt = adapters.parse(next(e for e in inventory.scan(tmp_path) if e.rel_path == "n.txt"), {})
    assert p_txt.projector is not None
    out_txt, _ = p_txt.projector([], {("n.txt", "doc"): "PSN-CCCCCCCC"})
    assert out_txt.decode("utf-8") == "Patient: PSN-CCCCCCCC\r\nFever 1-0-1\r\n"


def test_clinical_ranges_projected_accurately_with_multibyte_and_escapes(tmp_path: Path) -> None:
    # CSV with multibyte, doubled quote, and length-changing replacement before protected dose
    src = 'note\n"नाम ""x"" 1-0-1"\n'
    (tmp_path / "c.csv").write_text(src, encoding="utf-8")
    parsed = adapters.parse(inventory.scan(tmp_path)[0], {})
    row0 = next(r for r in parsed.records if r.key[1] == "row:0")
    txt = row0.fields[0].text
    dose_start = txt.index("1-0-1")
    shield = Span(row0.key, 0, dose_start, dose_start + 5, "CLINICAL", "none", 1.0, "shield", protected=True, exempt_kind="dose")
    name_start = txt.index("नाम")
    ed = edits.build(row0, [shield, Span(row0.key, 0, name_start, name_start + len("नाम"), "PATIENT_NAME", "patient", 0.9, "context")],
                     Resolution(patient_of={row0.key: "PSN-12345678"}, cluster_of={}, family_of={}, shift_days={}), PrivacyPlan())
    assert parsed.projector is not None
    out_bytes, loc_map = parsed.projector(ed, {row0.key: "PSN-12345678"}, protected_spans=[shield])
    output = out_bytes.decode("utf-8")
    start = output.index("1-0-1")
    dose_ranges = [(a, b) for a, b, kind in loc_map.clinical_ranges() if kind == "dose"]
    assert dose_ranges == [(start, start + 5)]
