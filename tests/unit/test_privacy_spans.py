"""Owner D: record-level privacy spans wired into edits (D1 dates, D3 ages, D4 districts)."""
from __future__ import annotations

from kavachmed import dates, residual_b
from kavachmed.contracts import Field, PrivacyPlan, Record, Resolution
from kavachmed.edits import build
from kavachmed.privacy import spans as ps

POLICY = {"dates": {"default_order": "dmy", "dataset_ref_year": 2026}, "location": {"keep": "district"}}


def _rec(*fields: tuple[str, str, str], loc: str = "row:0") -> Record:
    return Record(
        key=("registration/x.csv", loc),
        fields=[Field(idx=i, path=p, text=t, vtype=v) for i, (p, t, v) in enumerate(fields)],
    )


def _kinds(rec: Record) -> dict[str, str]:
    return {rec.fields[s.field_idx].path: s.typ for s in ps.privacy_spans(rec, POLICY, "dmy")}


def test_dob_date_age_and_district_spans() -> None:
    rec = _rec(("Reg Date", "12/03/2026", "str"), ("DOB", "01/02/1980", "str"),
               ("Age", "46", "str"), ("District", "Lucknow", "str"))
    assert _kinds(rec) == {"Reg Date": "DATE", "DOB": "DOB", "Age": "AGE", "District": "DISTRICT"}
    dist = [s for s in ps.privacy_spans(rec, POLICY, "dmy") if s.typ == "DISTRICT"][0]
    assert dist.protected and dist.exempt_kind == "district"


def test_durations_are_not_ages_and_districts_need_context() -> None:
    rec = _rec(("notes", "K/C/O DM2 x 12 yrs, seen at Lucknow OPD", "str"))
    assert _kinds(rec) == {}  # duration skipped; district in prose without cue is not protected


def test_month_first_shift_does_not_swap_day_and_month() -> None:
    assert dates.make_shifter("mdy")("01/13/2026", 30) == "02/12/2026"
    assert dates.make_shifter("dmy")("13/01/2026", 30) == "12/02/2026"


def test_edits_band_unlinked_record_via_orphan_cluster() -> None:
    rec = _rec(("Age", "46", "str"))
    sp = ps.privacy_spans(rec, POLICY, "dmy")
    res = Resolution(patient_of={}, cluster_of={}, family_of={}, shift_days={})
    plan = PrivacyPlan(age_band={f"orphan:{rec.key[0]}|{rec.key[1]}": "40-49"}, location_level={},
                       dx_rollup={}, generalized=[])
    eds = build(rec, sp, res, plan)
    assert [(e.kind, e.text) for e in eds] == [("BAND_AGE", "40-49")]


def test_meta_record_summarized_but_channels_skipped() -> None:
    assert ps.summarizable("@meta") and ps.summarizable("row:3")
    assert not ps.summarizable("@filename") and not ps.summarizable("@header")


def test_layer_b_does_not_flag_shifted_datetime_in_prose() -> None:
    txt = "[NAME]\n\n17/06/2026 22:00 - BP 117/82"
    assert [h.check for h in residual_b.scan(txt, "sanitized/notes/x.txt")] == []
    # a real unlabelled phone right next to a date is still caught
    hits = residual_b.scan("17/06/2026 98765 43210", "sanitized/notes/x.txt")
    assert any(h.check == "B2" for h in hits)
