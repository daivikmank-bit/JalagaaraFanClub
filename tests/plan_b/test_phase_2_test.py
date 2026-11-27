"""Synthetic Plan B phase 2 boundary test."""

from kavachmed.contracts import Context, Field, Record, Span
from kavachmed.detect.arbiter import arbitrate
from kavachmed.detect.arbiter import detect
from kavachmed.detect.hidden import detect_hidden
from kavachmed.propagate import build_index, sweep


def _record(file, text):
    return Record(key=(file, "doc"), fields=[Field(0, "doc", text)])


def test_propagation_is_cluster_and_file_scoped():
    seed = _record("synthetic/a.txt", "Velora Neris")
    linked = _record("synthetic/b.txt", "velora neris")
    unrelated = _record("synthetic/c.txt", "velora neris")
    unlinked_seed = _record("synthetic/d.txt", "Zarela Vora")
    same_file = Record(key=("synthetic/d.txt", "row:1"), fields=[Field(0, "doc", "zarela vora")])
    other_file = _record("synthetic/e.txt", "zarela vora")
    records = [seed, linked, unrelated, unlinked_seed, same_file, other_file]
    seeds = {
        r.key: [Span(r.key, 0, 0, len(r.fields[0].text), "PATIENT_NAME", "patient", 0.9, "context")]
        for r in (seed, unlinked_seed)
    }

    class Resolution:
        cluster_of = {seed.key: "one", linked.key: "one", unrelated.key: "two"}
        family_of = {}

    resolution = Resolution()
    index = build_index(records, resolution, seeds)
    for rec in (linked, same_file):
        spans = sweep(rec, resolution, [], index=index)
        assert any(s.src == "propagate" and rec.fields[0].text[s.start:s.end] == rec.fields[0].text for s in spans)
    for rec in (unrelated, other_file):
        assert not any(s.src == "propagate" for s in sweep(rec, resolution, [], index=index))


def test_arbitration_preserves_uncovered_identifier_suffix():
    rec = _record("synthetic/overlap.txt", "abcdefghijkl")
    candidates = [
        Span(rec.key, 0, 0, 7, "PHONE", "patient", 0.9, "ids"),
        Span(rec.key, 0, 5, 12, "MRN", "patient", 0.9, "ids"),
    ]
    spans = arbitrate(candidates, [], Context())
    assert all(any(s.start <= i < s.end for s in spans) for i in range(12))
    assert [(s.start, s.end) for s in spans] == [(0, 7), (7, 12)]


def test_propagation_does_not_join_numbers_across_lines():
    seed = _record("synthetic/seed.txt", "9876543210")
    target = _record("synthetic/target.txt", "9876543210\n42")
    span = Span(seed.key, 0, 0, 10, "PHONE", "patient", 0.9, "ids")
    index = build_index([seed, target], None, {seed.key: [span]})
    spans = sweep(target, None, [], index=index)
    assert any(s.src == "propagate" and target.fields[0].text[s.start:s.end] == "9876543210" for s in spans)


def test_encoded_unicode_name_and_hex_phone_keep_source_offsets():
    encoded_name = "%E0%A4%B0%E0%A4%B5%E0%A4%BF"
    encoded_phone = b"Phone: 9876543210".hex()
    text = f"नाम: {encoded_name}; blob={encoded_phone}; BP 120/80"
    rec = _record("synthetic/encoded.txt", text)
    spans = detect_hidden(rec, Context())
    assert any(text[s.start:s.end] == encoded_name and s.typ == "PATIENT_NAME" for s in spans)
    assert any(text[s.start:s.end] == encoded_phone and s.typ == "PHONE" for s in spans)
    bp_start = text.index("BP 120/80")
    assert not any(s.start < len(text) and s.end > bp_start for s in spans)


def test_name_that_is_also_a_place_is_not_exempted():
    ctx = Context(lexicons={"names": {"velora"}, "places": {"velora"}})
    rec = _record("synthetic/collision.txt", "Velora arrived")
    assert any(rec.fields[0].text[s.start:s.end] == "Velora" for s in detect(rec, ctx))
    district = _record("synthetic/district.txt", "District Velora")
    assert not any("Velora" in district.fields[0].text[s.start:s.end] for s in detect(district, ctx))
