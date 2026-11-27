"""Phase 2 Step 4 (A6): Iterative JSON parsing, traversal, projection, base64, and depth cap."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from kavachmed import adapters, inventory
from kavachmed.contracts import Edit
from kavachmed.security import StageError


def test_sibling_projection_and_key_edit(tmp_path: Path) -> None:
    (tmp_path / "sample.json").write_text('{"left":"aa","right":"bb"}', encoding="utf-8")
    parsed = adapters.parse(inventory.scan(tmp_path)[0], {})
    record = next(r for r in parsed.records if r.key[1] == "doc")
    field = next(f for f in record.fields if f.vtype == "str" and f.text == "bb")
    change = Edit(record.key, field.idx, 0, 2, "REPLACE", "edited")
    assert parsed.projector is not None
    payload, locations = parsed.projector([change], {})
    assert json.loads(payload) == {"left": "aa", "right": "edited"}
    start = payload.decode("utf-8").index("edited")
    hits = locations.lookup(start, start + 6)
    assert {(hit.key, hit.field_idx) for hit in hits} == {(record.key, field.idx)}


def test_record_array_envelope_and_numeric_surrogate(tmp_path: Path) -> None:
    src = '{"_meta":"export-1","visits":[{"aadhar_no":123456789012,"note":"ok"}]}'
    (tmp_path / "opd.json").write_text(src, encoding="utf-8")
    parsed = adapters.parse(inventory.scan(tmp_path)[0], {})

    visit0 = next(r for r in parsed.records if r.key[1] == "/visits/0")
    meta_rec = next(r for r in parsed.records if r.key[1] == "@meta")
    assert any(f.text == "export-1" for f in meta_rec.fields)

    num_field = next(f for f in visit0.fields if f.vtype == "int")
    assert num_field.text == "123456789012"
    edit = Edit(visit0.key, num_field.idx, 0, len(num_field.text), "NUM_SURROGATE", "999988887777")
    assert parsed.projector is not None
    out_bytes, _ = parsed.projector([edit], {})
    out_obj = json.loads(out_bytes)
    assert isinstance(out_obj["visits"][0]["aadhar_no"], int)
    assert out_obj["visits"][0]["aadhar_no"] == 999988887777


def test_base64_text_field_and_ordinary_short_string(tmp_path: Path) -> None:
    plain = "Patient: Alpha Beta"
    b64_val = base64.b64encode(plain.encode("utf-8")).decode("ascii")
    src = json.dumps({"content_base64": b64_val, "code": "ABCD"})
    (tmp_path / "consent.json").write_text(src, encoding="utf-8")

    parsed = adapters.parse(inventory.scan(tmp_path)[0], {})
    rec = next(r for r in parsed.records if r.key[1] == "doc")
    b64_field = next(f for f in rec.fields if f.vtype == "b64")
    assert b64_field.text == plain
    # Ordinary short string "ABCD" must remain "str", not "b64"
    code_field = next(f for f in rec.fields if f.text == "ABCD")
    assert code_field.vtype == "str"

    idx = plain.index("Alpha Beta")
    edit = Edit(rec.key, b64_field.idx, idx, idx + len("Alpha Beta"), "REPLACE", "[NAME]")
    assert parsed.projector is not None
    out_bytes, loc_map = parsed.projector([edit], {})
    out_obj = json.loads(out_bytes)
    decoded = base64.b64decode(out_obj["content_base64"]).decode("utf-8")
    assert decoded == "Patient: [NAME]"

    b64_out = out_obj["content_base64"]
    pos = out_bytes.decode("utf-8").index(b64_out)
    hits = loc_map.lookup(pos, pos + len(b64_out))
    assert {(h.key, h.field_idx) for h in hits} == {(rec.key, b64_field.idx)}


def test_deep_json_1501_4000_and_4001_cap(tmp_path: Path) -> None:
    for depth in (1501, 4000):
        raw = ("[" * depth + "0" + "]" * depth).encode("utf-8")
        p = tmp_path / f"deep_{depth}.json"
        p.write_bytes(raw)
        entry = next(e for e in inventory.scan(tmp_path) if e.rel_path == f"deep_{depth}.json")
        parsed = adapters.parse(entry, {"runtime": {"max_json_depth": 4000}})
        assert parsed.projector is not None
        out_bytes, _ = parsed.projector([], {})
        assert out_bytes == raw

    # Depth 4001 -> E_DEPTH
    (tmp_path / "too_deep.json").write_bytes(("[" * 4001 + "0" + "]" * 4001).encode("utf-8"))
    deep_entry = next(e for e in inventory.scan(tmp_path) if e.rel_path == "too_deep.json")
    with pytest.raises(StageError) as exc_depth:
        adapters.parse(deep_entry, {"runtime": {"max_json_depth": 4000}})
    assert exc_depth.value.code == "E_DEPTH"

    # Truncated below cap -> E_INVALID_JSON
    (tmp_path / "trunc.json").write_bytes(("[" * 100 + "0" + "]" * 99).encode("utf-8"))
    trunc_entry = next(e for e in inventory.scan(tmp_path) if e.rel_path == "trunc.json")
    with pytest.raises(StageError) as exc_trunc:
        adapters.parse(trunc_entry, {"runtime": {"max_json_depth": 4000}})
    assert exc_trunc.value.code == "E_INVALID_JSON"


def _parse_sample(tmp_path: Path, source: str):
    (tmp_path / "example.json").write_text(source, encoding="utf-8")
    return adapters.parse(inventory.scan(tmp_path)[0], {})


def test_root_record_array(tmp_path: Path) -> None:
    parsed = _parse_sample(tmp_path, '[{"v":1},{"v":2}]')
    public = [r.key[1] for r in parsed.records if not r.key[1].startswith("@")]
    assert public == ["/0", "/1"]


def test_empty_record_keeps_its_index(tmp_path: Path) -> None:
    parsed = _parse_sample(tmp_path, '{"visits":[{},{"v":1}]}')
    public = [r.key[1] for r in parsed.records if not r.key[1].startswith("@")]
    assert public == ["/visits/0", "/visits/1"]


def test_key_collision_reserves_later_unchanged_sibling(tmp_path: Path) -> None:
    parsed = _parse_sample(tmp_path, '{"a":1,"b":2}')
    rec = parsed.records[0]
    key_a = next(f for f in rec.fields if f.vtype == "key" and f.text == "a")
    edit = Edit(rec.key, key_a.idx, 0, 1, "REPLACE", "b")
    assert parsed.projector is not None
    payload, _ = parsed.projector([edit], {})

    seen: list[str] = []

    def no_dup_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        seen.extend(k for k, _ in pairs)
        assert len(seen) == len(set(seen)), f"duplicate keys emitted: {seen}"
        return dict(pairs)

    out = json.loads(payload.decode("utf-8"), object_pairs_hook=no_dup_hook)
    assert out["b"] == 2
    assert set(out.values()) == {1, 2}


def test_invalid_numeric_replacement_raises_stage_error(tmp_path: Path) -> None:
    parsed = _parse_sample(tmp_path, '{"id":123}')
    rec = parsed.records[0]
    num_f = next(f for f in rec.fields if f.vtype == "int")
    bad_edit = Edit(rec.key, num_f.idx, 0, 3, "REPLACE", "[ID]")
    assert parsed.projector is not None
    with pytest.raises(StageError):
        parsed.projector([bad_edit], {})


