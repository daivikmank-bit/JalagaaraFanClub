"""Phase 2 Step 3 (A5): CSV/TSV logical rows, header channel, repair & LocMap."""
from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest

from kavachmed import adapters, inventory
from kavachmed.contracts import Edit
from kavachmed.security import StageError


def test_csv_multiline_crlf_leading_zero_and_quoted_quotes(tmp_path: Path) -> None:
    source = 'code,note\r\n01,"first ""q"" line\r\nsecond, line"\r\n'
    (tmp_path / "notes.csv").write_bytes(source.encode("utf-8"))

    entry = inventory.scan(tmp_path)[0]
    parsed = adapters.parse(entry, {})
    assert parsed.newline == "\r\n"
    assert parsed.projector is not None

    # Zero-edit round-trip preserves CRLF, trailing newline, leading zero, and quoted quotes
    zero_bytes, _ = parsed.projector([], {})
    assert zero_bytes.decode("utf-8") == source

    # Header channel is @header, body row is row:0
    hdr = next(r for r in parsed.records if r.key[1] == "@header")
    row0 = next(r for r in parsed.records if r.key[1] == "row:0")
    assert [f.text for f in hdr.fields] == ["code", "note"]
    assert row0.fields[0].text == "01"
    assert row0.fields[1].text == 'first "q" line\r\nsecond, line'

    # Edit both header and body
    edits = [
        Edit(hdr.key, 1, 0, 4, "REPLACE", "remark"),
        Edit(row0.key, 1, 0, 5, "REPLACE", "entry"),
    ]
    out_bytes, loc_map = parsed.projector(edits, {})
    out_str = out_bytes.decode("utf-8")
    assert out_str.endswith("\r\n")
    reparsed = list(csv.reader(io.StringIO(out_str, newline=""), strict=True))
    assert reparsed == [["code", "remark"], ["01", 'entry "q" line\r\nsecond, line']]

    pos = out_str.index("entry")
    hits = loc_map.lookup(pos, pos + 5)
    assert {(h.key, h.field_idx) for h in hits} == {(row0.key, 1)}


def test_tsv_header_only_and_duplicate_headers(tmp_path: Path) -> None:
    source = "val\tval\n"
    (tmp_path / "hdr.tsv").write_bytes(source.encode("utf-8"))

    entry = inventory.scan(tmp_path)[0]
    parsed = adapters.parse(entry, {})
    assert parsed.fmt == "tsv"
    assert len(parsed.records) == 1
    assert parsed.records[0].key[1] == "@header"

    hdr = parsed.records[0]
    edit = Edit(hdr.key, 1, 0, 3, "REPLACE", "second_val")
    assert parsed.projector is not None
    out_bytes, _ = parsed.projector([edit], {})
    assert out_bytes.decode("utf-8") == "val\tsecond_val\n"


def test_csv_unambiguous_overwide_repair_and_ambiguous_quarantine(tmp_path: Path) -> None:
    # Unambiguous repair: code(int), note(text), date(date)
    good_and_broken = (
        "code,note,date\n"
        "01,fever,2025-02-01\n"
        "02,cough,2025-02-02\n"
        "03,follow,up,2025-02-03\n"
    )
    (tmp_path / "repair.csv").write_bytes(good_and_broken.encode("utf-8"))
    entry = inventory.scan(tmp_path)[0]
    parsed = adapters.parse(entry, {})
    row2 = next(r for r in parsed.records if r.key[1] == "row:2")
    assert [f.text for f in row2.fields] == ["03", "follow,up", "2025-02-03"]

    # Ambiguous overwide row: two adjacent text columns
    ambiguous = (
        "code,note,remark\n"
        "01,fever,mild\n"
        "02,cough,dry\n"
        "03,a,b,c\n"
    )
    (tmp_path / "ambig.csv").write_bytes(ambiguous.encode("utf-8"))
    ambig_entry = next(e for e in inventory.scan(tmp_path) if e.rel_path == "ambig.csv")
    with pytest.raises(StageError) as exc:
        adapters.parse(ambig_entry, {})
    assert exc.value.code == "E_CSV"


def test_csv_unterminated_quote_quarantines(tmp_path: Path) -> None:
    (tmp_path / "bad.csv").write_bytes(b'a,b\n1,"unterminated\n2,3\n')
    entry = inventory.scan(tmp_path)[0]
    with pytest.raises(StageError) as exc:
        adapters.parse(entry, {})
    assert exc.value.code == "E_CSV"

