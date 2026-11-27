"""Phase 2 Step 1 (A4): Deterministic inventory, magic-byte sniff, decode ladder."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from kavachmed import adapters, inventory
from kavachmed.security import StageError


def test_json_with_csv_suffix_and_duplicate_bytes(tmp_path: Path) -> None:
    raw = b'{"visits":[{"note":"sample"}]}'
    for name in ("z.csv", "a.json"):
        (tmp_path / name).write_bytes(raw)
    entries = inventory.scan(tmp_path)
    assert [entry.rel_path for entry in entries] == ["a.json", "z.csv"]
    assert [entry.sha256 for entry in entries] == [hashlib.sha256(raw).hexdigest()] * 2
    assert [adapters.parse(entry, {}).fmt for entry in entries] == ["json", "json"]


def test_binary_signatures_override_text_extensions(tmp_path: Path) -> None:
    (tmp_path / "photo.csv").write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")
    (tmp_path / "scan.txt").write_bytes(b"%PDF-1.4\n%...")
    (tmp_path / "img.json").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")
    (tmp_path / "empty.csv").write_bytes(b"   \n\t ")

    entries = inventory.scan(tmp_path)
    reasons = {e.rel_path: e.quarantine_reason for e in entries}
    assert reasons["photo.csv"] == "unsupported_format"
    assert reasons["scan.txt"] == "unsupported_format"
    assert reasons["img.json"] == "unsupported_format"
    assert reasons["empty.csv"] == "empty_file"


def test_decode_ladder_encodings_and_mixed_vs_undecodable(tmp_path: Path) -> None:
    # UTF-8 BOM
    (tmp_path / "bom_utf8.txt").write_bytes(b"\xef\xbb\xbfhello \xe0\xa4\xa8\n")
    # UTF-16 LE BOM
    (tmp_path / "bom_utf16.tsv").write_bytes("a\tb\n1\t2\n".encode("utf-16"))
    # BOM-less UTF-16 LE (alternating NULs)
    (tmp_path / "nobom_utf16le.txt").write_bytes("hello world\n".encode("utf-16-le"))
    # Ordinary cp1252 accented text
    (tmp_path / "legacy_cp1252.csv").write_bytes("col\ncaf\xe9\n".encode("latin-1"))
    # Undefined cp1252 byte (0x81) -> undecodable
    (tmp_path / "bad_byte.txt").write_bytes(b"abc\x81def")
    # Mixed encoding: valid multi-byte UTF-8 sequence followed by isolated invalid byte 0xff
    (tmp_path / "mixed.txt").write_bytes("नमस्ते ".encode("utf-8") + b"\xffworld")
    # Truncated UTF-16 BOM
    (tmp_path / "trunc_bom.txt").write_bytes(b"\xff\xfe\x41")

    entries = {e.rel_path: e for e in inventory.scan(tmp_path)}
    assert entries["bom_utf8.txt"].quarantine_reason is None
    assert entries["bom_utf16.tsv"].quarantine_reason is None
    assert entries["nobom_utf16le.txt"].quarantine_reason is None
    assert entries["legacy_cp1252.csv"].quarantine_reason is None
    assert entries["legacy_cp1252.csv"].encoding == "cp1252"
    assert entries["bad_byte.txt"].quarantine_reason == "undecodable"
    assert entries["mixed.txt"].quarantine_reason == "mixed_encoding"
    assert entries["trunc_bom.txt"].quarantine_reason == "undecodable"


def test_truncated_json_and_xml_routing(tmp_path: Path) -> None:
    (tmp_path / "trunc.txt").write_bytes(b'{"visits": [{"note": "hi"')
    (tmp_path / "cda.xml").write_bytes(b'<?xml version="1.0"?><ClinicalDocument/>')

    entries = {e.rel_path: e for e in inventory.scan(tmp_path)}
    with pytest.raises(StageError) as exc1:
        adapters.parse(entries["trunc.txt"], {})
    assert exc1.value.code == "E_INVALID_JSON"

    assert adapters.parse(entries["cda.xml"], {}).fmt == "xml"  # A12: XML is sanitised


def test_corrupt_ff_fe_bom_controls_and_unfamiliar_ext_csv(tmp_path: Path) -> None:
    (tmp_path / "corrupt.txt").write_bytes(b"note \xff\xfe \xc3\n")
    (tmp_path / "bom_ctrl.txt").write_bytes(b"\xef\xbb\xbfhello\x01world")
    (tmp_path / "table.dat").write_text("a,b,c\n1,2,3\n4,5,6\n", encoding="utf-8")

    entries = {e.rel_path: e for e in inventory.scan(tmp_path)}
    assert entries["corrupt.txt"].quarantine_reason == "mixed_encoding"
    assert entries["bom_ctrl.txt"].quarantine_reason == "undecodable"
    assert adapters.parse(entries["table.dat"], {}).fmt == "csv"


