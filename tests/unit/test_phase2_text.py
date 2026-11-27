"""Phase 2 Step 2 (A7): Plain text and WhatsApp field splitting + LocMap."""
from __future__ import annotations

from pathlib import Path

from kavachmed import adapters, inventory
from kavachmed.contracts import Edit


def test_whatsapp_sender_edit_preserves_crlf_and_no_trailing_nl(tmp_path: Path) -> None:
    raw_text = "[01/02/2025, 09:45] Example Sender: sample\r\ncontinuation line"
    (tmp_path / "chat.txt").write_bytes(raw_text.encode("utf-8"))

    entry = inventory.scan(tmp_path)[0]
    parsed = adapters.parse(entry, {})
    assert len(parsed.records) == 1
    rec = parsed.records[0]
    assert rec.key == ("chat.txt", "doc")

    # Zero edits -> exact round-trip
    assert parsed.projector is not None
    zero_bytes, _ = parsed.projector([], {})
    assert zero_bytes.decode("utf-8") == raw_text

    sender_field = next(f for f in rec.fields if f.path == "lines/0/sender")
    assert sender_field.text == "Example Sender"

    edit = Edit(rec.key, sender_field.idx, 0, len(sender_field.text), "REPLACE", "[SENDER]")
    out_bytes, loc_map = parsed.projector([edit], {})
    out_str = out_bytes.decode("utf-8")
    assert out_str == "[01/02/2025, 09:45] [SENDER]: sample\r\ncontinuation line"

    start = out_str.index("[SENDER]")
    hits = loc_map.lookup(start, start + len("[SENDER]"))
    assert {(h.key, h.field_idx) for h in hits} == {(rec.key, sender_field.idx)}


def test_devanagari_and_empty_line_offsets(tmp_path: Path) -> None:
    raw_text = "नाम: परीक्षण\n\n[02/02/2025, 10:00] कमला देवी: नमस्ते\n"
    (tmp_path / "hi_chat.txt").write_bytes(raw_text.encode("utf-8"))

    entry = inventory.scan(tmp_path)[0]
    parsed = adapters.parse(entry, {})
    rec = parsed.records[0]

    sender_field = next(f for f in rec.fields if f.path == "lines/2/sender")
    assert sender_field.text == "कमला देवी"

    edit = Edit(rec.key, sender_field.idx, 0, len(sender_field.text), "REPLACE", "PSN-12345678")
    assert parsed.projector is not None
    out_bytes, loc_map = parsed.projector([edit], {})
    out_str = out_bytes.decode("utf-8")
    assert out_str == "नाम: परीक्षण\n\n[02/02/2025, 10:00] PSN-12345678: नमस्ते\n"

    start = out_str.index("PSN-12345678")
    hits = loc_map.lookup(start, start + len("PSN-12345678"))
    assert {(h.key, h.field_idx) for h in hits} == {(rec.key, sender_field.idx)}

