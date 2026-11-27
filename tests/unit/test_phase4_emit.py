"""Phase 4 (A11): Reference-document byte-copy and manifest reconciliation tests."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from kavachmed import pipeline
from kavachmed.security import load_keys


def assert_release_tree(output_dir: Path, report: dict, input_digests: list[str]) -> None:
    files = report["files"]
    assert Counter(row["input_sha256"] for row in files) == Counter(input_digests)
    declared = [row["output"] for row in files if row["status"] == "sanitized"]
    actual = {
        p.relative_to(output_dir).as_posix()
        for p in (output_dir / "sanitized").rglob("*")
        if p.is_file()
    }
    assert len(declared) == len(set(declared))
    assert set(declared) == actual
    assert all(row["output"] is None for row in files if row["status"] == "quarantined")


def test_reference_byte_copy_ignores_folder_name(tmp_path: Path, monkeypatch) -> None:
    inp = tmp_path / "inp"
    (inp / "clean").mkdir(parents=True)
    (inp / "other").mkdir(parents=True)

    # Genuine reference with BOM + CRLF moved under other/
    ref_bytes = b"\xef\xbb\xbfdrug,dose\r\n1-0-1,1-0-1\r\n"
    (inp / "other" / "formulary.csv").write_bytes(ref_bytes)

    # Patient note placed inside clean/ directory -> must still be sanitized, NOT byte-copied
    (inp / "clean" / "note.txt").write_bytes(b"Patient Alpha Beta\n")

    # Monkeypatch detect.detect so reference has 0 detections and patient note has 1
    from kavachmed import detect
    from kavachmed.contracts import Span

    orig_detect = detect.detect

    def fake_detect(rec, ctx, sh):
        if rec.key[0] == "other/formulary.csv":
            return []
        return orig_detect(rec, ctx, sh)

    monkeypatch.setattr(detect, "detect", fake_detect)

    out = tmp_path / "out"
    keys = load_keys({"RAKSHA_SECRET": "practice"}, pbkdf2_iterations=1000)
    assert pipeline.run(inp, out, keys, {"runtime": {"reference_doc_byte_copy": True}}) == 0

    # Reference byte-copied verbatim (BOM + CRLF preserved)
    assert (out / "sanitized" / "other" / "formulary.csv").read_bytes() == ref_bytes
    # Patient note inside clean/ was sanitized, not copied raw
    assert b"Alpha" not in (out / "sanitized" / "clean" / "note.txt").read_bytes()

    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["counts"]["identifiers_detected"] == sum(report["entities"].values())
    # Reference emits 0 public records, patient note emits 1
    assert report["counts"]["records_processed"] == 1

