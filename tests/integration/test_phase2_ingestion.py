"""Phase 2 Step 5: End-to-end CLI integration, determinism, accounting, and input immutability."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path


def _input_hashes(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_cli_determinism_accounting_and_no_internal_locs(tmp_path: Path) -> None:
    inp = tmp_path / "inp"
    inp.mkdir()
    (inp / "patients.csv").write_text("id,note\n1,Dolo 1-0-1\n", encoding="utf-8")
    dup_bytes = b'{"_meta":"h1","visits":[{"note":"Dr. Alpha 1-0-1"}]}'
    (inp / "opd_a.json").write_bytes(dup_bytes)
    (inp / "opd_b.json").write_bytes(dup_bytes)
    (inp / "chat.txt").write_text("[01/02/2025, 09:00] Sender: 1-0-1\n", encoding="utf-8")
    (inp / "broken.json").write_bytes(b'{"visits": [')
    (inp / "fake.csv").write_bytes(b"%PDF-1.4\n...")

    before = _input_hashes(inp)
    env = dict(os.environ, RAKSHA_SECRET="practice")

    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    for out in (out1, out2):
        proc = subprocess.run(
            [sys.executable, "main.py", "--input", str(inp), "--output", str(out)],
            env=env,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr

    # Input immutability
    assert _input_hashes(inp) == before

    # Byte-for-byte determinism on sanitized/ and records.jsonl
    assert _tree_bytes(out1 / "sanitized") == _tree_bytes(out2 / "sanitized")
    assert (out1 / "records.jsonl").read_bytes() == (out2 / "records.jsonl").read_bytes()

    # Multiset accounting in report.json
    report = json.loads((out1 / "report.json").read_text(encoding="utf-8"))
    assert Counter(item["input_sha256"] for item in report["files"]) == Counter(before.values())
    assert report["counts"]["files_sanitized"] == 4
    assert report["counts"]["files_quarantined"] == 2

    # Quarantined files leave no artifact in sanitized/
    assert not (out1 / "sanitized" / "broken.json").exists()
    assert not (out1 / "sanitized" / "fake.csv").exists()

    # No internal locators (@header, @meta) in records.jsonl
    locs = [
        json.loads(line)["loc"]
        for line in (out1 / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert all(not loc.startswith("@") for loc in locs)
    assert "/visits/0" in locs
    assert "row:0" in locs
    assert "doc" in locs

