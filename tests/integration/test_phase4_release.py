"""Phase 4 (A10+A11): Full release integration test with mixed outcomes and repeatability."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from tests.unit.test_phase4_emit import assert_release_tree


def test_release_determinism_and_tree_reconciliation(tmp_path: Path) -> None:
    inp = tmp_path / "inp"
    inp.mkdir()
    (inp / "a.csv").write_bytes(b"code,val\n01,1-0-1\n")
    (inp / "b.json").write_bytes(b'{"visits":[{"note":"1-0-1"}]}')
    (inp / "bad.json").write_bytes(b'{"visits":[')
    (inp / "img.csv").write_bytes(b"%PDF-1.4\n...")

    digests = [hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(inp.rglob("*")) if p.is_file()]
    env = dict(os.environ, RAKSHA_SECRET="practice")

    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    for out in (out1, out2):
        proc = subprocess.run(
            [sys.executable, "main.py", "--input", str(inp), "--output", str(out)],
            env=env, capture_output=True, text=True,
        )
        assert proc.returncode == 0, proc.stderr

    assert (out1 / "records.jsonl").read_bytes() == (out2 / "records.jsonl").read_bytes()
    rep1 = json.loads((out1 / "report.json").read_text(encoding="utf-8"))
    assert_release_tree(out1, rep1, digests)
    assert rep1["counts"]["files_sanitized"] == 2
    assert rep1["counts"]["files_quarantined"] == 2


def test_reused_output_dir_removes_newly_quarantined_file(tmp_path: Path) -> None:
    inp = tmp_path / "inp"
    inp.mkdir()
    note = inp / "note.txt"
    note.write_bytes(b"Patient Alpha\n")
    out = tmp_path / "out"
    (out / "user_extra.txt").parent.mkdir(parents=True, exist_ok=True)
    (out / "user_extra.txt").write_text("keep me", encoding="utf-8")

    env = dict(os.environ, RAKSHA_SECRET="practice")
    p1 = subprocess.run([sys.executable, "main.py", "--input", str(inp), "--output", str(out)], env=env, capture_output=True)
    assert p1.returncode == 0
    assert (out / "sanitized" / "note.txt").is_file()

    # Replace input with empty bytes -> now quarantined; stale sanitized/note.txt must be removed
    note.write_bytes(b"")
    p2 = subprocess.run([sys.executable, "main.py", "--input", str(inp), "--output", str(out)], env=env, capture_output=True)
    assert p2.returncode == 0
    assert not (out / "sanitized" / "note.txt").exists()
    assert (out / "user_extra.txt").read_text(encoding="utf-8") == "keep me"
    rep2 = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert rep2["files"][0]["status"] == "quarantined"


def test_dose_only_note_not_treated_as_reference(tmp_path: Path) -> None:
    inp = tmp_path / "inp"
    inp.mkdir()
    (inp / "note.txt").write_bytes(b"1-0-1\n")
    out = tmp_path / "out"
    env = dict(os.environ, RAKSHA_SECRET="practice")
    p = subprocess.run([sys.executable, "main.py", "--input", str(inp), "--output", str(out)], env=env, capture_output=True)
    assert p.returncode == 0
    rows = [json.loads(l) for l in (out / "records.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 1 and rows[0]["loc"] == "doc"


def test_reused_output_preserves_unowned_file_inside_sanitized(tmp_path: Path) -> None:
    inp = tmp_path / "inp"
    inp.mkdir()
    (inp / "note.txt").write_bytes(b"Patient Alpha\n")
    out = tmp_path / "out"
    env = dict(os.environ, RAKSHA_SECRET="practice")

    p1 = subprocess.run([sys.executable, "main.py", "--input", str(inp), "--output", str(out)], env=env, capture_output=True)
    assert p1.returncode == 0

    unowned = out / "sanitized" / "custom_unowned.txt"
    unowned.write_text("keep inside sanitized", encoding="utf-8")

    p2 = subprocess.run([sys.executable, "main.py", "--input", str(inp), "--output", str(out)], env=env, capture_output=True)
    assert p2.returncode == 0
    assert unowned.is_file() and unowned.read_text(encoding="utf-8") == "keep inside sanitized"


def test_unowned_collision_without_manifest_fails_before_write(tmp_path: Path) -> None:
    inp = tmp_path / "inp"
    inp.mkdir()
    (inp / "note.txt").write_bytes(b"Patient Alpha\n")
    (inp / "other.txt").write_bytes(b"Patient Beta\n")
    out = tmp_path / "out"
    (out / "sanitized").mkdir(parents=True)
    colliding = out / "sanitized" / "note.txt"
    colliding.write_bytes(b"UNOWNED_ORIGINAL_BYTES")

    env = dict(os.environ, RAKSHA_SECRET="practice")
    p = subprocess.run([sys.executable, "main.py", "--input", str(inp), "--output", str(out)], env=env, capture_output=True)
    assert p.returncode != 0
    assert colliding.read_bytes() == b"UNOWNED_ORIGINAL_BYTES"
    assert not (out / "sanitized" / "other.txt").exists()


def test_unowned_collision_with_prior_manifest_fails_before_write(tmp_path: Path) -> None:
    inp1 = tmp_path / "inp1"
    inp1.mkdir()
    (inp1 / "alpha_doc.txt").write_bytes(b"Patient Beta\n")
    out = tmp_path / "out"
    env = dict(os.environ, RAKSHA_SECRET="practice")

    p1 = subprocess.run([sys.executable, "main.py", "--input", str(inp1), "--output", str(out)], env=env, capture_output=True)
    assert p1.returncode == 0
    prior_files = [p for p in (out / "sanitized").rglob("*") if p.is_file()]
    assert len(prior_files) == 1
    prior_path = prior_files[0]
    prior_bytes = prior_path.read_bytes()

    # Place an unowned file at a path the next run will allocate (note.txt -> sanitized/note.txt)
    colliding = out / "sanitized" / "note.txt"
    colliding.write_bytes(b"UNOWNED_COLLISION")

    inp2 = tmp_path / "inp2"
    inp2.mkdir()
    (inp2 / "note.txt").write_bytes(b"Patient Alpha\n")
    (inp2 / "alpha_doc.txt").write_bytes(b"Patient Gamma\n")

    p2 = subprocess.run([sys.executable, "main.py", "--input", str(inp2), "--output", str(out)], env=env, capture_output=True)
    assert p2.returncode != 0
    assert colliding.read_bytes() == b"UNOWNED_COLLISION"
    assert prior_path.read_bytes() == prior_bytes

