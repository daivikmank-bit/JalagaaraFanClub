"""Unit tests for tools/self_audit.py (Phase 8 §0a)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_self_audit_canary_hit_redacts_matched_string(tmp_path: Path) -> None:
    kit = tmp_path / "kit"
    (kit / "dev" / "truth").mkdir(parents=True)
    secret_canary = "CANARY-SYNTH-998877"
    (kit / "dev" / "truth" / "truth.json").write_text(
        json.dumps({"canaries": [{"surface": secret_canary}]}),
        encoding="utf-8",
    )

    cand = tmp_path / "candidate"
    (cand / "kavachmed").mkdir(parents=True)
    (cand / "kavachmed" / "leak.py").write_text(f'X = "{secret_canary}"\n', encoding="utf-8")

    res = subprocess.run(
        [sys.executable, "tools/self_audit.py", "--root", str(cand), "--kit", str(kit)],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 1
    assert "dev_canary_leak" in res.stderr
    assert secret_canary not in res.stdout
    assert secret_canary not in res.stderr


def test_self_audit_clean_fixture_passes(tmp_path: Path) -> None:
    kit = tmp_path / "kit"
    (kit / "dev" / "truth").mkdir(parents=True)
    (kit / "dev" / "truth" / "truth.json").write_text(
        json.dumps({"canaries": [{"surface": "CANARY-SYNTH-998877"}]}),
        encoding="utf-8",
    )

    cand = tmp_path / "candidate"
    (cand / "kavachmed").mkdir(parents=True)
    (cand / "kavachmed" / "clean.py").write_text('X = "ok"\n', encoding="utf-8")

    res = subprocess.run(
        [sys.executable, "tools/self_audit.py", "--root", str(cand), "--kit", str(kit)],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "SELF AUDIT OK" in res.stdout

