"""Unit tests for tools/experiments/benchmark.py qualification guards (Phase 8 §5)."""
from __future__ import annotations

from pathlib import Path

import pytest

from tools.experiments import benchmark


def test_rss_and_growth_gates() -> None:
    assert benchmark.rss_gate([]) == "unmeasured"
    assert benchmark.rss_gate([0]) == "unmeasured"
    assert benchmark.rss_gate([50.0, None, 100.0]) == "unmeasured"
    assert benchmark.rss_gate([50.0, 160.0, 420.0]) == "fail"
    assert benchmark.rss_gate([30.0, 63.0, 186.0]) == "pass"

    assert benchmark.growth_gate([1, 5], 0.9) == "unmeasured"
    assert benchmark.growth_gate([1, 5, 20], 1.20) == "fail"
    assert benchmark.growth_gate([1, 5, 20], 1.04) == "pass"


def test_qualifies_rejects_missing_validation_rss_repeats_or_determinism() -> None:
    targets_ok = {
        "startup_le_0.7s": {"value": 0.1, "status": "pass"},
        "release_1x_le_12s": {"value": 1.0, "status": "pass"},
        "peak_rss_le_400mib": {"value": 186.0, "status": "pass"},
        "growth_exp_le_1.15": {"value": 1.04, "status": "pass"},
    }
    base_scales = [
        {"scale": 1, "observations_s": [1.0, 1.0, 1.0], "deterministic": True, "validated": True, "rss_complete": True},
        {"scale": 5, "observations_s": [5.0, 5.0, 5.0], "deterministic": True, "validated": True, "rss_complete": True},
        {"scale": 20, "observations_s": [20.0, 20.0, 20.0], "deterministic": True, "validated": True, "rss_complete": True},
    ]
    assert benchmark.qualifies(base_scales, targets_ok, 3) is True
    assert benchmark.qualifies(base_scales, targets_ok, 1) is False

    # Failed validator
    bad_val = [dict(s) for s in base_scales]
    bad_val[2]["validated"] = False
    assert benchmark.qualifies(bad_val, targets_ok, 3) is False

    # Missing RSS sample
    bad_rss = [dict(s) for s in base_scales]
    bad_rss[1]["rss_complete"] = False
    assert benchmark.qualifies(bad_rss, targets_ok, 3) is False

    # Same-key mismatch
    bad_det = [dict(s) for s in base_scales]
    bad_det[0]["deterministic"] = False
    assert benchmark.qualifies(bad_det, targets_ok, 3) is False


def test_validate_release_fails_when_missing_or_failing(tmp_path: Path) -> None:
    inp = tmp_path / "in"
    out = tmp_path / "out"
    inp.mkdir()
    out.mkdir()
    with pytest.raises(RuntimeError, match="validator missing"):
        benchmark._validate_release(tmp_path, inp, out)

    bad_val = tmp_path / "validate_output.py"
    bad_val.write_text("import sys; sys.exit(1)\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="validate_output failed"):
        benchmark._validate_release(tmp_path, inp, out)


def test_timed_startup_and_kdf_fail_on_missing_entrypoint(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="startup exited"):
        benchmark.timed_startup(tmp_path, "practice")
    with pytest.raises(RuntimeError, match="kdf exited"):
        benchmark.timed_kdf(tmp_path, "practice")


def test_prepare_scaled_inputs_rebuilds_when_source_changes(tmp_path: Path) -> None:
    inp = tmp_path / "input"
    out = tmp_path / "out"
    inp.mkdir()
    (inp / "a.txt").write_text("v1", encoding="utf-8")

    scales1 = benchmark._prepare_scaled_inputs(inp, 5, out)
    scaled_dir = dict(scales1)[5]
    assert (scaled_dir / "copy_000" / "a.txt").read_text(encoding="utf-8") == "v1"

    # Mutate source input and reuse the same out directory
    (inp / "a.txt").write_text("v2", encoding="utf-8")
    scales2 = benchmark._prepare_scaled_inputs(inp, 5, out)
    scaled_dir2 = dict(scales2)[5]
    assert (scaled_dir2 / "copy_000" / "a.txt").read_text(encoding="utf-8") == "v2"
    assert (scaled_dir2 / "copy_004" / "a.txt").read_text(encoding="utf-8") == "v2"

