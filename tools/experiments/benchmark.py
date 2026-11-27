"""A15 — Developer-only benchmark harness.

Measures wall-time and peak RSS across E10 (scale), E13 (stage timing),
and E21 (KDF) for the KavachMed pipeline.

Usage (from Jalagaara/):
    python tools/experiments/benchmark.py \\
        --input dev/input --scale 1 --repeat 3 --out ../kavachmed-bench-out

The harness NEVER persists child stdout/stderr that could contain data.
Results are written as aggregate JSON outside the submission tree.

Targets (Linux, Python 3.12, 2 CPU / 4 GB):
    startup     <= 0.70 s
    1x release  <= 12.0 s
    peak RSS    <= 400 MiB
    growth exp  <= 1.15
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from math import isnan, log
from pathlib import Path
from statistics import mean, median
from time import perf_counter


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _env(test_secret: str) -> dict[str, str]:
    env = dict(os.environ)
    env["RAKSHA_SECRET"] = test_secret
    # Suppress colour codes to keep timing output clean
    env.setdefault("NO_COLOR", "1")
    return env


def _fresh_outdir(parent: Path, tag: str) -> Path:
    p = parent / tag
    p.mkdir(parents=True, exist_ok=True)
    return p


def rss_gate(observations: list[float | None]) -> str:
    """Return 'unmeasured', 'pass', or 'fail' for MiB RSS observations."""
    if not observations or any(v is None or v <= 0 for v in observations):
        return "unmeasured"
    return "pass" if max(v for v in observations if v is not None) <= 400.0 else "fail"


def growth_gate(scales: list[int], exp: float | None) -> str:
    """Require >= 3 scales and finite exponent <= 1.15."""
    if len(scales) < 3 or exp is None or isnan(exp):
        return "unmeasured"
    return "pass" if exp <= 1.15 else "fail"


def _tree_digest_input(inp: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(inp.rglob("*")):
        if f.is_file():
            h.update(f.relative_to(inp).as_posix().encode("utf-8") + b"\0" + f.read_bytes())
    return h.hexdigest()


def _release_digest(out: Path) -> str:
    h = hashlib.sha256()
    for name in ("sanitized", "records.jsonl"):
        p = out / name
        items = sorted(p.rglob("*")) if p.is_dir() else ([p] if p.exists() else [])
        for f in items:
            if f.is_file():
                h.update(f.relative_to(out).as_posix().encode("utf-8") + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


def _validate_release(
    repo: Path,
    input_dir: Path,
    output_dir: Path,
    validator_path: Path | None = None,
) -> None:
    validator = validator_path if validator_path is not None else (repo / "validate_output.py")
    if not validator.is_file():
        raise RuntimeError(f"validator missing: {validator}")
    res = subprocess.run(
        [sys.executable, str(validator), "--input", str(input_dir), "--output", str(output_dir)],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if res.returncode != 0 or "CONTRACT OK" not in res.stdout:
        raise RuntimeError(f"validate_output failed (exit {res.returncode})")


def qualifies(scales: list[dict], targets: dict[str, dict], expected_repeats: int) -> bool:
    """Require full 1x/5x/20x coverage, >=3 repeats, passing targets, determinism, validation, and complete RSS."""
    required = {
        "startup_le_0.7s",
        "release_1x_le_12s",
        "peak_rss_le_400mib",
        "growth_exp_le_1.15",
    }
    if set(targets) != required or expected_repeats < 3:
        return False
    if sorted(run["scale"] for run in scales) != [1, 5, 20]:
        return False
    return all(targets[name]["status"] == "pass" for name in required) and all(
        len(run["observations_s"]) == expected_repeats
        and run.get("deterministic")
        and run.get("validated")
        and run.get("rss_complete")
        for run in scales
    )


def _git_revision(repo: Path) -> str:
    res = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return res.stdout.strip() if res.returncode == 0 and res.stdout.strip() else "unknown"


def timed_release(
    repo: Path,
    input_dir: Path,
    output_dir: Path,
    test_secret: str,
    timeout: int = 900,
) -> tuple[float, float | None]:
    """Run one release and return (elapsed_wall_seconds, peak_rss_mib).

    peak_rss_mib is read from report.json (None if missing/unmeasured).
    """
    env = _env(test_secret)
    start = perf_counter()
    result = subprocess.run(
        [sys.executable, "main.py", "--input", str(input_dir), "--output", str(output_dir)],
        cwd=repo,
        env=env,
        capture_output=True,
        timeout=timeout,
    )
    elapsed = perf_counter() - start
    if result.returncode:
        raise RuntimeError(f"release exited {result.returncode}")
    rss: float | None = None
    report_path = output_dir / "report.json"
    if report_path.is_file():
        try:
            rpt = json.loads(report_path.read_text(encoding="utf-8"))
            val = float(rpt.get("resources", {}).get("peak_rss_mb", 0))
            rss = val if val > 0 else None
        except Exception:  # noqa: BLE001
            rss = None
    return elapsed, rss


def timed_startup(repo: Path, test_secret: str) -> float:
    """Measure interpreter + CLI startup via --help (no input reads)."""
    env = _env(test_secret)
    start = perf_counter()
    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        cwd=repo,
        env=env,
        capture_output=True,
        timeout=30,
    )
    elapsed = perf_counter() - start
    if result.returncode != 0:
        raise RuntimeError(f"startup exited {result.returncode}")
    return elapsed


def timed_kdf(repo: Path, test_secret: str) -> float:
    """E21: measure load_keys in a fresh child process."""
    env = _env(test_secret)
    code = (
        "from kavachmed.security import load_keys; "
        "import os, time; "
        "t=time.perf_counter(); "
        "load_keys(os.environ); "
        "print(f'{time.perf_counter()-t:.6f}')"
    )
    start = perf_counter()
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo,
        env=env,
        capture_output=True,
        timeout=30,
    )
    _ = perf_counter() - start  # wall includes process spawn
    if result.returncode != 0:
        raise RuntimeError(f"kdf exited {result.returncode}")
    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError("kdf returned non-numeric output") from exc


def growth_exponent(sizes: list[int], median_seconds: list[float]) -> float:
    """Fit log-log slope (least-squares) to estimate scaling exponent."""
    if len(sizes) < 2:
        return float("nan")
    x = [log(s) for s in sizes]
    y = [log(t) for t in median_seconds]
    mx, my = mean(x), mean(y)
    denom = sum((a - mx) ** 2 for a in x)
    if denom == 0:
        return float("nan")
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / denom


def _get_stage_seconds(output_dir: Path) -> dict[str, float]:
    report_path = output_dir / "report.json"
    if report_path.is_file():
        try:
            rpt = json.loads(report_path.read_text(encoding="utf-8"))
            return rpt.get("timing", {}).get("stages", {})
        except Exception:  # noqa: BLE001
            pass
    return {}


# ---------------------------------------------------------------------------
# main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    repo: Path,
    input_dir: Path,
    scale: int,
    repeat: int,
    out_dir: Path,
    test_secret: str,
    validator_path: Path | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    revision = _git_revision(repo)
    input_digest = _tree_digest_input(input_dir)

    # --- E21: KDF timing (fresh process) ---
    kdf_times = [timed_kdf(repo, test_secret) for _ in range(3)]
    kdf_median = median(kdf_times)

    # --- startup ---
    startup_times = [timed_startup(repo, test_secret) for _ in range(3)]
    startup_median = median(startup_times)

    # --- E10/E13: scale 1× (always measured) ---
    input_copies = _prepare_scaled_inputs(input_dir, scale, out_dir, input_digest=input_digest)

    scale_results: list[dict] = []
    for sz, idir in input_copies:
        obs_elapsed: list[float] = []
        obs_rss: list[float | None] = []
        release_digests: list[str] = []
        stages: dict[str, float] = {}
        validated = True
        # warmup
        with tempfile.TemporaryDirectory(prefix="kav_warm_", dir=out_dir) as wdir:
            timed_release(repo, idir, Path(wdir) / "out", test_secret)
        # measured runs
        for i in range(repeat):
            with tempfile.TemporaryDirectory(prefix=f"kav_run{i}_", dir=out_dir) as rdir:
                odir = Path(rdir) / "out"
                elapsed, rss = timed_release(repo, idir, odir, test_secret)
                # Validate output OUTSIDE timed interval
                try:
                    _validate_release(repo, idir, odir, validator_path=validator_path)
                except RuntimeError:
                    validated = False
                stages = _get_stage_seconds(odir)
                obs_elapsed.append(elapsed)
                obs_rss.append(rss)
                release_digests.append(_release_digest(odir))

        # Check same-secret determinism
        deterministic = len(release_digests) == repeat and len(set(release_digests)) <= 1
        rss_complete = len(obs_rss) == repeat and all(v is not None and v > 0 for v in obs_rss)
        rss_values = [v for v in obs_rss if v is not None and v > 0]

        scale_results.append({
            "scale": sz,
            "input_dir": str(idir),
            "observations_s": [round(x, 3) for x in obs_elapsed],
            "median_s": round(median(obs_elapsed), 3),
            "peak_rss_mib": round(max(rss_values), 1) if rss_complete and rss_values else None,
            "rss_gate": rss_gate(obs_rss),
            "rss_complete": rss_complete,
            "deterministic": deterministic,
            "validated": validated,
            "stage_seconds": stages,
        })

    sizes = [r["scale"] for r in scale_results]
    medians = [r["median_s"] for r in scale_results]
    exp = growth_exponent(sizes, medians) if len(sizes) > 1 else None

    rss_all: list[float | None] = [
        r["peak_rss_mib"] if r["rss_complete"] else None for r in scale_results
    ]
    rss_status = rss_gate(rss_all)
    growth_status = growth_gate(sizes, exp)

    targets = {
        "startup_le_0.7s": {"value": round(startup_median, 3), "status": "pass" if startup_median <= 0.7 else "fail"},
        "release_1x_le_12s": {
            "value": scale_results[0]["median_s"] if scale_results else None,
            "status": "pass" if scale_results and scale_results[0]["median_s"] <= 12.0 else "fail",
        },
        "peak_rss_le_400mib": {
            "value": max(v for v in rss_all if v is not None) if rss_status != "unmeasured" else None,
            "status": rss_status,
        },
        "growth_exp_le_1.15": {"value": round(exp, 3) if exp is not None and not isnan(exp) else None, "status": growth_status},
    }

    all_met = qualifies(scale_results, targets, repeat)

    report = {
        "environment": {
            "python": sys.version,
            "platform": sys.platform,
            "repeat": repeat,
            "candidate_revision": revision,
            "input_digest": input_digest,
        },
        "e21_kdf": {
            "observations_s": [round(x, 6) for x in kdf_times],
            "median_s": round(kdf_median, 6),
        },
        "startup": {
            "observations_s": [round(x, 3) for x in startup_times],
            "median_s": round(startup_median, 3),
        },
        "scale_runs": scale_results,
        "growth_exponent": round(exp, 3) if exp is not None and not isnan(exp) else None,
        "targets": targets,
        "all_targets_met": all_met,
    }

    result_path = out_dir / "benchmark_results.json"
    result_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _prepare_scaled_inputs(
    input_dir: Path,
    max_scale: int,
    out_dir: Path,
    input_digest: str | None = None,
) -> list[tuple[int, Path]]:
    """Return list of (scale_factor, input_path) for scales 1 and optionally 5×, 20×."""
    scales = [1]
    if max_scale >= 5:
        scales.append(5)
    if max_scale >= 20:
        scales.append(20)

    result = [(1, input_dir)]
    files = sorted(input_dir.rglob("*"))
    if len(scales) == 1:
        return result

    src_digest = input_digest or _tree_digest_input(input_dir)
    for sc in scales[1:]:
        scaled_dir = out_dir / f"scale_{sc}x_input"
        digest_marker = scaled_dir / ".source_digest"
        if scaled_dir.exists() and digest_marker.is_file() and digest_marker.read_text(encoding="utf-8").strip() == src_digest:
            result.append((sc, scaled_dir))
            continue
        if scaled_dir.exists():
            shutil.rmtree(scaled_dir)
        scaled_dir.mkdir(parents=True)
        # Copy input sc times with unique subdirectory prefixes so duplicates are addressable
        for copy_idx in range(sc):
            prefix = scaled_dir / f"copy_{copy_idx:03d}"
            prefix.mkdir()
            for src in files:
                if src.is_file():
                    rel = src.relative_to(input_dir)
                    dst = prefix / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(src.read_bytes())
        digest_marker.write_text(src_digest, encoding="utf-8")
        result.append((sc, scaled_dir))
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="A15 performance benchmark harness")
    ap.add_argument("--input", required=True, help="Input directory (dev/input)")
    ap.add_argument("--scale", type=int, default=1, help="Max scale factor (1, 5, 20)")
    ap.add_argument("--repeat", type=int, default=3, help="Measured repetitions per scale")
    ap.add_argument("--out", default="tools/experiments/results", help="Output dir for results")
    ap.add_argument("--secret", default="practice", help="Test RAKSHA_SECRET value")
    ap.add_argument("--validator", default=None, help="Optional path to external validate_output.py")
    args = ap.parse_args()

    repo = Path(".").resolve()
    validator_path = Path(args.validator).resolve() if args.validator else None
    report = run_benchmark(
        repo=repo,
        input_dir=Path(args.input).resolve(),
        scale=args.scale,
        repeat=args.repeat,
        out_dir=Path(args.out).resolve(),
        test_secret=args.secret,
        validator_path=validator_path,
    )

    print(f"\n=== A15 Benchmark Results ===")
    print(f"Revision:         {report['environment'].get('candidate_revision', 'unknown')}")
    print(f"Startup median:   {report['startup']['median_s']:.3f} s")
    print(f"E21 KDF median:   {report['e21_kdf']['median_s']:.6f} s")
    for sr in report["scale_runs"]:
        rss_str = f"{sr['peak_rss_mib']:.1f} MiB" if sr.get("peak_rss_mib") is not None else "unmeasured"
        det_str = " deterministic" if sr.get("deterministic") else " NON-DETERMINISTIC"
        print(f"Scale {sr['scale']:>2}x median: {sr['median_s']:.3f} s  peak RSS {rss_str}{det_str}")
    exp = report.get("growth_exponent")
    if exp is not None:
        print(f"Growth exponent:  {exp:.3f}")
    print(f"\nTargets:")
    for k, v in report["targets"].items():
        print(f"  {k}: {v['status']} (value: {v['value']})")
    print(f"All targets met:  {report['all_targets_met']}")
    result_file = Path(args.out) / "benchmark_results.json"
    print(f"\nFull results: {result_file}")
    return 0 if report["all_targets_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

