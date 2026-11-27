"""Output emitter for report.json and records.jsonl (A11)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def peak_rss_mb() -> float:
    """Measure peak RSS in MB (Linux VmHWM -> Unix ru_maxrss -> 0.0 fallback)."""
    proc_status = Path("/proc/self/status")
    if proc_status.is_file():
        for line in proc_status.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("VmHWM:"):
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    return round(int(parts[1]) / 1024.0, 1)
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(r / 1048576.0 if sys.platform == "darwin" else r / 1024.0, 1)
    except Exception:  # noqa: BLE001
        return 0.0


def write(
    out_dir: Path,
    file_reports: list[dict[str, Any]],
    record_rows: list[dict[str, Any]],
    entity_counts: dict[str, int],
    identifiers_detected: int,
    records_generalized: int,
    records_suppressed: int,
    elapsed_seconds: float,
    records_quarantined: int = 0,
    stage_seconds: dict[str, float] | None = None,
) -> None:
    exported = {f["output"] for f in file_reports if f.get("output")}
    filtered_records = sorted(
        (r for r in record_rows if r["file"] in exported),
        key=lambda r: (r["file"], r["loc"]),
    )

    with (out_dir / "records.jsonl").open("w", encoding="utf-8") as fh:
        for row in filtered_records:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    n_san = sum(1 for f in file_reports if f.get("status") == "sanitized")
    n_quar = sum(1 for f in file_reports if f.get("status") == "quarantined")
    report = {
        "schema": "raksha-report/1",
        "files": file_reports,
        "counts": {
            "files_total": len(file_reports),
            "files_sanitized": n_san,
            "files_quarantined": n_quar,
            "records_processed": len(filtered_records),
            "identifiers_detected": identifiers_detected,
            "records_linked": sum(1 for r in filtered_records if r.get("patient") is not None),
            "records_generalized": records_generalized,
            "records_suppressed": records_suppressed,
            "records_quarantined": records_quarantined,
        },
        "entities": dict(sorted(entity_counts.items())),
        "timing": {"seconds": round(elapsed_seconds, 2), "stages": stage_seconds or {}},
        "resources": {"peak_rss_mb": peak_rss_mb()},
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
