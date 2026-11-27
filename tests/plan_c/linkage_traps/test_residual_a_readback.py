"""Read-back gate and quarantine tests for Plan C V8 (Phase 4 of c-v8.md)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kavachmed.evidence import Evidence
from kavachmed.indic import normalize_name
from kavachmed.residual_a import ExemptSpan, Hit, LeakKeys, compile_keys, scan_with_status
from tools.residual_eval import E11Fixture, run_e11_ablation


def _run_synthetic_readback_gate(
    out_dir: Path,
    rel_path: str,
    in_memory_text: str,
    keys: LeakKeys,
    *,
    exempt_index: list[ExemptSpan] | None = None,
    disk_injector: str | None = None,
    max_passes: int = 3,
    unrepairable: bool = False,
) -> dict[str, object]:
    """Synthetic A8/A10 read-back gate fixture: write -> close -> re-read -> repair/quarantine."""
    target = out_dir / "sanitized" / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)

    # Simulate clean in-memory content, optionally dirtied on disk by a faulty writer.
    written_text = disk_injector if disk_injector is not None else in_memory_text
    target.write_text(written_text, encoding="utf-8")

    repairs = 0
    for attempt in range(max_passes + 1):
        read_back = target.read_text(encoding="utf-8")
        outcome = scan_with_status(read_back, rel_path, keys, exempt_index=exempt_index)
        if outcome.verified and not outcome.hits:
            return {
                "status": "passed",
                "out_path": f"sanitized/{rel_path}",
                "repairs": repairs,
                "reason": None,
            }
        if attempt == max_passes or not outcome.verified:
            target.unlink(missing_ok=True)
            return {
                "status": "quarantined",
                "out_path": None,
                "repairs": repairs,
                "reason": "residual_pii" if outcome.hits else (outcome.reason or "unverified_scan"),
            }
        # Targeted default-deny repair on hit ranges (unless simulated unrepairable).
        repairs += 1
        if unrepairable:
            continue
        chars = list(read_back)
        for h in sorted(outcome.hits, key=lambda x: x.start, reverse=True):
            if h.channel == "content":
                chars[h.start : h.end] = list("[REDACTED]")
        target.write_text("".join(chars), encoding="utf-8")

    target.unlink(missing_ok=True)
    return {"status": "quarantined", "out_path": None, "repairs": repairs, "reason": "residual_pii"}


class TestResidualAReadbackGate(unittest.TestCase):
    def setUp(self) -> None:
        self.keys = compile_keys(
            [],
            [
                Evidence(
                    key=("src.txt", "doc"),
                    names=normalize_name("Kavita"),
                    phones=["9876543210"],
                )
            ],
        )

    def test_disk_dirty_fault_injection_repaired_on_reread(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            res = _run_synthetic_readback_gate(
                Path(tmp),
                "visit.txt",
                in_memory_text="Patient [REDACTED] stable.",
                disk_injector="Patient Kavita stable.",
                keys=self.keys,
            )
            self.assertEqual(res["status"], "passed")
            self.assertEqual(res["repairs"], 1)
            self.assertEqual(res["out_path"], "sanitized/visit.txt")
            final_text = (Path(tmp) / "sanitized" / "visit.txt").read_text(encoding="utf-8")
            self.assertNotIn("Kavita", final_text)

    def test_persistent_leak_quarantines_and_removes_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            res = _run_synthetic_readback_gate(
                Path(tmp),
                "bad.txt",
                in_memory_text="Patient Kavita stable.",
                keys=self.keys,
                unrepairable=True,
                max_passes=3,
            )
            self.assertEqual(res["status"], "quarantined")
            self.assertIsNone(res["out_path"])
            self.assertEqual(res["reason"], "residual_pii")
            self.assertFalse((Path(tmp) / "sanitized" / "bad.txt").exists())

    def test_clean_controls_pass_without_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            text = "Dr. Kavita ordered Hb 13.1 g/dL and Tab Metformin 500mg."
            doc_end = len("Dr. Kavita")
            res = _run_synthetic_readback_gate(
                Path(tmp),
                "clean_ctrl.txt",
                in_memory_text=text,
                keys=self.keys,
                exempt_index=[
                    ExemptSpan("clean_ctrl.txt", "content", 0, doc_end, exempt_kind="doctor_registry")
                ],
            )
            self.assertEqual(res["status"], "passed")
            self.assertEqual(res["repairs"], 0)

    def test_e11_ablation_four_configurations(self) -> None:
        def _stub_layer_b(text: str, out_path: str) -> list[Hit]:
            marker = "UnseenNameZ"
            idx = text.find(marker)
            if idx == -1:
                return []
            return [Hit(out_path=out_path, channel="content", start=idx, end=idx + len(marker), layer="B", check="B1")]

        fixtures = [
            E11Fixture("harvested.txt", "Patient Kavita 9876543210", planted_harvested=2),
            E11Fixture("unseen.txt", "Relative UnseenNameZ", planted_unseen=1),
            E11Fixture(
                "ctrl.txt",
                "Dr. Kavita ordered Tab Metformin 500mg",
                is_clean_control=True,
                exempt_index=(ExemptSpan("ctrl.txt", "content", 0, 10, exempt_kind="doctor_registry"),),
            ),
        ]
        res = run_e11_ablation(fixtures, self.keys, layer_b_scanner=_stub_layer_b)
        self.assertGreater(res["off"].harvested_survivors, 0)
        self.assertEqual(res["a_only"].harvested_survivors, 0)
        self.assertGreater(res["a_only"].unseen_survivors, 0)
        self.assertEqual(res["both"].harvested_survivors, 0)
        self.assertEqual(res["both"].unseen_survivors, 0)
        self.assertEqual(res["both"].good_file_quarantines, 0)
        self.assertEqual(res["both"].over_redaction_controls, 0)
        self.assertLess(res["a_only"].scan_seconds, 0.8)


if __name__ == "__main__":
    unittest.main()
