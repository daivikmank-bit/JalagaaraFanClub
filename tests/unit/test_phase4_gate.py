"""Phase 4 (A10): Read-back gate, bounded repair, write-fault, and fail-closed tests."""
from __future__ import annotations

from pathlib import Path

from kavachmed import adapters, gate, inventory
from kavachmed.contracts import Edit, Hit, LocMap


def test_gate_worked_repair_on_second_pass(tmp_path: Path) -> None:
    src = '"नाम, x",LEAK_TOKEN\n'
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "t.csv").write_text(
        'col1,col2\n"नाम, x",LEAK_TOKEN\n', encoding="utf-8"
    )
    parsed = adapters.parse(inventory.scan(tmp_path / "in")[0], {})
    records_by_key = {r.key: r for r in parsed.records}

    def reproject(repairs: list[Edit]) -> tuple[bytes, LocMap]:
        assert parsed.projector is not None
        return parsed.projector(repairs, {})

    init_bytes, init_loc = reproject([])

    def scan_a(_txt: str, _path: str, _lk: dict, _ranges: list) -> list[Hit]:
        return []

    def scan_b(txt: str, path: str, _ranges: list) -> list[Hit]:
        idx = txt.find("LEAK_TOKEN")
        if idx == -1:
            return []
        return [Hit(path, "content", idx, idx + len("LEAK_TOKEN"), "B", "B1")]

    dest = tmp_path / "out" / "t.csv"
    ok, final_bytes, _, counts = gate.run_with_repair(
        dest, init_bytes, "sanitized/t.csv", init_loc, {},
        scan_a, scan_b, reproject, records_by_key, max_passes=3,
    )
    assert ok is True
    assert counts["B1"] == 1
    assert final_bytes is not None
    out_str = final_bytes.decode("utf-8")
    assert '"नाम, x"' in out_str
    assert "LEAK_TOKEN" not in out_str
    assert "[REDACTED]" in out_str


def test_gate_write_fault_caught_and_persistent_fault_quarantines(tmp_path: Path) -> None:
    clean_bytes = b"clean content\n"
    loc_map = LocMap([])
    seen_disk: list[str] = []

    def scan_a(_txt: str, _path: str, _lk: dict, _ranges: list) -> list[Hit]:
        return []

    def scan_b(txt: str, path: str, _ranges: list) -> list[Hit]:
        seen_disk.append(txt)
        idx = txt.find("DIRTY_SENTINEL")
        if idx != -1:
            return [Hit(path, "content", idx, idx + 14, "B", "B2")]
        return []

    # 1. Fault on first write only -> caught, retry writes clean_bytes -> passes
    calls = [0]

    def flaky_writer(p: Path, data: bytes) -> None:
        calls[0] += 1
        p.write_bytes(b"DIRTY_SENTINEL\n" if calls[0] == 1 else data)

    dest1 = tmp_path / "out1" / "f.txt"
    ok1, _, _, counts1 = gate.run_with_repair(
        dest1, clean_bytes, "sanitized/f.txt", loc_map, {},
        scan_a, scan_b, None, {}, max_passes=3, writer_fn=flaky_writer,
    )
    assert ok1 is True
    assert calls[0] == 2
    assert counts1["B2"] == 1
    assert "DIRTY_SENTINEL\n" in seen_disk

    # 2. Persistent fault -> bounded quarantine, file deleted
    dest2 = tmp_path / "out2" / "f.txt"
    ok2, _, _, counts2 = gate.run_with_repair(
        dest2, clean_bytes, "sanitized/f.txt", loc_map, {},
        scan_a, scan_b, None, {}, max_passes=3,
        writer_fn=lambda p, _d: p.write_bytes(b"DIRTY_SENTINEL\n"),
    )
    assert ok2 is False
    assert not dest2.exists()
    assert counts2["B2"] == 3


def test_gate_path_hit_unmappable_and_bad_bytes_fail_closed(tmp_path: Path) -> None:
    dest = tmp_path / "out" / "f.txt"

    # Path hit
    ok_p, _, _, _ = gate.run_with_repair(
        dest, b"ok", "sanitized/f.txt", LocMap([]), {},
        lambda *_: [],
        lambda _t, p, _r: [Hit(p, "path", 0, 5, "B", "B4")],
        lambda _e: (b"ok", LocMap([])), {}, max_passes=3,
    )
    assert ok_p is False and not dest.exists()

    # Scanner exception
    ok_e, _, _, _ = gate.run_with_repair(
        dest, b"ok", "sanitized/f.txt", LocMap([]), {},
        lambda *_: (_ for _ in ()).throw(RuntimeError("boom")),
        lambda *_: [], None, {}, max_passes=3,
    )
    assert ok_e is False and not dest.exists()

