"""Phase 3 Step 5 (A9): Path, directory, key, and header renaming + collisions."""
from __future__ import annotations

from pathlib import Path

from kavachmed import adapters, inventory, rename
from kavachmed.contracts import Span
from kavachmed.security import load_keys


def test_filename_signals_harvested_before_renaming() -> None:
    sig = rename.extract_filename_signals("lab/7582322248_report_0047.json")
    assert "7582322248" in sig["digits"]


def test_path_allocation_collisions_casefold_and_shared_parent() -> None:
    keys = load_keys({"RAKSHA_SECRET": "practice"}, pbkdf2_iterations=1000)
    paths = [
        "Ishita_Varshney/note1.txt",
        "Ishita_Varshney/note2.txt",
        "lab/7582322248_report.json",
        "lab/report.json",
        "clean/protocol.txt",
    ]
    out1 = rename.allocate_output_paths(paths, keys)
    out2 = rename.allocate_output_paths(paths, keys)
    assert out1 == out2

    # Shared sensitive parent directory maps consistently for both children
    p1 = Path(out1["Ishita_Varshney/note1.txt"]).parts[0]
    p2 = Path(out1["Ishita_Varshney/note2.txt"]).parts[0]
    assert p1 == p2
    assert "Anita" not in p1 and "Sharma" not in p1

    # Phone number stripped from filename
    assert "7582322248" not in out1["lab/7582322248_report.json"]
    # Clean reference path preserved
    assert out1["clean/protocol.txt"] == "clean/protocol.txt"
    # All output paths unique (even case-folded)
    assert len({v.lower() for v in out1.values()}) == len(paths)

    # Different secret -> different tokens
    keys2 = load_keys({"RAKSHA_SECRET": "other-secret"}, pbkdf2_iterations=1000)
    out_other = rename.allocate_output_paths(paths, keys2)
    assert out_other["lab/7582322248_report.json"] != out1["lab/7582322248_report.json"]


def test_header_and_key_renaming(tmp_path: Path) -> None:
    (tmp_path / "vitals.csv").write_bytes(b"JJH-0411556 Zainab,BP\n120/80,ok\n")
    parsed = adapters.parse(inventory.scan(tmp_path)[0], {})
    hdr = next(r for r in parsed.records if r.key[1] == "@header")
    sp = Span(hdr.key, 0, 0, len(hdr.fields[0].text), "PATIENT_NAME", "patient", 0.9, "context")
    ed = rename.rename_headers_and_keys(parsed, [sp])
    assert parsed.projector is not None
    out_bytes, _ = parsed.projector(ed, {})
    out_str = out_bytes.decode("utf-8")
    assert "Zainab" not in out_str
    assert "JJH-0411556" not in out_str
    assert "col_0,BP" in out_str

