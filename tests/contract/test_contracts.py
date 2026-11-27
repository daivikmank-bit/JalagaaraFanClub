"""Contract tests verifying every stage stub adheres to kavachmed.contracts."""
from __future__ import annotations

from pathlib import Path

from kavachmed import (
    adapters,
    detect,
    edits,
    evidence,
    inventory,
    linkage,
    privacy,
    propagate,
    residual_a,
    residual_b,
    shield,
    typing as col_typing,
)
from kavachmed.contracts import Evidence, LocMap, PrivacyPlan, Resolution
from kavachmed.security import load_keys


def test_end_to_end_stub_contracts(tmp_path: Path) -> None:
    inp = tmp_path / "inp"
    inp.mkdir()
    (inp / "sample.csv").write_text("id,note\n1,Dolo 1-0-1\n", encoding="utf-8")
    (inp / "sample.json").write_text('{"visits":[{"note":"Dr. X reviewed"}]}', encoding="utf-8")
    (inp / "sample.txt").write_text("Patient fever 1-0-1\n", encoding="utf-8")

    entries = inventory.scan(inp)
    assert len(entries) == 3

    keys = load_keys({"RAKSHA_SECRET": "practice"}, pbkdf2_iterations=1000)
    policy: dict = {}

    evidences: list[Evidence] = []
    parsed_list = []
    for entry in entries:
        pf = adapters.parse(entry, policy)
        col_typing.type_columns(pf)
        parsed_list.append(pf)
        for rec in pf.records:
            sh = shield.protect(rec, {"policy": policy, "keys": keys})
            det = detect.detect(rec, {"policy": policy, "keys": keys}, sh)
            evidences.append(evidence.extract(rec, sh + det))

    # C's linkage fails closed without input digests (orphan shift offsets are file-keyed)
    res = linkage.resolve(evidences, keys, policy, file_digests={e.rel_path: e.sha256 for e in entries})
    assert isinstance(res, Resolution)
    plan = privacy.plan(res, evidences, parsed_list, policy)
    assert isinstance(plan, PrivacyPlan)

    for pf in parsed_list:
        file_edits = []
        for rec in pf.records:
            swept = propagate.sweep(rec, res, [])
            file_edits.extend(edits.build(rec, swept, res, plan))
        assert pf.projector is not None
        out_bytes, loc_map = pf.projector(file_edits, res.patient_of)
        assert isinstance(out_bytes, bytes)
        assert isinstance(loc_map, LocMap)
        assert residual_a.scan(out_bytes.decode("utf-8"), pf.rel_path, {}, []) == []
        assert residual_b.scan(out_bytes.decode("utf-8"), pf.rel_path, []) == []

