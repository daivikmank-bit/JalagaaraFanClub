"""Plan B attack probes and the kit's optional submission check.

The synthetic probes exercise detector spans, not the exported release or the
organisers' attacker. A2-A7 require the integrated system and sealed evaluator.
"""

from __future__ import annotations

import base64
import os
import subprocess
import sys
from pathlib import Path
from typing import Literal

import pytest

from kavachmed.contracts import Context, Field, Record, Span
from kavachmed.detect.arbiter import detect
from kavachmed.propagate import build_index, sweep
from kavachmed.shield import protect


ROOT = Path(__file__).resolve().parents[2]
KIT = ROOT.parent / "Secret-Wards"


def _words(path: Path) -> set[str]:
    return {
        line.strip().lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


@pytest.fixture
def context() -> Context:
    lexicons = ROOT / "config" / "lexicons"
    return Context(
        policy={"detect": {"stack": "recall_first"}},
        lexicons={
            "safe_words": _words(lexicons / "safe" / "safe_words.txt"),
            "names": _words(lexicons / "names" / "generic_names_seed.txt"),
            "drugs": _words(lexicons / "clinical" / "drugs.txt"),
        },
        doctor_registry={"Ramesh Srivastava"},
    )


def _covered(text: str, target: str, spans: list[Span], field_idx: int = 0) -> bool:
    start = text.index(target)
    return all(
        any(s.field_idx == field_idx and s.start <= pos < s.end for s in spans)
        for pos in range(start, start + len(target))
        if text[pos].isalnum()
    )


@pytest.mark.parametrize(
    ("text", "target", "vtype"),
    [
        ("Name: Velora Neris", "Velora Neris", "str"),
        ("नाम: देवेंद्र राठौड़", "देवेंद्र राठौड़", "str"),
        ("Zoravar ji ka BP 120/80", "Zoravar ji", "str"),
        ("Phone: +91 98765 43210", "+91 98765 43210", "str"),
        ("Aadhaar XXXX-XXXX-4321", "XXXX-XXXX-4321", "str"),
        ("Email audit%40example.org", "audit%40example.org", "str"),
        ("exports/Velora_Neris_9876543210.json", "9876543210", "path"),
    ],
    ids=["latin-name", "devanagari-name", "hinglish-name", "phone", "masked-id", "encoded-email", "filename"],
)
def test_a1_direct_exposure_detector_spans(
    context: Context, text: str, target: str, vtype: Literal["str", "path"]
):
    rec = Record(key=("synthetic/a1", "doc"), fields=[Field(0, "doc", text, vtype=vtype)])
    assert _covered(text, target, detect(rec, context))


def test_a1_encoded_attachment_detector_span(context):
    blob = base64.b64encode(b"Phone: 9876543210").decode("ascii")
    text = f"Attachment: {blob}"
    rec = Record(key=("synthetic/a1-attachment", "doc"), fields=[Field(0, "doc", text)])
    assert _covered(text, blob, detect(rec, context))


@pytest.mark.parametrize(
    "masked",
    ["98XXXXXX10", "98-xxxxxx-10", "+91 98******10", "९८XXXXXX१०"],
)
def test_a2_masked_phone_is_not_left_as_linkage_evidence(context, masked):
    text = f"Phone: {masked}"
    rec = Record(key=("synthetic/a2", "doc"), fields=[Field(0, "doc", text)])
    spans = detect(rec, context)
    assert _covered(text, masked, spans)
    assert any(s.typ == "PHONE" and text[s.start:s.end] == masked for s in spans)


def test_a4_cluster_sweep_adds_missing_name_without_harming_doctor(context):
    seed = Record(key=("synthetic/seed", "doc"), fields=[Field(0, "doc", "Relative: Velora Neris")])
    target_text = "velora neris visited. Dr. Ramesh Srivastava; BP 120/80"
    target = Record(key=("synthetic/target", "doc"), fields=[Field(0, "doc", target_text)])

    class Resolution:
        cluster_of = {seed.key: "one", target.key: "one"}
        family_of = {}

    resolution = Resolution()
    seed_spans = detect(seed, context)
    index = build_index([seed, target], resolution, {seed.key: seed_spans})
    shield = protect(target, context)
    spans = sweep(target, resolution, [], protected_spans=shield, index=index)
    assert _covered(target_text, "velora neris", spans)
    for keep in ("Dr. Ramesh Srivastava", "BP 120/80"):
        start = target_text.index(keep)
        assert not any(s.start < start + len(keep) and s.end > start for s in spans)


@pytest.mark.parametrize("attack", ["A2/A3", "A4", "A5", "A6", "A7"])
def test_full_attack_arena_requires_organiser_evaluator(attack):
    pytest.skip(f"{attack} needs an integrated release and the organiser's unpublished attack evaluator")


def test_official_three_run_submission_check():
    if not (ROOT / "raksha.toml").is_file():
        pytest.skip("Plan B has no root raksha.toml or integrated submission runner")
    if os.environ.get("KAVACHMED_RUN_SUBMISSION_CHECK") != "1":
        pytest.skip("set KAVACHMED_RUN_SUBMISSION_CHECK=1 to run the kit's three-run check")
    checker = KIT / "check_submission.py"
    assert checker.is_file(), "Secret-Wards/check_submission.py is missing"
    result = subprocess.run(
        [sys.executable, str(checker), "--repo", str(ROOT), "--run"],
        cwd=ROOT,
        capture_output=True,
        timeout=2800,
        check=False,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert result.returncode == 0, "Official pre-submit check failed; inspect it locally without exposing sensitive logs"
