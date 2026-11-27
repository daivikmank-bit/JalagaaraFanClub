"""Synthetic Plan B phase 3 boundary tests."""

import json

import pytest

from kavachmed.contracts import Context, Field, Record
from kavachmed.detect.labels import detect_labels, induce_labels, write_promoted_labels


def _record(file, text):
    return Record(key=(file, "doc"), fields=[Field(0, "custommarker", text)])


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["Velora Neris", "Zarela Vora"], False),
        (["Velora Neris", "Zarela Vora", "normal", "negative", "positive"], False),
        (["Velora Neris", "Zarela Vora", "Tarev Neris", "negative", "positive"], True),
    ],
)
def test_induction_minimum_and_ratio(tmp_path, values, expected):
    ctx = Context()
    records = [_record(f"synthetic/{i}.txt", value) for i, value in enumerate(values)]
    promoted = induce_labels(records, ctx)
    assert ("custommarker" in promoted) is expected
    if expected:
        artefact = tmp_path / "labels.json"
        write_promoted_labels(promoted, artefact)
        labels = json.loads(artefact.read_text(encoding="utf-8"))["labels"]
        rec = _record("synthetic/unknown.txt", "zx9912")
        spans = detect_labels(rec, ctx, labels)
        assert any(s.start == 0 and s.end == len(rec.fields[0].text) for s in spans)
