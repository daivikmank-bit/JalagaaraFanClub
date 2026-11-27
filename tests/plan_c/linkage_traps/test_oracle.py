import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.oracle import (
    OracleBoundaryError,
    OracleConfigError,
    is_oracle_enabled,
    load_oracle,
)
from tools.oracle.loader import Field, Record


def _build_temp_kit(root: Path, truth_payload: dict | str | None = None) -> tuple[Path, Path]:
    kit = root / "kit"
    dev_input = kit / "dev" / "input"
    dev_truth = kit / "dev" / "truth"
    dev_input.mkdir(parents=True)
    dev_truth.mkdir(parents=True)
    if truth_payload is not None:
        truth_file = dev_truth / "truth.json"
        if isinstance(truth_payload, str):
            truth_file.write_text(truth_payload, encoding="utf-8")
        else:
            truth_file.write_text(json.dumps(truth_payload), encoding="utf-8")
    return kit, dev_input


class TestOracleBoundaryAndAdapters(unittest.TestCase):
    def test_runtime_package_does_not_import_oracle(self) -> None:
        repo_root = str(Path(__file__).resolve().parents[3])
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys, kavachmed, kavachmed.indic; "
                "assert 'tools.oracle' not in sys.modules",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertFalse(is_oracle_enabled({}))
        self.assertTrue(is_oracle_enabled({"KM_ORACLE": "1"}))

    def test_boundary_refusals_never_read_truth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kit, dev_input = _build_temp_kit(root, truth_payload={"files": {}})
            sibling = root / "sibling_dir"
            sibling.mkdir()

            symlink_escape = dev_input / "escape_link"
            symlink_escape.symlink_to(sibling)

            with mock.patch("pathlib.Path.read_text") as mock_read:
                with self.assertRaises(OracleBoundaryError):
                    load_oracle(sibling, kit_dir=kit)
                with self.assertRaises(OracleBoundaryError):
                    load_oracle(dev_input / ".." / ".." / "sibling_dir", kit_dir=kit)
                with self.assertRaises(OracleBoundaryError):
                    load_oracle(symlink_escape, kit_dir=kit)
                mock_read.assert_not_called()

    def test_missing_malformed_and_duplicate_truth_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kit, dev_input = _build_temp_kit(root, truth_payload=None)
            with self.assertRaises(OracleConfigError):
                load_oracle(dev_input, kit_dir=kit)

            (kit / "dev" / "truth" / "truth.json").write_text("{bad json", encoding="utf-8")
            with self.assertRaises(OracleConfigError):
                load_oracle(dev_input, kit_dir=kit)

            # Duplicate RecKey via JSON text with duplicate keys is collapsed by json.loads,
            # so test schema validation failure on malformed units.
            (kit / "dev" / "truth" / "truth.json").write_text(
                json.dumps({"files": {"a.csv": {"units": "invalid"}}}),
                encoding="utf-8",
            )
            with self.assertRaises(OracleConfigError):
                load_oracle(dev_input, kit_dir=kit)

    def test_spans_dates_fields_and_coverage(self) -> None:
        truth = {
            "files": {
                "reg/patients.csv": {
                    "units": {
                        "row:0": {
                            "patient": "PT0001",
                            "link": "required",
                            "entities": [
                                {
                                    "type": "PATIENT_NAME",
                                    "surface": "आरिफ",
                                    "channel": "content",
                                    "role": "patient",
                                },
                                {
                                    "type": "PHONE",
                                    "surface": "9999900000",
                                    "channel": "content",
                                    "role": "patient",
                                },
                                {
                                    "type": "PHONE",
                                    "surface": "8888811111",
                                    "channel": "filename",
                                    "role": "patient",
                                },
                                {
                                    "type": "MRN",
                                    "surface": "XYZ",
                                    "channel": "weird_channel",
                                    "role": "patient",
                                },
                            ],
                            "dates": ["2025-06-30"],
                            "fields": {"age": 48, "gender": "M"},
                        }
                    }
                },
                "opd/visits.json": {
                    "units": {
                        "/visits/0": {
                            "patient": "PT0001",
                            "link": "required",
                            "entities": [
                                {
                                    "type": "PATIENT_NAME",
                                    "surface": "आरिफ",
                                    "channel": "content",
                                    "role": "patient",
                                }
                            ],
                            "dates": ["2025-07-02"],
                            "fields": {"age": 48},
                        }
                    }
                },
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            kit, dev_input = _build_temp_kit(Path(tmp), truth_payload=truth)
            oracle = load_oracle(dev_input, kit_dir=kit)

            csv_rec = Record(
                key=("reg/patients.csv", "row:0"),
                fields=[
                    Field(idx=0, path="name", text="नाम: आरिफ"),
                    Field(idx=1, path="notes", text="call 9999900000 or 9999900000"),
                    Field(idx=2, path="@meta", text="आरिफ"),  # ineligible for content
                    Field(idx=3, path="@filename", text="8888811111_rep.csv", vtype="path"),
                ],
                meta={},
            )
            json_rec = Record(
                key=("opd/visits.json", "/visits/0"),
                fields=[Field(idx=0, path="/visits/0/name", text="आरिफ")],
                meta={},
            )
            missing_rec = Record(key=("missing.txt", "doc"), fields=[], meta={})

            csv_spans = oracle.oracle_spans(csv_rec)
            json_spans = oracle.oracle_spans(json_rec)
            self.assertEqual(oracle.oracle_spans(missing_rec), [])

            self.assertEqual(len(csv_spans), 2)
            deva_span = csv_spans[0]
            self.assertEqual(csv_rec.fields[deva_span.field_idx].text[deva_span.start:deva_span.end], "आरिफ")
            self.assertEqual(csv_spans[1].field_idx, 3)

            self.assertEqual(len(json_spans), 1)
            self.assertEqual(json_spans[0].key, ("opd/visits.json", "/visits/0"))

            self.assertEqual(oracle.oracle_dates(csv_rec), ["2025-06-30"])
            self.assertEqual(oracle.oracle_fields(csv_rec), {"age": 48, "gender": "M"})

            cov = oracle.coverage.as_dict()
            self.assertEqual(cov["units_queried"], 3)
            self.assertEqual(cov["units_missing"], 1)
            self.assertEqual(cov["spans_emitted"], 3)
            self.assertEqual(cov["spans_ambiguous"], 1)
            self.assertEqual(cov["spans_unsupported_channel"], 1)

            # Same downstream callback accepts oracle and a detector stub
            def run_detector(detector_fn, rec):
                return detector_fn(rec)

            stub_spans = run_detector(lambda r: [], csv_rec)
            ora_spans = run_detector(oracle.oracle_spans, csv_rec)
            self.assertEqual(stub_spans, [])
            self.assertEqual(len(ora_spans), 2)


if __name__ == "__main__":
    unittest.main()
