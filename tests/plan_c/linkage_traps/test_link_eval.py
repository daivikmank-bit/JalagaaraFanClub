import json
import tempfile
import unittest
from pathlib import Path

from tools.link_eval import (
    compute_b3_metrics,
    compute_parity_metrics,
    evaluate_output,
)


class TestLinkEval(unittest.TestCase):
    def test_parity_and_b3_on_mixed_cluster_fixture(self) -> None:
        # Truth: {a,b}, {c,d}; Pred: {a,b,c}, unlinked d
        units = [
            ("P1", "PSN-1"),
            ("P1", "PSN-1"),
            ("P2", "PSN-1"),
            ("P2", None),
        ]
        parity = compute_parity_metrics(units)
        self.assertEqual(parity.tp, 1)
        self.assertEqual(parity.fp, 2)
        self.assertEqual(parity.fn, 1)
        self.assertAlmostEqual(parity.precision, 1 / 3)
        self.assertAlmostEqual(parity.recall, 1 / 2)
        self.assertAlmostEqual(parity.f05, 5 / 14)
        self.assertEqual(parity.merged_clusters, 1)

        b3 = compute_b3_metrics(units)
        self.assertAlmostEqual(b3.precision, 2 / 3)
        self.assertAlmostEqual(b3.recall, 3 / 4)
        self.assertAlmostEqual(b3.f1, 12 / 17)
        self.assertFalse(b3.all_null)

    def test_perfect_all_null_and_empty_predictions(self) -> None:
        perfect = compute_parity_metrics([("P1", "A"), ("P1", "A"), ("P2", "B")])
        self.assertEqual((perfect.tp, perfect.fp, perfect.fn), (1, 0, 0))
        self.assertAlmostEqual(perfect.f05, 1.0)

        all_null_units = [("P1", None), ("P1", None), ("P2", None)]
        null_p = compute_parity_metrics(all_null_units)
        self.assertEqual((null_p.tp, null_p.fp, null_p.fn), (0, 0, 1))
        self.assertEqual(null_p.f05, 0.0)
        null_b3 = compute_b3_metrics(all_null_units)
        self.assertTrue(null_b3.all_null)
        self.assertTrue(null_b3.documented_identity_score_zero)

        empty_p = compute_parity_metrics([])
        self.assertEqual(empty_p.evaluated_records, 0)
        self.assertEqual(empty_p.f05, 0.0)

    def test_output_dir_renames_fallbacks_explicit_null_and_traps(self) -> None:
        truth = {
            "traps": {
                "language": ["P1"],
                "homonym": ["P1", "P2"],
            },
            "files": {
                "lab/7148_rep.json": {
                    "sha256": "sha_lab",
                    "units": {
                        "/visits/0/obs": {"patient": "P1", "link": "required"},
                        "/visits/1/obs": {"patient": "P1", "link": "required"},
                    },
                },
                "notes/n1.txt": {
                    "sha256": "sha_note",
                    "units": {
                        "doc": {"patient": "P2", "link": "required"},
                    },
                },
                "notes/ambig.txt": {
                    "sha256": "sha_ambig",
                    "units": {
                        "doc": {"patient": "P2", "link": "ambiguous"},
                    },
                },
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            truth_path = root / "truth.json"
            truth_path.write_text(json.dumps(truth), encoding="utf-8")

            out_dir = root / "out"
            out_dir.mkdir()
            (out_dir / "report.json").write_text(
                json.dumps(
                    {
                        "files": [
                            {"input_sha256": "sha_lab", "output": "sanitized/lab/renamed.json"}
                        ]
                    }
                ),
                encoding="utf-8",
            )
            records = [
                {"file": "sanitized/lab/renamed.json", "loc": "/visits/0", "patient": "PSN-A"},
                {"file": "sanitized/lab/renamed.json", "loc": "/visits/1/obs", "patient": None},
                {"file": "sanitized/lab/renamed.json", "loc": "doc", "patient": "PSN-FALLBACK"},
                {"file": "sanitized/notes/n1.txt", "loc": "doc", "patient": "PSN-A"},
                {"file": "sanitized/notes/ambig.txt", "loc": "doc", "patient": "PSN-X"},
            ]
            (out_dir / "records.jsonl").write_text(
                "\n".join(json.dumps(r) for r in records) + "\n",
                encoding="utf-8",
            )

            res = evaluate_output(truth_path, out_dir)
            # /visits/0/obs -> PSN-A (ancestor), /visits/1/obs -> None (explicit null overrides doc!),
            # notes/n1.txt -> PSN-A (false merge with P1)
            self.assertEqual(res["parity"]["tp"], 0)
            self.assertEqual(res["parity"]["fp"], 1)
            self.assertEqual(res["parity"]["fn"], 1)
            self.assertEqual(res["parity"]["merged_clusters"], 1)

            diag = res["diagnostics"]
            self.assertEqual(diag["language_pair_recall"]["status"], "ok")
            self.assertEqual(diag["language_pair_recall"]["value"], 0.0)
            self.assertEqual(diag["fragmented_pair_recall"]["status"], "unavailable")
            self.assertEqual(diag["homonym_separation"]["status"], "ok")
            self.assertEqual(diag["homonym_separation"]["value"], 0.0)
            self.assertEqual(diag["ambiguous_unit_merges"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
