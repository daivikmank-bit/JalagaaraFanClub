"""Unit tests for tools.build_lexicons and MANIFEST schema (Plan C V1 Phases 1, 4, 5)."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from kavachmed.indic import lexicon_tokens
from tools.build_lexicons import (
    LocalNameSource,
    SourceProvenance,
    build_seed_lexicon,
    build_term_frequencies,
    validate_manifest,
)


class TestNameLexiconsAndManifest(unittest.TestCase):
    def _prov(self, name: str = "Synthetic List", licence: str = "CC0-1.0") -> SourceProvenance:
        return SourceProvenance(
            name=name,
            url="https://example.invalid/names",
            licence=licence,
            retrieved="2026-11-26",
        )

    def test_manifest_validation_passes_with_two_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lex_dir = Path(tmp) / "lexicons"
            (lex_dir / "names").mkdir(parents=True)
            (lex_dir / "names" / "sample.txt").write_text("arif\nrahul\n", encoding="utf-8")
            manifest = {
                "names/sample.txt": {
                    "sources": [
                        self._prov("Source A").to_dict(),
                        self._prov("Source B").to_dict(),
                    ],
                    "build_script": "tools/build_lexicons.py",
                    "count": 2,
                }
            }
            (lex_dir / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
            validated = validate_manifest(lex_dir)
            self.assertIn("names/sample.txt", validated)

    def test_manifest_validation_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lex_dir = Path(tmp) / "lexicons"
            (lex_dir / "names").mkdir(parents=True)
            (lex_dir / "names" / "sample.txt").write_text("arif\nrahul\n", encoding="utf-8")

            # Missing licence
            bad_licence = {
                "names/sample.txt": {
                    "sources": [{"name": "A", "url": "u", "licence": "", "retrieved": "2026-11-26"}],
                    "build_script": "tools/build_lexicons.py",
                    "count": 2,
                }
            }
            (lex_dir / "MANIFEST.json").write_text(json.dumps(bad_licence), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_manifest(lex_dir)

            # Count mismatch
            bad_count = {
                "names/sample.txt": {
                    "sources": [self._prov().to_dict()],
                    "build_script": "tools/build_lexicons.py",
                    "count": 99,
                }
            }
            (lex_dir / "MANIFEST.json").write_text(json.dumps(bad_count), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_manifest(lex_dir)

            # Emitted file missing provenance
            (lex_dir / "names" / "extra.txt").write_text("sana\n", encoding="utf-8")
            valid_one = {
                "names/sample.txt": {
                    "sources": [self._prov().to_dict()],
                    "build_script": "tools/build_lexicons.py",
                    "count": 2,
                }
            }
            (lex_dir / "MANIFEST.json").write_text(json.dumps(valid_one), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_manifest(lex_dir)
            (lex_dir / "names" / "extra.txt").unlink()

            # Duplicate artifact path in raw JSON
            dup_json = (
                '{\n'
                '  "names/sample.txt": {"sources": [{"name":"A","url":"u","licence":"MIT","retrieved":"2026-11-26"}], "build_script": "b.py", "count": 2},\n'
                '  "names/sample.txt": {"sources": [{"name":"A","url":"u","licence":"MIT","retrieved":"2026-11-26"}], "build_script": "b.py", "count": 2}\n'
                '}\n'
            )
            (lex_dir / "MANIFEST.json").write_text(dup_json, encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_manifest(lex_dir)

    def test_seed_lexicon_reproducible_order_independent_and_preserves_other_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src1 = root / "a_names.txt"
            src2 = root / "b_names.txt"
            src1.write_text("Arif\nARIF\nब्रिजेश\n", encoding="utf-8")
            src2.write_text("Sana\nआरिफ\n", encoding="utf-8")

            s1 = LocalNameSource(src1, self._prov("S1"))
            s2 = LocalNameSource(src2, self._prov("S2"))

            lex_a = root / "lex_a"
            (lex_a / "drugs").mkdir(parents=True)
            (lex_a / "drugs" / "rx.txt").write_text("metformin\n", encoding="utf-8")
            (lex_a / "MANIFEST.json").write_text(
                json.dumps(
                    {
                        "drugs/rx.txt": {
                            "sources": [self._prov("Drugs").to_dict()],
                            "build_script": "tools/build_drugs.py",
                            "count": 1,
                        }
                    }
                ),
                encoding="utf-8",
            )

            build_seed_lexicon(lex_a, local_sources=[s1, s2], compress=True)
            manifest_a = json.loads((lex_a / "MANIFEST.json").read_text(encoding="utf-8"))
            self.assertIn("drugs/rx.txt", manifest_a)

            # Reordered sources and separate directory yield identical bytes for names + manifest
            lex_b = root / "lex_b"
            (lex_b / "drugs").mkdir(parents=True)
            (lex_b / "drugs" / "rx.txt").write_text("metformin\n", encoding="utf-8")
            (lex_b / "MANIFEST.json").write_text(
                json.dumps(
                    {
                        "drugs/rx.txt": {
                            "sources": [self._prov("Drugs").to_dict()],
                            "build_script": "tools/build_drugs.py",
                            "count": 1,
                        }
                    }
                ),
                encoding="utf-8",
            )
            build_seed_lexicon(lex_b, local_sources=[s2, s1], compress=True)

            self.assertEqual(
                (lex_a / "names" / "latin.txt.gz").read_bytes(),
                (lex_b / "names" / "latin.txt.gz").read_bytes(),
            )
            self.assertEqual(
                (lex_a / "names" / "devanagari.txt.gz").read_bytes(),
                (lex_b / "names" / "devanagari.txt.gz").read_bytes(),
            )
            self.assertEqual(
                (lex_a / "MANIFEST.json").read_bytes(),
                (lex_b / "MANIFEST.json").read_bytes(),
            )

            # Safe-word subtraction check for Owner B
            lex_c = root / "lex_c"
            build_seed_lexicon(lex_c, local_sources=[s1, s2], compress=False)
            lat_vocab = set((lex_c / "names" / "latin.txt").read_text(encoding="utf-8").splitlines())
            deva_vocab = set((lex_c / "names" / "devanagari.txt").read_text(encoding="utf-8").splitlines())
            name_vocab = lat_vocab | deva_vocab
            candidate_safe_words = set(lexicon_tokens("fever tablet Arif ब्रिजेश pulse"))
            subtracted = candidate_safe_words - name_vocab
            self.assertEqual(subtracted, {"fever", "tablet", "pulse"})

    def test_seed_lexicon_missing_file_or_metadata_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = LocalNameSource(root / "does_not_exist.txt", self._prov())
            with self.assertRaises(FileNotFoundError):
                build_seed_lexicon(root / "lex", local_sources=[missing])

            present = root / "ok.txt"
            present.write_text("Arif\n", encoding="utf-8")
            bad_meta = LocalNameSource(present, self._prov(licence=""))
            with self.assertRaises(ValueError):
                build_seed_lexicon(root / "lex", local_sources=[bad_meta])

    def test_term_frequencies_and_shard_partition_invariance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed_file = root / "seed.txt"
            seed_file.write_text("Arif\nSana\n", encoding="utf-8")

            shard_single = root / "all.csv"
            shard_single.write_text(
                "firstName,lastName,secretExtra\n"
                "Arif,Arif,RAW_LEAK_1\n"
                "ARIF,Khan,RAW_LEAK_2\n"
                " , ,RAW_LEAK_3\n"
                "Rohan,Mehta,RAW_LEAK_4\n",
                encoding="utf-8",
            )

            shard_p1 = root / "part1.csv"
            shard_p2 = root / "part2.csv"
            shard_p1.write_text(
                "firstName,lastName,secretExtra\n"
                "Arif,Arif,RAW_LEAK_1\n"
                "ARIF,Khan,RAW_LEAK_2\n",
                encoding="utf-8",
            )
            shard_p2.write_text(
                "firstName,lastName,secretExtra\n"
                " , ,RAW_LEAK_3\n"
                "Rohan,Mehta,RAW_LEAK_4\n",
                encoding="utf-8",
            )

            lex_1 = root / "lex1"
            build_seed_lexicon(lex_1, local_sources=[LocalNameSource(seed_file, self._prov())])
            res_1 = build_term_frequencies([shard_single], lex_1, self._prov("IndianPersona-fixture"))

            self.assertEqual(res_1.total_rows, 4)
            self.assertEqual(res_1.counts["arif"], 2)
            self.assertNotIn("khan", res_1.counts)  # generic token excluded
            self.assertEqual(res_1.counts["rohan"], 1)  # IndianPersona-only token present in TF

            seed_lat = (lex_1 / "names" / "latin.txt").read_text(encoding="utf-8").splitlines()
            self.assertNotIn("rohan", seed_lat)  # absent from seed vocabulary

            tf_bytes_1 = (lex_1 / "names" / "term_freq.csv").read_bytes()
            self.assertNotIn(b"RAW_LEAK", tf_bytes_1)
            self.assertNotIn(b"secretExtra", tf_bytes_1)

            # Partitioned shards yield identical term_freq.csv
            lex_2 = root / "lex2"
            build_seed_lexicon(lex_2, local_sources=[LocalNameSource(seed_file, self._prov())])
            res_2 = build_term_frequencies([shard_p2, shard_p1], lex_2, self._prov("IndianPersona-fixture"))
            self.assertEqual(res_1.total_rows, res_2.total_rows)
            self.assertEqual(
                tf_bytes_1,
                (lex_2 / "names" / "term_freq.csv").read_bytes(),
            )

            # Missing shard or missing column fails clearly
            with self.assertRaises(FileNotFoundError):
                build_term_frequencies([root / "missing.csv"], lex_1, self._prov())

            bad_col_shard = root / "bad_col.csv"
            bad_col_shard.write_text("given,family\nArif,Khan\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                build_term_frequencies([bad_col_shard], lex_1, self._prov())


if __name__ == "__main__":
    unittest.main()
