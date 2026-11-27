"""Unit tests for kavachmed.indic (Plan C V1 Phases 1-3)."""

from __future__ import annotations

import unittest

from kavachmed.indic import NameToken, lexicon_tokens, normalize_name


class TestIndicNormalization(unittest.TestCase):
    def test_empty_punctuation_and_whitespace(self) -> None:
        self.assertEqual(normalize_name(""), ())
        self.assertEqual(normalize_name("   \t\n "), ())
        self.assertEqual(normalize_name("...---,,,!?!"), ())
        self.assertEqual(lexicon_tokens(""), ())
        self.assertEqual(lexicon_tokens("  ... "), ())

    def test_unicode_nfkc_casefold_and_latin_accents(self) -> None:
        for variant in ("Ａrif", "ARIF", "Ari\u0301f", "Aríf"):
            tokens = normalize_name(variant)
            self.assertEqual(len(tokens), 1)
            self.assertEqual(tokens[0].latin, "arif")
            self.assertEqual(tokens[0].kind, "identifying")
            self.assertEqual(tokens[0].skeleton, "rf")

    def test_initials_and_abbreviations(self) -> None:
        tokens = normalize_name("A. Arif")
        self.assertEqual([t.kind for t in tokens], ["initial", "identifying"])
        self.assertEqual(tokens[0].latin, "a")
        self.assertIsNone(tokens[0].skeleton)
        self.assertEqual(tokens[1].skeleton, "rf")

        md_tokens = normalize_name("Md. Arif")
        self.assertEqual(md_tokens[0], NameToken(surface="md", latin="mhmd", kind="generic", skeleton=None))
        self.assertEqual(md_tokens[1].skeleton, "rf")

    def test_honorifics_generics_and_aliases(self) -> None:
        deva_md = normalize_name("मोहम्मद")
        self.assertEqual(len(deva_md), 1)
        self.assertEqual(deva_md[0].surface, "मोहम्मद")
        self.assertEqual(deva_md[0].latin, "mhmd")
        self.assertEqual(deva_md[0].kind, "generic")
        self.assertIsNone(deva_md[0].skeleton)

        sahab = normalize_name("Arif sahab")
        self.assertEqual(sahab[1].kind, "honorific")
        self.assertIsNone(sahab[1].skeleton)

        kumar_singh = normalize_name("Kumar Singh")
        self.assertTrue(all(t.kind == "generic" and t.skeleton is None for t in kumar_singh))

    def test_lexicon_tokens_preserves_source_script_and_aliases(self) -> None:
        self.assertEqual(lexicon_tokens("मोहम्मद Mohd. Ari\u0301f"), ("मोहम्मद", "mohd", "arif"))

    def test_devanagari_transliteration_and_skeletons(self) -> None:
        brij_deva = normalize_name("ब्रिजेश")
        self.assertEqual(len(brij_deva), 1)
        self.assertEqual(brij_deva[0].latin, "brijesh")
        self.assertEqual(brij_deva[0].skeleton, "brjsh")

        for name in ("Brijesh", "Brajesh", "ब्रिजेश"):
            toks = normalize_name(name)
            self.assertEqual(toks[0].skeleton, "brjsh")

        self.assertEqual(normalize_name("Sukhwinder")[0].skeleton, normalize_name("Sukhvinder")[0].skeleton)

    def test_cross_script_arif_variants_share_identifying_skeleton(self) -> None:
        variants = ("Mohammad Arif", "मोहम्मद आरिफ", "Md. Arif", "Arif sahab")
        for variant in variants:
            toks = normalize_name(variant)
            identifying = [t.skeleton for t in toks if t.kind == "identifying"]
            self.assertEqual(identifying, ["rf"], msg=f"Failed on {variant!r}: {toks}")

        self.assertEqual(normalize_name("Mohammad Arif")[0].kind, "generic")
        self.assertEqual(normalize_name("Arif sahab")[1].kind, "honorific")

    def test_schwa_and_conjunct_table(self) -> None:
        cases = (
            ("रमा", "rama"),
            ("कमल", "kamal"),
            ("ब्रिजेश", "brijesh"),
        )
        for source, expected_latin in cases:
            with self.subTest(source=source):
                toks = normalize_name(source)
                self.assertEqual(len(toks), 1)
                self.assertEqual(toks[0].latin, expected_latin)

    def test_distinct_and_colliding_skeletons_and_no_skeleton_cases(self) -> None:
        self.assertNotEqual(normalize_name("Arif")[0].skeleton, normalize_name("Amit")[0].skeleton)
        self.assertEqual(normalize_name("Sana")[0].skeleton, normalize_name("Sona")[0].skeleton)

        for empty_skel in ("Ai", "ईआ"):
            toks = normalize_name(empty_skel)
            self.assertEqual(len(toks), 1)
            self.assertIsNone(toks[0].skeleton)

        unsupported = normalize_name("அருண்")
        self.assertEqual(len(unsupported), 1)
        self.assertIsNone(unsupported[0].latin)
        self.assertIsNone(unsupported[0].skeleton)


if __name__ == "__main__":
    unittest.main()
