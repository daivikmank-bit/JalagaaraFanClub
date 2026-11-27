"""Unit tests for Fellegi-Sunter scoring, policy validation, TF, and waterfall (Plan C V5)."""

from __future__ import annotations

import math
import unittest

from kavachmed.evidence import Evidence
from kavachmed.indic import normalize_name
from kavachmed.linkage import LinkageError, resolve
from kavachmed.linkage.fs import jaro_winkler, score_pair
from kavachmed.linkage.policy import DEFAULT_FS_DISTRIBUTIONS, parse_linkage_policy, validate_distributions
from tools.waterfall import from_pair_trace, render_waterfall


class _StubKeys:
    def keyed(self, purpose: str, *fields: str) -> bytes:
        import hashlib

        return hashlib.sha256(f"{purpose}|{'|'.join(fields)}".encode("utf-8")).digest()


def _ev(
    rec_id: str,
    *,
    name: str = "",
    kin: str = "",
    sex: str | None = None,
    birth_years: tuple[int, int] | None = None,
    ids: dict[str, list[str]] | None = None,
    phones: list[str] | None = None,
    phone_shared: bool = False,
    district: str | None = None,
    hospital: str | None = None,
) -> Evidence:
    geo = {"district_code": district} if district else {}
    return Evidence(
        key=(f"{rec_id}.json", "/0"),
        names=normalize_name(name) if name else [],
        kin=normalize_name(kin) if kin else [],
        sex=sex,
        birth_years=birth_years,
        ids=ids or {},
        phones=phones or [],
        phone_shared=phone_shared,
        geo=geo,
        hospital=hospital,
    )


class TestPolicyAndFS(unittest.TestCase):
    def test_policy_validation_and_legacy_rejection(self) -> None:
        validate_distributions(DEFAULT_FS_DISTRIBUTIONS)
        cfg = parse_linkage_policy(None)
        self.assertEqual(cfg.prior_bits, -4.0)
        self.assertEqual(cfg.tau_high_bits, 4.0)
        self.assertEqual(cfg.tau_low_bits, 1.0)

        with self.assertRaises(LinkageError) as ctx:
            parse_linkage_policy({"match_threshold": 0.9})
        self.assertEqual(ctx.exception.code, "legacy_probability_policy_key")
        self.assertIn("tau_high_bits", str(ctx.exception))

        for bad in (
            {"tau_high_bits": 4.0},
            {"tau_high_bits": 2.0, "tau_low_bits": 2.0},
            {"tau_high_bits": float("nan"), "tau_low_bits": 1.0},
            {"prior_bits": float("inf")},
        ):
            with self.assertRaises(LinkageError):
                parse_linkage_policy(bad)

    def test_exact_only_baseline_path(self) -> None:
        a = _ev("a", name="Arif Khan", sex="M", birth_years=(1980, 1980), kin="Salim")
        b = _ev("b", name="आरिफ", sex="M", birth_years=(1980, 1980), kin="सलीम")
        res = resolve(
            [a, b],
            _StubKeys(),
            policy={"mode": "exact_only"},
            file_digests={"a.json": "11" * 16, "b.json": "22" * 16},
        )
        self.assertIsNone(res.patient_of[a.key])
        self.assertIsNone(res.patient_of[b.key])

    def test_name_levels_jw_boundary_and_generic_guard(self) -> None:
        # Exact 0.90 Jaro-Winkler boundary check
        self.assertGreaterEqual(jaro_winkler("martha", "marhta"), 0.90)
        self.assertLess(jaro_winkler("ramesh", "suresh"), 0.90)

        # Generic tokens alone cannot create agreement
        g1 = _ev("g1", name="Kumar Singh")
        g2 = _ev("g2", name="Kumar Sharma")
        tr_g = score_pair(g1, g2)
        self.assertEqual(tr_g.contributions[0].level, "missing")

        # Exact, skeleton, JW, initial, disagreement
        cases = [
            (_ev("a", name="Ramesh"), _ev("b", name="Ramesh"), "exact"),
            (_ev("a", name="Arif"), _ev("b", name="Aarif"), "skeleton"),
            (_ev("a", name="martha"), _ev("b", name="marhta"), "jaro_winkler"),
            (_ev("a", name="R. Sharma"), _ev("b", name="Ramesh"), "initial"),
            (_ev("a", name="Ramesh"), _ev("b", name="Suresh"), "disagreement"),
        ]
        for ea, eb, expected_lvl in cases:
            tr = score_pair(ea, eb)
            self.assertEqual(tr.contributions[0].level, expected_lvl)

    def test_custom_m_u_two_bits_and_tf_correction(self) -> None:
        # m=0.8, u=0.2 -> log2(0.8 / 0.2) = 2.0 bits
        custom = dict(DEFAULT_FS_DISTRIBUTIONS)
        custom["sex"] = {"agrees": (0.8, 0.2), "disagrees": (0.2, 0.8)}
        ea = _ev("a", sex="M")
        eb = _ev("b", sex="M")
        tr = score_pair(ea, eb, policy={"fs_distributions": custom})
        sex_c = next(c for c in tr.contributions if c.feature == "sex")
        self.assertAlmostEqual(sex_c.base_bits, 2.0, places=6)

        # TF: rare vs common token within [-2, 2], least rare shared token wins, unknown -> 0
        freqs = {"rarename": 1, "commonname": 5000}
        e_rare1 = _ev("r1", name="Rarename")
        e_rare2 = _ev("r2", name="Rarename")
        tr_rare = score_pair(
            e_rare1,
            e_rare2,
            policy={"tf_total_rows": 10000, "tf_frequencies": freqs},
        )
        tr_comm = score_pair(
            _ev("c1", name="Rarename Commonname"),
            _ev("c2", name="Rarename Commonname"),
            policy={"tf_total_rows": 10000, "tf_frequencies": freqs},
        )
        tr_notf = score_pair(e_rare1, e_rare2, policy={"tf_total_rows": None})
        self.assertFalse(tr_notf.tf_enabled)
        self.assertGreater(
            tr_rare.contributions[0].tf_adjusted_bits,
            tr_rare.contributions[0].base_bits,
        )
        # Common token (5000/10000 = 0.5 -> log2(0.01/0.5) < 0) gives negative TF adjustment
        self.assertLess(
            tr_comm.contributions[0].tf_adjusted_bits,
            tr_comm.contributions[0].base_bits,
        )
        self.assertAlmostEqual(
            tr_comm.contributions[0].tf_adjusted_bits - tr_comm.contributions[0].base_bits,
            -2.0,
            places=6,
        )

    def test_shared_phone_vetoes_and_waterfall_adapter(self) -> None:
        ea = _ev("a", name="Arif", sex="M", birth_years=(1980, 1980), phones=["9811122233"])
        eb = _ev("b", name="Arif", sex="F", birth_years=(1980, 1980), phones=["9811122233"])
        tr = score_pair(
            ea,
            eb,
            shared_phones={"9811122233"},
            pair_label="rec_a-rec_b",
        )
        self.assertEqual(tr.veto_code, "sex_conflict")
        phone_c = next(c for c in tr.contributions if c.feature == "phone")
        self.assertEqual(phone_c.level, "missing")
        self.assertEqual(phone_c.tf_adjusted_bits, 0.0)

        mt = from_pair_trace(tr)
        self.assertEqual(mt.total_bits, tr.total_bits)
        self.assertEqual(mt.decision, "VETO:sex_conflict")
        rendered = render_waterfall(mt, forbidden_strings=["Arif", "9811122233"])
        self.assertIn("VETO:sex_conflict", rendered)


if __name__ == "__main__":
    unittest.main()
