import unittest

from tools.waterfall import (
    FeatureContribution,
    MatchTrace,
    WaterfallLeakError,
    render_waterfall,
)


class TestWaterfall(unittest.TestCase):
    def test_feature_order_arithmetic_and_decisions(self) -> None:
        trace = MatchTrace(
            pair_label="rec_001<->rec_002",
            prior_bits=-2.0,
            contributions=(
                FeatureContribution("name_skeleton", "exact", 4.0, 3.0),
                FeatureContribution("sex", "agree", 1.0, 1.0),
            ),
            threshold_bits=1.5,
        )
        self.assertAlmostEqual(trace.total_bits, 2.0)
        self.assertEqual(trace.decision, "MATCH")

        text = render_waterfall(trace, forbidden_strings=["Mahesh Chauhan", "dev/input"])
        self.assertIn("prior=-2.000", text)
        self.assertIn("total=+2.000", text)
        self.assertLess(text.index("name_skeleton"), text.index("sex"))

    def test_veto_abstention_and_sensitive_guard(self) -> None:
        veto_trace = MatchTrace(
            pair_label="rec_001<->rec_003",
            prior_bits=-2.0,
            contributions=(FeatureContribution("age", "conflict_gt3y", -5.0, -5.0),),
            threshold_bits=1.5,
            veto_code="AGE_CONFLICT",
        )
        self.assertEqual(veto_trace.decision, "VETO:AGE_CONFLICT")
        self.assertIn("veto=AGE_CONFLICT", render_waterfall(veto_trace))

        abstain_trace = MatchTrace(
            pair_label="rec_004<->rec_005",
            prior_bits=-2.0,
            contributions=(),
            threshold_bits=1.5,
            abstention_code="AMBIGUOUS_CANDIDATES",
        )
        self.assertEqual(abstain_trace.decision, "ABSTAIN:AMBIGUOUS_CANDIDATES")
        self.assertIn("abstention=AMBIGUOUS_CANDIDATES", render_waterfall(abstain_trace))

        with self.assertRaises(WaterfallLeakError):
            render_waterfall(abstain_trace, forbidden_strings=["rec_004"])


if __name__ == "__main__":
    unittest.main()
