"""Match-weight waterfall formatter for linkage diagnostics (Plan C V2).

Formats per-feature log2 Bayes-factor contributions, prior, total, threshold,
and veto/abstention codes in a deterministic, text-free representation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


class WaterfallLeakError(ValueError):
    """Raised when a forbidden sensitive value is detected in a waterfall trace."""


@dataclass(frozen=True)
class FeatureContribution:
    """Single feature's comparison level and log2 Bayes-factor weight."""

    feature: str
    level: str
    base_weight: float
    tf_adjusted_weight: float


@dataclass(frozen=True)
class MatchTrace:
    """Structured match trace supplied by a scorer (no raw PII or paths)."""

    pair_label: str
    prior_bits: float
    contributions: tuple[FeatureContribution, ...]
    threshold_bits: float
    veto_code: str | None = None
    abstention_code: str | None = None

    @property
    def total_bits(self) -> float:
        return round(
            self.prior_bits + sum(c.tf_adjusted_weight for c in self.contributions),
            6,
        )

    @property
    def decision(self) -> str:
        if self.veto_code:
            return f"VETO:{self.veto_code}"
        if self.abstention_code:
            return f"ABSTAIN:{self.abstention_code}"
        return "MATCH" if self.total_bits >= self.threshold_bits else "NO_MATCH"


def from_pair_trace(
    trace: object,
    *,
    abstention_code: str | None = None,
) -> MatchTrace:
    """Convert a runtime ``PairTrace`` into a dev ``MatchTrace`` without importing runtime internals."""
    contribs = tuple(
        FeatureContribution(
            feature=c.feature,
            level=c.level,
            base_weight=c.base_bits,
            tf_adjusted_weight=c.tf_adjusted_bits,
        )
        for c in trace.contributions
    )
    return MatchTrace(
        pair_label=trace.pair_label,
        prior_bits=trace.prior_bits,
        contributions=contribs,
        threshold_bits=trace.tau_high_bits,
        veto_code=trace.veto_code,
        abstention_code=abstention_code if abstention_code is not None else trace.abstention_code,
    )


def render_waterfall(
    trace: MatchTrace,
    *,
    forbidden_strings: Iterable[str] = (),
) -> str:
    """Render ``trace`` deterministically and guard against sensitive leaks."""
    lines = [
        f"pair={trace.pair_label} decision={trace.decision} "
        f"prior={trace.prior_bits:+.3f} total={trace.total_bits:+.3f} "
        f"threshold={trace.threshold_bits:+.3f}"
    ]
    if trace.veto_code:
        lines.append(f"veto={trace.veto_code}")
    if trace.abstention_code:
        lines.append(f"abstention={trace.abstention_code}")

    for contrib in trace.contributions:
        lines.append(
            f"  {contrib.feature:<20} level={contrib.level:<16} "
            f"base={contrib.base_weight:+.3f} tf_adj={contrib.tf_adjusted_weight:+.3f}"
        )

    rendered = "\n".join(lines)
    folded = rendered.casefold()
    for secret in forbidden_strings:
        if secret and secret.casefold() in folded:
            raise WaterfallLeakError("waterfall_sensitive_value_detected")
    return rendered
