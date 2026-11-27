"""Linkage policy parsing and validation (Plan C V5)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

_LEGACY_PROB_KEYS: frozenset[str] = frozenset(
    {
        "match_threshold",
        "no_match_threshold",
        "prob_threshold",
        "probability_threshold",
        "tau_high",
        "tau_low",
        "min_prob",
        "min_probability",
        "linkage_threshold",
        "fs_threshold",
        "threshold",
    }
)

DEFAULT_FS_DISTRIBUTIONS: dict[str, dict[str, tuple[float, float]]] = {
    "name": {
        "exact": (0.52, 0.015),
        "skeleton": (0.23, 0.025),
        "jaro_winkler": (0.12, 0.035),
        "initial": (0.08, 0.075),
        "disagreement": (0.05, 0.85),
    },
    "relative": {
        "agrees": (0.85, 0.05),
        "disagrees": (0.15, 0.95),
    },
    "birth_interval": {
        "overlaps": (0.75, 0.10),
        "near": (0.22, 0.15),
        "disagrees": (0.03, 0.75),
    },
    "sex": {
        "agrees": (0.99, 0.50),
        "disagrees": (0.01, 0.50),
    },
    "geography": {
        "same_district": (0.70, 0.20),
        "different_district": (0.30, 0.80),
    },
    "phone": {
        "exact_non_shared": (0.65, 0.02),
        "different": (0.35, 0.98),
    },
    "hospital": {
        "agrees": (0.55, 0.40),
        "disagrees": (0.45, 0.60),
    },
}


@dataclass(frozen=True)
class LinkagePolicy:
    """Validated configuration for V4/V5 linkage."""

    mode: str = "scored"  # "scored" | "exact_only"
    family_mode: str = "single_graph"  # "single_graph" | "temporal_family"
    prior_bits: float = -4.0
    tau_high_bits: float = 4.0
    tau_low_bits: float = 1.0
    max_component_size: int = 200
    max_pair_checks: int = 2000
    max_bucket_size: int = 40
    pair_budget_factor: int = 30
    pair_budget: int | None = None
    tf_enabled: bool = True
    tf_total_rows: int | None = None
    tf_frequencies: Mapping[str, int] | None = None
    disabled_id_namespaces: tuple[str, ...] = ()
    allowed_id_namespaces: tuple[str, ...] = ()
    year_reused_uhid_issuers: tuple[str, ...] = ()
    fs_distributions: Mapping[str, Mapping[str, tuple[float, float]]] | None = None


def _extract_linkage_mapping(policy: Mapping[str, Any] | Any) -> dict[str, Any]:
    if policy is None:
        return {}
    if isinstance(policy, LinkagePolicy):
        return {
            "mode": policy.mode,
            "family_mode": policy.family_mode,
            "prior_bits": policy.prior_bits,
            "tau_high_bits": policy.tau_high_bits,
            "tau_low_bits": policy.tau_low_bits,
            "max_component_size": policy.max_component_size,
            "max_pair_checks": policy.max_pair_checks,
            "max_bucket_size": policy.max_bucket_size,
            "pair_budget_factor": policy.pair_budget_factor,
            "pair_budget": policy.pair_budget,
            "tf_enabled": policy.tf_enabled,
            "tf_total_rows": policy.tf_total_rows,
            "tf_frequencies": policy.tf_frequencies,
            "disabled_id_namespaces": policy.disabled_id_namespaces,
            "allowed_id_namespaces": policy.allowed_id_namespaces,
            "year_reused_uhid_issuers": policy.year_reused_uhid_issuers,
            "fs_distributions": policy.fs_distributions,
        }
    merged: dict[str, Any] = {}
    if isinstance(policy, Mapping):
        linkage_sub = policy.get("linkage")
        if isinstance(linkage_sub, Mapping):
            merged.update(linkage_sub)
        for k, v in policy.items():
            if k != "linkage":
                merged[k] = v
        return merged

    linkage_sub = getattr(policy, "linkage", None)
    if linkage_sub is not None:
        if isinstance(linkage_sub, Mapping):
            merged.update(linkage_sub)
        else:
            for attr in dir(linkage_sub):
                if not attr.startswith("_"):
                    merged[attr] = getattr(linkage_sub, attr)
    for attr in dir(policy):
        if not attr.startswith("_") and attr != "linkage":
            val = getattr(policy, attr)
            if not callable(val):
                merged[attr] = val
    return merged


def _require_finite_float(val: Any, name: str) -> float:
    from kavachmed.linkage import LinkageError

    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise LinkageError("invalid_linkage_policy", f"{name} must be numeric")
    fval = float(val)
    if not math.isfinite(fval):
        raise LinkageError("invalid_linkage_policy", f"{name} must be finite")
    return fval


def _require_positive_int(val: Any, name: str) -> int:
    from kavachmed.linkage import LinkageError

    if isinstance(val, bool) or not isinstance(val, int) or val <= 0:
        raise LinkageError("invalid_linkage_policy", f"{name} must be a positive integer")
    return val


def validate_distributions(
    dists: Mapping[str, Mapping[str, tuple[float, float]]],
) -> None:
    """Validate that every feature's m and u probabilities are in (0, 1) and sum to 1."""
    from kavachmed.linkage import LinkageError

    for feat, levels in dists.items():
        if not levels:
            raise LinkageError("invalid_linkage_policy", f"empty distribution for {feat}")
        m_sum = 0.0
        u_sum = 0.0
        for lvl, pair in levels.items():
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise LinkageError("invalid_linkage_policy", f"invalid (m, u) for {feat}.{lvl}")
            m = _require_finite_float(pair[0], f"{feat}.{lvl}.m")
            u = _require_finite_float(pair[1], f"{feat}.{lvl}.u")
            if not (0.0 < m < 1.0) or not (0.0 < u < 1.0):
                raise LinkageError("invalid_linkage_policy", f"m/u out of (0,1) for {feat}.{lvl}")
            m_sum += m
            u_sum += u
        if abs(m_sum - 1.0) > 1e-6 or abs(u_sum - 1.0) > 1e-6:
            raise LinkageError(
                "invalid_linkage_policy",
                f"m/u must sum to 1.0 for {feat} (got m={m_sum:.6f}, u={u_sum:.6f})",
            )


def parse_linkage_policy(policy: Mapping[str, Any] | Any = None) -> LinkagePolicy:
    """Parse and validate linkage policy settings before scoring."""
    from kavachmed.linkage import LinkageError

    if isinstance(policy, LinkagePolicy):
        return policy

    raw = _extract_linkage_mapping(policy)

    # Reject legacy probability-looking keys with migration guidance.
    for key in sorted(raw):
        if key in _LEGACY_PROB_KEYS:
            raise LinkageError(
                "legacy_probability_policy_key",
                f"Legacy key '{key}' is unsupported; migrate to 'tau_high_bits' and 'tau_low_bits' in linkage policy.",
            )

    has_high = "tau_high_bits" in raw
    has_low = "tau_low_bits" in raw
    if has_high != has_low:
        raise LinkageError(
            "invalid_linkage_policy",
            "Specify both 'tau_high_bits' and 'tau_low_bits' together.",
        )

    mode = str(raw.get("mode", "exact_only" if raw.get("exact_only") else "scored"))
    if mode not in {"scored", "exact_only"}:
        raise LinkageError("invalid_linkage_policy", f"Unsupported linkage mode '{mode}'.")

    family_mode = str(raw.get("family_mode", "single_graph"))
    if family_mode not in {"single_graph", "temporal_family"}:
        raise LinkageError(
            "invalid_linkage_policy",
            f"Unsupported linkage family_mode '{family_mode}'.",
        )

    prior_bits = _require_finite_float(raw.get("prior_bits", -4.0), "prior_bits")
    tau_high_bits = _require_finite_float(raw.get("tau_high_bits", 4.0), "tau_high_bits")
    tau_low_bits = _require_finite_float(raw.get("tau_low_bits", 1.0), "tau_low_bits")
    if tau_low_bits >= tau_high_bits:
        raise LinkageError(
            "invalid_linkage_policy",
            "Require tau_low_bits < tau_high_bits.",
        )

    max_component_size = _require_positive_int(
        raw.get("max_component_size", 200), "max_component_size"
    )
    max_pair_checks = _require_positive_int(
        raw.get("max_pair_checks", 2000), "max_pair_checks"
    )
    max_bucket_size = _require_positive_int(
        raw.get("max_bucket_size", 40), "max_bucket_size"
    )
    pair_budget_factor = _require_positive_int(
        raw.get("pair_budget_factor", 30), "pair_budget_factor"
    )

    pair_budget = raw.get("pair_budget")
    if pair_budget is not None:
        pair_budget = _require_positive_int(pair_budget, "pair_budget")

    tf_total_rows = raw.get("tf_total_rows")
    if tf_total_rows is not None:
        tf_total_rows = _require_positive_int(tf_total_rows, "tf_total_rows")

    tf_enabled = bool(raw.get("tf_enabled", True))
    tf_frequencies = raw.get("tf_frequencies")

    fs_distributions = raw.get("fs_distributions")
    if fs_distributions is not None:
        validate_distributions(fs_distributions)

    return LinkagePolicy(
        mode=mode,
        family_mode=family_mode,
        prior_bits=prior_bits,
        tau_high_bits=tau_high_bits,
        tau_low_bits=tau_low_bits,
        max_component_size=max_component_size,
        max_pair_checks=max_pair_checks,
        max_bucket_size=max_bucket_size,
        pair_budget_factor=pair_budget_factor,
        pair_budget=pair_budget,
        tf_enabled=tf_enabled,
        tf_total_rows=tf_total_rows,
        tf_frequencies=tf_frequencies,
        disabled_id_namespaces=tuple(raw.get("disabled_id_namespaces", ())),
        allowed_id_namespaces=tuple(raw.get("allowed_id_namespaces", ())),
        year_reused_uhid_issuers=tuple(raw.get("year_reused_uhid_issuers", ())),
        fs_distributions=fs_distributions,
    )
