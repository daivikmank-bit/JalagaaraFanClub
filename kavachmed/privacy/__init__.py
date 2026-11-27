"""Privacy layer entrypoint (D3, D4, D13, D14)."""
from __future__ import annotations

from typing import Any

from kavachmed.contracts import Evidence, ParsedFile, PrivacyPlan, Resolution
from kavachmed.privacy.controller import build_plan


def plan(
    res: Resolution,
    evidences: list[Evidence],
    parsed_all: list[ParsedFile],
    policy: dict[str, Any],
) -> PrivacyPlan:
    return build_plan(res, evidences, parsed_all, policy)
