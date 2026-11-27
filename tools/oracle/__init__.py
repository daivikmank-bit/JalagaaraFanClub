"""Dev-only truth oracle package (Plan C V2).

Never import this package from runtime modules under ``kavachmed/``.
"""
from __future__ import annotations

from .loader import (
    Oracle,
    OracleBoundaryError,
    OracleConfigError,
    OracleCoverage,
    is_oracle_enabled,
    load_oracle,
)

__all__ = [
    "Oracle",
    "OracleBoundaryError",
    "OracleConfigError",
    "OracleCoverage",
    "is_oracle_enabled",
    "load_oracle",
]
