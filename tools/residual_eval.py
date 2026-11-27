"""Dev-only E11 residual gate ablation harness (Plan C V8 Phase 5).

Evaluates Layer A (``kavachmed/residual_a.py``) and optional Layer B hook across
four configurations (``off``, ``a_only``, ``b_only``, ``both``) on synthetic
fixtures without printing matched PII text.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

from kavachmed.residual_a import ExemptSpan, Hit, LeakKeys, scan_with_status

LayerBScanner = Callable[[str, str], Iterable[Hit]]


@dataclass(frozen=True)
class E11Fixture:
    out_path: str
    text: str
    planted_harvested: int = 0
    planted_unseen: int = 0
    is_clean_control: bool = False
    exempt_index: tuple[ExemptSpan, ...] = ()
    structured_surfaces: Mapping[str, tuple[str, ...]] | None = None


@dataclass(frozen=True)
class E11ConfigResult:
    mode: str
    files_evaluated: int
    harvested_survivors: int
    unseen_survivors: int
    repair_attempts: int
    good_file_quarantines: int
    over_redaction_controls: int
    scan_seconds: float


def run_e11_ablation(
    fixtures: Iterable[E11Fixture],
    keys: LeakKeys,
    *,
    layer_b_scanner: LayerBScanner | None = None,
    max_passes: int = 3,
) -> dict[str, E11ConfigResult]:
    """Run the four E11 configurations (off, a_only, b_only, both) and return aggregate counts."""
    fixture_list = list(fixtures)
    results: dict[str, E11ConfigResult] = {}

    for mode, use_a, use_b in (
        ("off", False, False),
        ("a_only", True, False),
        ("b_only", False, True),
        ("both", True, True),
    ):
        t0 = time.perf_counter()
        harvested_surv = 0
        unseen_surv = 0
        repairs = 0
        good_quarantines = 0
        over_redacts = 0

        for fx in fixture_list:
            current_text = fx.text
            quarantined = False
            for attempt in range(max_passes + 1):
                hits: list[Hit] = []
                verified = True
                if use_a:
                    out_a = scan_with_status(
                        current_text,
                        fx.out_path,
                        keys,
                        exempt_index=fx.exempt_index,
                        structured_surfaces=fx.structured_surfaces,  # type: ignore[arg-type]
                    )
                    hits.extend(out_a.hits)
                    verified = verified and out_a.verified
                if use_b and layer_b_scanner is not None:
                    hits.extend(layer_b_scanner(current_text, fx.out_path))

                if verified and not hits:
                    break
                if attempt == max_passes or not verified:
                    quarantined = True
                    break
                repairs += 1
                if fx.is_clean_control:
                    over_redacts += 1
                chars = list(current_text)
                for h in sorted(hits, key=lambda x: x.start, reverse=True):
                    if h.channel == "content" and 0 <= h.start < h.end <= len(chars):
                        chars[h.start : h.end] = list("[REDACTED]")
                current_text = "".join(chars)

            if quarantined and fx.is_clean_control:
                good_quarantines += 1
            if not quarantined:
                if fx.planted_harvested > 0:
                    check_a = scan_with_status(
                        current_text,
                        fx.out_path,
                        keys,
                        exempt_index=fx.exempt_index,
                        structured_surfaces=fx.structured_surfaces,  # type: ignore[arg-type]
                    )
                    if check_a.hits:
                        harvested_surv += len(check_a.hits)
                if fx.planted_unseen > 0 and layer_b_scanner is not None:
                    b_hits = list(layer_b_scanner(current_text, fx.out_path))
                    if b_hits:
                        unseen_surv += len(b_hits)

        elapsed = time.perf_counter() - t0
        results[mode] = E11ConfigResult(
            mode=mode,
            files_evaluated=len(fixture_list),
            harvested_survivors=harvested_surv,
            unseen_survivors=unseen_surv,
            repair_attempts=repairs,
            good_file_quarantines=good_quarantines,
            over_redaction_controls=over_redacts,
            scan_seconds=round(elapsed, 6),
        )

    return results
