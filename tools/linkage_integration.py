"""Dev-only V7 integration harness for real spans, oracle parity, and stage diagnostics.

Runs identical ``Record`` inputs through ``kavachmed.evidence.extract`` and
``kavachmed.linkage.resolve`` with two span providers (detector/release path
and dev-only oracle diagnostic path), records aggregate evidence coverage and
pair/B3 metrics, classifies missed or incorrect links by first failing stage,
and renders only text-free diagnostics and V5 waterfall traces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

from kavachmed.evidence import DateProvider, Evidence, LgdLookup, RecKey, Record, Span, extract
from kavachmed.linkage import KeysProtocol, Resolution, resolve
from kavachmed.linkage.blocking import generate_candidates
from kavachmed.linkage.filename import FilenameSignal, prepare_filename_evidence
from kavachmed.linkage.fs import PairTrace, score_pair
from kavachmed.linkage.policy import LinkagePolicy, parse_linkage_policy
from kavachmed.linkage.strict import build_strict_clusters
from tools.link_eval import B3Metrics, ParityMetrics, compute_b3_metrics, compute_parity_metrics
from tools.waterfall import from_pair_trace, render_waterfall

SpanProvider = Callable[[Record], list[Span]]

STAGE_MISSING_SPAN = "missing_or_mis_scoped_span"
STAGE_NORM_MISMATCH = "normalization_mismatch"
STAGE_NO_CANDIDATE = "no_candidate"
STAGE_VETO = "veto"
STAGE_BELOW_THRESHOLD = "score_below_threshold"
STAGE_AMBIGUOUS_CLUSTER = "ambiguous_cluster"
STAGE_REJECTED_FILENAME = "rejected_filename_attribution"

VALID_FAILURE_STAGES: frozenset[str] = frozenset(
    {
        STAGE_MISSING_SPAN,
        STAGE_NORM_MISMATCH,
        STAGE_NO_CANDIDATE,
        STAGE_VETO,
        STAGE_BELOW_THRESHOLD,
        STAGE_AMBIGUOUS_CLUSTER,
        STAGE_REJECTED_FILENAME,
    }
)


@dataclass(frozen=True)
class EvidenceCoverage:
    """Text-free aggregate evidence coverage counts."""

    records_total: int = 0
    with_names: int = 0
    with_ids: int = 0
    with_phones: int = 0
    with_birth_years: int = 0
    with_geo: int = 0
    with_dates: int = 0


@dataclass
class IntegrationRunResult:
    """Aggregate result of one integration run (detector or oracle)."""

    label: str
    evidences: list[Evidence]
    resolution: Resolution
    coverage: EvidenceCoverage
    diagnostics: dict[str, int] = field(default_factory=dict)
    parity: ParityMetrics | None = None
    b3: B3Metrics | None = None
    stage_counts: dict[str, int] = field(default_factory=dict)
    waterfall_samples: list[str] = field(default_factory=list)


def compute_evidence_coverage(evidences: list[Evidence]) -> EvidenceCoverage:
    """Compute aggregate text-free evidence coverage across records."""
    return EvidenceCoverage(
        records_total=len(evidences),
        with_names=sum(1 for e in evidences if e.names),
        with_ids=sum(1 for e in evidences if e.ids),
        with_phones=sum(1 for e in evidences if e.phones),
        with_birth_years=sum(1 for e in evidences if e.birth_years is not None),
        with_geo=sum(1 for e in evidences if e.geo),
        with_dates=sum(1 for e in evidences if e.dates),
    )


def _classify_pair_failure(
    rk_a: RecKey,
    rk_b: RecKey,
    *,
    by_key: dict[RecKey, Evidence],
    oracle_by_key: dict[RecKey, Evidence] | None,
    candidate_set: set[tuple[RecKey, RecKey]],
    shared_phones: set[str],
    attached_fn: Mapping[RecKey, Any],
    policy: LinkagePolicy,
    is_false_merge: bool,
) -> tuple[str, PairTrace]:
    ev_a = by_key[rk_a]
    ev_b = by_key[rk_b]
    tr = score_pair(ev_a, ev_b, policy=policy, shared_phones=shared_phones)

    if is_false_merge:
        if rk_a in attached_fn or rk_b in attached_fn:
            return STAGE_REJECTED_FILENAME, tr
        return STAGE_AMBIGUOUS_CLUSTER, tr

    # Missed true link: check earliest stage explaining why it did not merge.
    if oracle_by_key is not None:
        oa = oracle_by_key.get(rk_a)
        ob = oracle_by_key.get(rk_b)
        if oa is not None and ob is not None:
            if (oa.names and not ev_a.names) or (ob.names and not ev_b.names):
                return STAGE_MISSING_SPAN, tr
            if (oa.ids and not ev_a.ids) or (ob.ids and not ev_b.ids):
                return STAGE_MISSING_SPAN, tr
            if (oa.phones and not ev_a.phones) or (ob.phones and not ev_b.phones):
                return STAGE_MISSING_SPAN, tr
            if (oa.names and ev_a.names and {t.skeleton for t in oa.names} != {t.skeleton for t in ev_a.names}) or (
                ob.names and ev_b.names and {t.skeleton for t in ob.names} != {t.skeleton for t in ev_b.names}
            ):
                return STAGE_NORM_MISMATCH, tr

    if rk_a in attached_fn or rk_b in attached_fn:
        return STAGE_REJECTED_FILENAME, tr

    pair = (rk_a, rk_b) if rk_a < rk_b else (rk_b, rk_a)
    if pair not in candidate_set:
        return STAGE_NO_CANDIDATE, tr

    if tr.veto_code is not None:
        return STAGE_VETO, tr

    if tr.total_bits < policy.tau_high_bits:
        return STAGE_BELOW_THRESHOLD, tr

    return STAGE_AMBIGUOUS_CLUSTER, tr


def run_linkage_pipeline(
    records: list[Record],
    span_provider: SpanProvider,
    keys: KeysProtocol,
    *,
    label: str = "detector",
    policy: Mapping[str, Any] | LinkagePolicy | Any = None,
    filename_signals: Iterable[FilenameSignal] | None = None,
    file_digests: Mapping[str, str] | None = None,
    date_provider: DateProvider | None = None,
    lgd_lookup: LgdLookup | None = None,
    truth_by_key: Mapping[RecKey, str | None] | None = None,
    oracle_evidences: list[Evidence] | None = None,
    max_waterfalls: int = 3,
) -> IntegrationRunResult:
    """Run extract -> resolve using ``span_provider`` and optionally score vs truth."""
    cfg = parse_linkage_policy(policy)
    evidences: list[Evidence] = []
    extract_diag: dict[str, int] = {}
    for rec in records:
        spans = span_provider(rec)
        ev = extract(
            rec,
            spans,
            date_provider=date_provider,
            lgd_lookup=lgd_lookup,
            diagnostics=extract_diag,
        )
        evidences.append(ev)

    diag: dict[str, int] = dict(extract_diag)
    res = resolve(
        evidences,
        keys,
        policy=cfg,
        file_digests=file_digests,
        filename_signals=filename_signals,
        diagnostics=diag,
    )
    cov = compute_evidence_coverage(evidences)

    parity: ParityMetrics | None = None
    b3: B3Metrics | None = None
    stage_counts: dict[str, int] = {}
    waterfalls: list[str] = []

    if truth_by_key is not None:
        units: list[tuple[str, str | None]] = []
        ordered_keys = sorted(truth_by_key)
        for rk in ordered_keys:
            t_label = truth_by_key[rk]
            if t_label is not None:
                units.append((t_label, res.patient_of.get(rk)))
        parity = compute_parity_metrics(units)
        b3 = compute_b3_metrics(units)

        by_key = {e.key: e for e in evidences}
        oracle_by_key = {e.key: e for e in oracle_evidences} if oracle_evidences else None
        strict_res = build_strict_clusters(evidences, policy=cfg)
        cand_res = generate_candidates(
            evidences,
            policy=cfg,
            contested_bridges=strict_res.contested_bridges,
        )
        cand_set = set(cand_res.candidates)
        shared_phones = set(strict_res.shared_phones)
        attached_fn = prepare_filename_evidence(filename_signals, evidences, policy=cfg)

        for i in range(len(ordered_keys)):
            for j in range(i + 1, len(ordered_keys)):
                ra, rb = ordered_keys[i], ordered_keys[j]
                ta, tb = truth_by_key[ra], truth_by_key[rb]
                if ta is None or tb is None:
                    continue
                same_truth = ta == tb
                pa, pb = res.patient_of.get(ra), res.patient_of.get(rb)
                same_pred = pa is not None and pa == pb
                if same_truth == same_pred:
                    continue
                stage, trace = _classify_pair_failure(
                    ra,
                    rb,
                    by_key=by_key,
                    oracle_by_key=oracle_by_key,
                    candidate_set=cand_set,
                    shared_phones=shared_phones,
                    attached_fn=attached_fn,
                    policy=cfg,
                    is_false_merge=same_pred and not same_truth,
                )
                stage_counts[stage] = stage_counts.get(stage, 0) + 1
                if len(waterfalls) < max_waterfalls:
                    abstain = stage if not same_pred else None
                    mt = from_pair_trace(trace, abstention_code=abstain)
                    waterfalls.append(render_waterfall(mt))

    return IntegrationRunResult(
        label=label,
        evidences=evidences,
        resolution=res,
        coverage=cov,
        diagnostics=diag,
        parity=parity,
        b3=b3,
        stage_counts=stage_counts,
        waterfall_samples=waterfalls,
    )


def compare_detector_and_oracle(
    records: list[Record],
    detector_spans: SpanProvider,
    oracle_spans: SpanProvider,
    keys: KeysProtocol,
    *,
    policy: Mapping[str, Any] | LinkagePolicy | Any = None,
    filename_signals: Iterable[FilenameSignal] | None = None,
    file_digests: Mapping[str, str] | None = None,
    date_provider: DateProvider | None = None,
    lgd_lookup: LgdLookup | None = None,
    truth_by_key: Mapping[RecKey, str | None] | None = None,
) -> tuple[IntegrationRunResult, IntegrationRunResult]:
    """Run paired diagnostic (oracle) and release-path (detector) evaluations."""
    oracle_run = run_linkage_pipeline(
        records,
        oracle_spans,
        keys,
        label="oracle",
        policy=policy,
        filename_signals=filename_signals,
        file_digests=file_digests,
        date_provider=date_provider,
        lgd_lookup=lgd_lookup,
        truth_by_key=truth_by_key,
    )
    detector_run = run_linkage_pipeline(
        records,
        detector_spans,
        keys,
        label="detector",
        policy=policy,
        filename_signals=filename_signals,
        file_digests=file_digests,
        date_provider=date_provider,
        lgd_lookup=lgd_lookup,
        truth_by_key=truth_by_key,
        oracle_evidences=oracle_run.evidences,
    )
    return detector_run, oracle_run


def render_integration_summary(result: IntegrationRunResult) -> str:
    """Render a strictly text-free aggregate summary for one integration run."""
    lines = [
        f"run={result.label} records={result.coverage.records_total} "
        f"names={result.coverage.with_names} ids={result.coverage.with_ids} "
        f"phones={result.coverage.with_phones} birth_years={result.coverage.with_birth_years} "
        f"geo={result.coverage.with_geo} dates={result.coverage.with_dates}",
    ]
    if result.parity is not None and result.b3 is not None:
        lines.append(
            f"parity tp={result.parity.tp} fp={result.parity.fp} fn={result.parity.fn} "
            f"P={result.parity.precision:.4f} R={result.parity.recall:.4f} "
            f"F0.5={result.parity.f05:.4f} B3_F1={result.b3.f1:.4f}"
        )
    if result.stage_counts:
        stage_str = " ".join(f"{k}={v}" for k, v in sorted(result.stage_counts.items()))
        lines.append(f"stages {stage_str}")
    return "\n".join(lines)
