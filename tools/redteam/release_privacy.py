"""Dev-only Plan C V10 (C15) verification for A6 pseudonym inversion and A7 key rotation.

Implements the synthetic fixture alignment, A6 positive/negative control checks,
same-secret determinism verification, and A7 cross-release linkability measurement
specified in ``plans/plan-C/c-v10.md`` and ``docs/ATTACKS.md``.

All outputs are strictly aggregate and text-free: secrets, raw identifiers,
sensitive file paths, and per-patient rows are never emitted in summaries.
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import hmac
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from kavachmed.evidence import Evidence, Field, Record, Span, extract
from kavachmed.linkage import resolve
from kavachmed.linkage.pseudo import CROCKFORD_ALPHABET

_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_DMY_DATE_RE = re.compile(r"\b(\d{2})[-/](\d{2})[-/](\d{4})\b")


class RedteamFixtureError(ValueError):
    """Raised when a synthetic red-team fixture has duplicate or invalid keys."""


@dataclass(frozen=True)
class SyntheticEvent:
    """One dated clinical event in a synthetic fixture."""

    event_id: str
    patient_id: str
    input_file: str
    loc: str
    date_field: str
    iso_date: str


@dataclass(frozen=True)
class SyntheticRecordSpec:
    """One input record in a synthetic fixture."""

    record_id: str
    patient_id: str | None  # None for unlinked control records
    input_file: str
    loc: str
    format: str
    fields: Mapping[str, Any]
    text: str = ""
    spans: tuple[Span, ...] = ()
    dates: tuple[str, ...] = ()
    events: tuple[SyntheticEvent, ...] = ()


@dataclass(frozen=True)
class SyntheticPatient:
    """One synthetic patient with opaque fixture ID and A6 original identifiers."""

    patient_id: str
    uhids: tuple[str, ...] = ()
    aadhaar: tuple[str, ...] = ()
    phones: tuple[str, ...] = ()
    abhas: tuple[str, ...] = ()
    latin_names: tuple[str, ...] = ()

    def all_originals(self) -> tuple[str, ...]:
        """Return all non-empty original identifiers in deterministic order."""
        items: list[str] = []
        for group in (self.uhids, self.aadhaar, self.phones, self.abhas, self.latin_names):
            for val in group:
                stripped = val.strip()
                if stripped:
                    items.append(stripped)
        return tuple(items)


@dataclass(frozen=True)
class SyntheticCohort:
    """Frozen hand-authored synthetic cohort with opaque IDs and locator mappings."""

    patients: tuple[SyntheticPatient, ...]
    records: tuple[SyntheticRecordSpec, ...]

    def __post_init__(self) -> None:
        seen_patients: set[str] = set()
        for p in self.patients:
            if not p.patient_id:
                raise RedteamFixtureError("empty_patient_id")
            if p.patient_id in seen_patients:
                raise RedteamFixtureError(f"duplicate_patient_id:{p.patient_id}")
            seen_patients.add(p.patient_id)

        seen_records: set[str] = set()
        seen_locators: set[tuple[str, str]] = set()
        seen_events: set[str] = set()
        for r in self.records:
            if not r.record_id:
                raise RedteamFixtureError("empty_record_id")
            if r.record_id in seen_records:
                raise RedteamFixtureError(f"duplicate_record_id:{r.record_id}")
            seen_records.add(r.record_id)

            if r.patient_id is not None and r.patient_id not in seen_patients:
                raise RedteamFixtureError(f"unknown_patient_id:{r.patient_id}")

            loc_key = (r.input_file, r.loc)
            if loc_key in seen_locators:
                raise RedteamFixtureError(f"duplicate_input_locator:{r.input_file}#{r.loc}")
            seen_locators.add(loc_key)

            for ev in r.events:
                if not ev.event_id:
                    raise RedteamFixtureError("empty_event_id")
                if ev.event_id in seen_events:
                    raise RedteamFixtureError(f"duplicate_event_id:{ev.event_id}")
                if ev.patient_id != r.patient_id:
                    raise RedteamFixtureError(f"event_patient_mismatch:{ev.event_id}")
                if (ev.input_file, ev.loc) != loc_key:
                    raise RedteamFixtureError(f"event_locator_mismatch:{ev.event_id}")
                seen_events.add(ev.event_id)

    def patient_map(self) -> dict[str, SyntheticPatient]:
        return {p.patient_id: p for p in self.patients}

    def all_events(self) -> tuple[SyntheticEvent, ...]:
        events: list[SyntheticEvent] = []
        for r in self.records:
            events.extend(r.events)
        return tuple(events)


@dataclass(frozen=True)
class FrozenReleaseConfig:
    """Frozen linkage and release parameters for V10 verification."""

    linkage_mode: str = "scored"
    family_mode: str = "temporal_family"
    tau_high_bits: float = 12.0
    tau_low_bits: float = 6.0
    notes: str = "frozen_v10_config"

    def to_policy_dict(self) -> dict[str, Any]:
        return {
            "linkage": {
                "mode": self.linkage_mode,
                "family_mode": self.family_mode,
                "tau_high_bits": self.tau_high_bits,
                "tau_low_bits": self.tau_low_bits,
            }
        }


@dataclass(frozen=True)
class ReleaseRunRecord:
    """Observed output record from one release run (``records.jsonl`` + dates)."""

    out_file: str
    loc: str
    patient: str | None
    dates: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ReleaseRunArtifacts:
    """Materialized outputs of one release run in a fresh directory."""

    output_dir: Path
    records: tuple[ReleaseRunRecord, ...]
    # Maps input (file, loc) -> output (out_file, loc); supports file renames.
    source_to_output: Mapping[tuple[str, str], tuple[str, str]] = field(default_factory=dict)
    # Optional file-level rename map: input_rel_path -> output_rel_path.
    renamed_files: Mapping[str, str] = field(default_factory=dict)


ReleaseRunner = Callable[[SyntheticCohort, Path, str, FrozenReleaseConfig], ReleaseRunArtifacts]


@dataclass(frozen=True)
class A6ControlVector:
    """Hand-labeled A6 control vector for positive/negative validation."""

    vector_id: str
    patient_id: str
    candidate_pseudonym: str
    expected_unsafe: bool
    category: str


@dataclass(frozen=True)
class A6ControlSummary:
    """Aggregate A6 control verification counts (separate from release estimate)."""

    positive_total: int
    positive_flagged: int
    negative_total: int
    negative_passed: int

    @property
    def all_passed(self) -> bool:
        return (
            self.positive_flagged == self.positive_total
            and self.negative_passed == self.negative_total
        )


@dataclass(frozen=True)
class A6EvaluationResult:
    """Aggregate A6 pseudonym inversion evaluation on a release."""

    patients_evaluated: int
    patients_flagged: int
    rate: float | None
    unlinked_records_excluded: int
    missing_or_ambiguous_locators: int

    @property
    def available(self) -> bool:
        return self.rate is not None


@dataclass(frozen=True)
class DeterminismResult:
    """Result of comparing two same-secret runs (``run_a1`` vs ``run_a2``)."""

    passed: bool
    files_compared: int
    records_compared: int
    intervals_checked: int
    failure_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class CoverageSummary:
    """Alignment coverage across the three release runs."""

    expected_records: int
    aligned_records_a1: int
    aligned_records_b: int
    missing_or_ambiguous_records: int
    expected_events: int
    aligned_events_a1: int
    aligned_events_b: int
    missing_or_ambiguous_events: int

    @property
    def adequate(self) -> bool:
        return (
            self.missing_or_ambiguous_records == 0
            and self.missing_or_ambiguous_events == 0
            and self.aligned_records_a1 == self.expected_records
            and self.aligned_records_b == self.expected_records
        )


@dataclass(frozen=True)
class A7EvaluationResult:
    """Aggregate A7 cross-release linkability measurement."""

    psn_reused: int
    psn_comparable: int
    psn_rate: float | None
    date_reused: int
    date_comparable: int
    date_rate: float | None
    a7_rate: float | None
    dropped_or_unresolved: int
    systematic_date_reuse: bool
    coverage: CoverageSummary


@dataclass(frozen=True)
class V10VerificationSummary:
    """Top-level aggregate V10 verification summary (text-free, secret-free)."""

    config: FrozenReleaseConfig
    determinism: DeterminismResult
    a6_controls: A6ControlSummary
    a6_release: A6EvaluationResult
    a7_rotation: A7EvaluationResult
    real_runner_wired: bool = False
    pending_checks: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Date parsing and A6 transform helpers
# ---------------------------------------------------------------------------


def _parse_iso_or_dmy_date(value: str) -> dt.date | None:
    """Parse common ISO or DD-MM-YYYY / DD/MM/YYYY date strings into a ``date``."""
    s = value.strip()
    if not s:
        return None
    m_iso = _ISO_DATE_RE.search(s)
    if m_iso:
        try:
            return dt.date(int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3)))
        except ValueError:
            return None
    m_dmy = _DMY_DATE_RE.search(s)
    if m_dmy:
        try:
            return dt.date(int(m_dmy.group(3)), int(m_dmy.group(2)), int(m_dmy.group(1)))
        except ValueError:
            return None
    return None


def _expand_original_transforms(original: str) -> tuple[set[str], set[str]]:
    """Expand one original identifier into A6 exact/prefix forms and digit cores.

    Returns ``(exact_or_hash_forms, digit_cores)``:
    - ``exact_or_hash_forms``: lowercased strings/digests (full or >=8 prefixes)
      corresponding to case, digits-only, MD5/SHA-1/SHA-256/SHA-512 hex, base64,
      and reversal transforms from ``docs/ATTACKS.md`` A6.
    - ``digit_cores``: numeric digit cores (length >= 6) used to detect embedded
      UHID digit cores or phone numbers inside a pseudonym.
    """
    forms: set[str] = set()
    digit_cores: set[str] = set()

    raw = original.strip()
    if not raw:
        return forms, digit_cores

    digits_only = "".join(ch for ch in raw if ch.isdigit())
    variants = [raw, raw.lower(), raw.upper()]
    if digits_only:
        variants.append(digits_only)
        if len(digits_only) >= 6:
            digit_cores.add(digits_only)
        if len(digits_only) >= 10:
            digit_cores.add(digits_only[-10:])
            variants.append(digits_only[-10:])

    # Also extract trailing digit cores of length >= 5 (e.g., UHID core in SNJ/26/019672).
    for match in re.findall(r"\d{5,}", raw):
        digit_cores.add(match)

    seen_variants: list[str] = []
    seen_set: set[str] = set()
    for v in variants:
        if v and v not in seen_set:
            seen_set.add(v)
            seen_variants.append(v)

    for v in seen_variants:
        v_bytes = v.encode("utf-8")
        forms.add(v.lower())
        forms.add(v[::-1].lower())
        b64 = base64.b64encode(v_bytes).decode("ascii").rstrip("=").lower()
        if b64:
            forms.add(b64)
            forms.add(base64.b64encode(v_bytes).decode("ascii").lower())
        for algo in ("md5", "sha1", "sha256", "sha512"):
            digest_hex = hashlib.new(algo, v_bytes).hexdigest().lower()
            forms.add(digest_hex)

    return forms, digit_cores


def _unwrap_psn(pseudonym: str) -> str:
    """Strip ``PSN-`` prefix if present, returning the candidate payload."""
    s = pseudonym.strip()
    if s.upper().startswith("PSN-"):
        return s[4:]
    return s


def _is_pseudonym_inverted(pseudonym: str, patient: SyntheticPatient) -> bool:
    """Return True if ``pseudonym`` matches any A6 unsafe form for ``patient``."""
    raw_psn = pseudonym.strip()
    if not raw_psn:
        return False

    payload = _unwrap_psn(raw_psn)
    candidates = [raw_psn.lower(), payload.lower()]
    # If payload is a 9-char Crockford collision extension (8 + 1), do not strip
    # arbitrary characters unless checking exact/prefix on the 8-char core would match;
    # checking payload directly already catches prefixes >= 8 of hashes/transforms.
    for original in patient.all_originals():
        forms, digit_cores = _expand_original_transforms(original)
        for cand in candidates:
            if not cand:
                continue
            if cand in forms:
                return True
            if len(cand) >= 8:
                for form in forms:
                    if len(form) >= 8 and (form.startswith(cand) or cand.startswith(form)):
                        return True
            for core in digit_cores:
                if core in cand:
                    return True
    return False


# ---------------------------------------------------------------------------
# A6 evaluation (controls and release)
# ---------------------------------------------------------------------------


def evaluate_a6_controls(
    cohort: SyntheticCohort,
    controls: Sequence[A6ControlVector],
) -> A6ControlSummary:
    """Evaluate hand-labeled A6 positive and negative controls."""
    patients_by_id = cohort.patient_map()
    pos_total = 0
    pos_flagged = 0
    neg_total = 0
    neg_passed = 0

    for vec in controls:
        patient = patients_by_id.get(vec.patient_id)
        if patient is None:
            raise RedteamFixtureError(f"unknown_control_patient:{vec.patient_id}")
        flagged = _is_pseudonym_inverted(vec.candidate_pseudonym, patient)
        if vec.expected_unsafe:
            pos_total += 1
            if flagged:
                pos_flagged += 1
        else:
            neg_total += 1
            if not flagged:
                neg_passed += 1

    return A6ControlSummary(
        positive_total=pos_total,
        positive_flagged=pos_flagged,
        negative_total=neg_total,
        negative_passed=neg_passed,
    )


def _index_run_records(
    cohort: SyntheticCohort,
    artifacts: ReleaseRunArtifacts,
) -> tuple[dict[str, ReleaseRunRecord], int, int]:
    """Align input records in ``cohort`` to ``artifacts.records`` by locator.

    Returns ``(by_record_id, missing_or_ambiguous_count, unlinked_excluded_count)``.
    """
    by_out_loc: dict[tuple[str, str], list[ReleaseRunRecord]] = {}
    for rec in artifacts.records:
        by_out_loc.setdefault((rec.out_file, rec.loc), []).append(rec)

    aligned: dict[str, ReleaseRunRecord] = {}
    missing_or_ambiguous = 0
    unlinked_excluded = 0

    for spec in cohort.records:
        in_loc = (spec.input_file, spec.loc)
        if in_loc in artifacts.source_to_output:
            out_loc = artifacts.source_to_output[in_loc]
        elif spec.input_file in artifacts.renamed_files:
            out_loc = (artifacts.renamed_files[spec.input_file], spec.loc)
        else:
            out_loc = in_loc

        matches = by_out_loc.get(out_loc, [])
        if len(matches) != 1:
            missing_or_ambiguous += 1
            continue

        matched = matches[0]
        aligned[spec.record_id] = matched
        if spec.patient_id is None and matched.patient is None:
            unlinked_excluded += 1

    return aligned, missing_or_ambiguous, unlinked_excluded


def evaluate_a6_release(
    cohort: SyntheticCohort,
    artifacts: ReleaseRunArtifacts,
) -> A6EvaluationResult:
    """Measure A6 pseudonym inversion rate on a release run.

    Each patient with at least one non-null pseudonym is counted once.
    ``patient: null`` records stay outside the pseudonym denominator.
    """
    patients_by_id = cohort.patient_map()
    aligned, missing_or_ambiguous, unlinked_excluded = _index_run_records(cohort, artifacts)

    # Collect distinct non-null pseudonyms per patient_id in sorted order.
    psns_by_patient: dict[str, list[str]] = {}
    for spec in cohort.records:
        if spec.patient_id is None:
            continue
        rec = aligned.get(spec.record_id)
        if rec is None or rec.patient is None:
            continue
        psns_by_patient.setdefault(spec.patient_id, [])
        if rec.patient not in psns_by_patient[spec.patient_id]:
            psns_by_patient[spec.patient_id].append(rec.patient)

    evaluated = 0
    flagged = 0
    for pid in sorted(psns_by_patient):
        patient = patients_by_id[pid]
        evaluated += 1
        if any(_is_pseudonym_inverted(psn, patient) for psn in psns_by_patient[pid]):
            flagged += 1

    rate: float | None = (flagged / evaluated) if evaluated > 0 else None
    return A6EvaluationResult(
        patients_evaluated=evaluated,
        patients_flagged=flagged,
        rate=rate,
        unlinked_records_excluded=unlinked_excluded,
        missing_or_ambiguous_locators=missing_or_ambiguous,
    )


# ---------------------------------------------------------------------------
# Determinism & A7 rotation measurement
# ---------------------------------------------------------------------------


def _collect_sanitized_inventory(output_dir: Path) -> dict[str, bytes]:
    """Return sorted relative-path -> bytes mapping for ``output_dir/sanitized``."""
    sanitized_dir = output_dir / "sanitized"
    if not sanitized_dir.is_dir():
        return {}
    inventory: dict[str, bytes] = {}
    for path in sorted(sanitized_dir.rglob("*")):
        if path.is_file():
            rel = path.relative_to(sanitized_dir).as_posix()
            inventory[rel] = path.read_bytes()
    return inventory


def verify_same_secret_determinism(
    cohort: SyntheticCohort,
    run_a1: ReleaseRunArtifacts,
    run_a2: ReleaseRunArtifacts,
) -> DeterminismResult:
    """Verify byte-level determinism, PSN resolution, and stay-interval preservation."""
    reasons: list[str] = []

    inv_1 = _collect_sanitized_inventory(run_a1.output_dir)
    inv_2 = _collect_sanitized_inventory(run_a2.output_dir)
    if list(inv_1.keys()) != list(inv_2.keys()):
        reasons.append("sanitized_inventory_mismatch")
    else:
        for rel_path in sorted(inv_1):
            if inv_1[rel_path] != inv_2[rel_path]:
                reasons.append("sanitized_file_bytes_mismatch")
                break

    rec_path_1 = run_a1.output_dir / "records.jsonl"
    rec_path_2 = run_a2.output_dir / "records.jsonl"
    if not rec_path_1.is_file() or not rec_path_2.is_file():
        reasons.append("records_jsonl_missing")
    elif rec_path_1.read_bytes() != rec_path_2.read_bytes():
        reasons.append("records_jsonl_bytes_mismatch")

    aligned_1, miss_1, _ = _index_run_records(cohort, run_a1)
    aligned_2, miss_2, _ = _index_run_records(cohort, run_a2)
    if miss_1 > 0 or miss_2 > 0:
        reasons.append("locator_alignment_incomplete")

    records_compared = 0
    for spec in cohort.records:
        r1 = aligned_1.get(spec.record_id)
        r2 = aligned_2.get(spec.record_id)
        if r1 is None or r2 is None:
            continue
        records_compared += 1
        if r1.patient != r2.patient:
            reasons.append("pseudonym_mismatch_same_secret")
            break
        if spec.patient_id is None and r1.patient is not None:
            reasons.append("unlinked_control_linked")
            break

    # Verify timeline stay intervals are preserved after shifting within each patient.
    intervals_checked = 0
    events_by_patient: dict[str, list[SyntheticEvent]] = {}
    record_by_loc: dict[tuple[str, str], SyntheticRecordSpec] = {
        (s.input_file, s.loc): s for s in cohort.records
    }
    for ev in cohort.all_events():
        events_by_patient.setdefault(ev.patient_id, []).append(ev)

    for pid in sorted(events_by_patient):
        evs = events_by_patient[pid]
        if len(evs) < 2:
            continue
        anchor = evs[0]
        anchor_raw = _parse_iso_or_dmy_date(anchor.iso_date)
        anchor_spec = record_by_loc[(anchor.input_file, anchor.loc)]
        anchor_out_1 = aligned_1.get(anchor_spec.record_id)
        anchor_out_2 = aligned_2.get(anchor_spec.record_id)
        if anchor_raw is None or anchor_out_1 is None or anchor_out_2 is None:
            continue
        anchor_d1 = _parse_iso_or_dmy_date(anchor_out_1.dates.get(anchor.date_field, ""))
        anchor_d2 = _parse_iso_or_dmy_date(anchor_out_2.dates.get(anchor.date_field, ""))
        if anchor_d1 is None or anchor_d2 is None:
            continue

        for other in evs[1:]:
            other_raw = _parse_iso_or_dmy_date(other.iso_date)
            other_spec = record_by_loc[(other.input_file, other.loc)]
            other_out_1 = aligned_1.get(other_spec.record_id)
            other_out_2 = aligned_2.get(other_spec.record_id)
            if other_raw is None or other_out_1 is None or other_out_2 is None:
                continue
            other_d1 = _parse_iso_or_dmy_date(other_out_1.dates.get(other.date_field, ""))
            other_d2 = _parse_iso_or_dmy_date(other_out_2.dates.get(other.date_field, ""))
            if other_d1 is None or other_d2 is None:
                continue
            intervals_checked += 1
            raw_delta = (other_raw - anchor_raw).days
            if (other_d1 - anchor_d1).days != raw_delta or (other_d2 - anchor_d2).days != raw_delta:
                reasons.append("timeline_interval_not_preserved")
                break
        if "timeline_interval_not_preserved" in reasons:
            break

    return DeterminismResult(
        passed=len(reasons) == 0,
        files_compared=len(inv_1),
        records_compared=records_compared,
        intervals_checked=intervals_checked,
        failure_reasons=tuple(reasons),
    )


def evaluate_a7_rotation(
    cohort: SyntheticCohort,
    run_a1: ReleaseRunArtifacts,
    run_b: ReleaseRunArtifacts,
) -> A7EvaluationResult:
    """Measure A7 cross-release linkability between secret A and rotated secret B."""
    aligned_a, miss_rec_a, _ = _index_run_records(cohort, run_a1)
    aligned_b, miss_rec_b, _ = _index_run_records(cohort, run_b)

    # 1. Patient-level PSN retention (non-null in both runs).
    psns_a: dict[str, set[str]] = {}
    psns_b: dict[str, set[str]] = {}
    for spec in cohort.records:
        if spec.patient_id is None:
            continue
        ra = aligned_a.get(spec.record_id)
        rb = aligned_b.get(spec.record_id)
        if ra is not None and ra.patient is not None:
            psns_a.setdefault(spec.patient_id, set()).add(ra.patient)
        if rb is not None and rb.patient is not None:
            psns_b.setdefault(spec.patient_id, set()).add(rb.patient)

    psn_comparable = 0
    psn_reused = 0
    for patient in cohort.patients:
        pid = patient.patient_id
        set_a = psns_a.get(pid)
        set_b = psns_b.get(pid)
        if not set_a or not set_b:
            continue
        psn_comparable += 1
        if any(p in set_b for p in sorted(set_a)):
            psn_reused += 1

    psn_rate: float | None = (psn_reused / psn_comparable) if psn_comparable > 0 else None

    # 2. Event-level shifted date retention (parsed actual date, not formatting).
    record_by_loc = {(s.input_file, s.loc): s for s in cohort.records}
    all_events = cohort.all_events()
    date_comparable = 0
    date_reused = 0
    aligned_events_a = 0
    aligned_events_b = 0
    missing_or_ambiguous_events = 0

    for ev in all_events:
        spec = record_by_loc[(ev.input_file, ev.loc)]
        ra = aligned_a.get(spec.record_id)
        rb = aligned_b.get(spec.record_id)
        da = _parse_iso_or_dmy_date(ra.dates.get(ev.date_field, "")) if ra is not None else None
        db = _parse_iso_or_dmy_date(rb.dates.get(ev.date_field, "")) if rb is not None else None
        if da is not None:
            aligned_events_a += 1
        if db is not None:
            aligned_events_b += 1
        if da is None or db is None:
            missing_or_ambiguous_events += 1
            continue
        date_comparable += 1
        if da == db:
            date_reused += 1

    date_rate: float | None = (date_reused / date_comparable) if date_comparable > 0 else None

    available_rates = [r for r in (psn_rate, date_rate) if r is not None]
    a7_rate: float | None = max(available_rates) if available_rates else None

    missing_rec_total = miss_rec_a + miss_rec_b
    coverage = CoverageSummary(
        expected_records=len(cohort.records),
        aligned_records_a1=len(aligned_a),
        aligned_records_b=len(aligned_b),
        missing_or_ambiguous_records=missing_rec_total,
        expected_events=len(all_events),
        aligned_events_a1=aligned_events_a,
        aligned_events_b=aligned_events_b,
        missing_or_ambiguous_events=missing_or_ambiguous_events,
    )

    systematic_date_reuse = (
        date_comparable > 0 and (date_reused / date_comparable) >= 0.5
    )

    return A7EvaluationResult(
        psn_reused=psn_reused,
        psn_comparable=psn_comparable,
        psn_rate=psn_rate,
        date_reused=date_reused,
        date_comparable=date_comparable,
        date_rate=date_rate,
        a7_rate=a7_rate,
        dropped_or_unresolved=missing_rec_total + missing_or_ambiguous_events,
        systematic_date_reuse=systematic_date_reuse,
        coverage=coverage,
    )


# ---------------------------------------------------------------------------
# Default synthetic cohort & runner double (Phase 1 / Phase 4)
# ---------------------------------------------------------------------------


def build_default_synthetic_cohort() -> SyntheticCohort:
    """Build a deterministic hand-authored synthetic cohort for V10 verification."""
    patients = (
        SyntheticPatient(
            patient_id="fx_p1",
            uhids=("SNJ/26/019672",),
            phones=("9811100001",),
            latin_names=("Ramesh Sharma",),
        ),
        SyntheticPatient(
            patient_id="fx_p2",
            uhids=("SNJ/26/064128",),
            phones=("9811100002",),
            latin_names=("Sunita Devi",),
        ),
    )
    rk1 = ("registration/reg_01.csv", "row:1")
    rk2 = ("discharge/disch_01.txt", "doc:0")
    rk3 = ("opd/opd_01.csv", "row:1")
    rk3b = ("lab/lab_01.csv", "row:1")
    records = (
        SyntheticRecordSpec(
            record_id="r1_reg",
            patient_id="fx_p1",
            input_file=rk1[0],
            loc=rk1[1],
            format="csv",
            fields={"patient_name": "Ramesh Sharma", "uhid": "SNJ/26/019672", "sex": "M", "dob": "1978-04-12"},
            spans=(
                Span(rk1, 0, 0, 13, "PATIENT_NAME", "patient", 1.0, "fixture"),
                Span(rk1, 1, 0, 13, "UHID", "patient", 1.0, "fixture"),
            ),
            dates=("2025-01-10",),
            events=(
                SyntheticEvent(
                    event_id="ev_p1_adm",
                    patient_id="fx_p1",
                    input_file=rk1[0],
                    loc=rk1[1],
                    date_field="visit_date",
                    iso_date="2025-01-10",
                ),
            ),
        ),
        SyntheticRecordSpec(
            record_id="r2_disch",
            patient_id="fx_p1",
            input_file=rk2[0],
            loc=rk2[1],
            format="txt",
            fields={"patient_name": "Ramesh Sharma", "uhid": "SNJ/26/019672", "sex": "M"},
            text="Patient Ramesh Sharma UHID SNJ/26/019672 discharged.",
            spans=(
                Span(rk2, 0, 8, 21, "PATIENT_NAME", "patient", 1.0, "fixture"),
                Span(rk2, 0, 27, 40, "UHID", "patient", 1.0, "fixture"),
            ),
            dates=("2025-01-19",),
            events=(
                SyntheticEvent(
                    event_id="ev_p1_dis",
                    patient_id="fx_p1",
                    input_file=rk2[0],
                    loc=rk2[1],
                    date_field="discharge_date",
                    iso_date="2025-01-19",
                ),
            ),
        ),
        SyntheticRecordSpec(
            record_id="r3_opd",
            patient_id="fx_p2",
            input_file=rk3[0],
            loc=rk3[1],
            format="csv",
            fields={"patient_name": "Sunita Devi", "uhid": "SNJ/26/064128", "sex": "F", "dob": "1985-09-20"},
            spans=(
                Span(rk3, 0, 0, 11, "PATIENT_NAME", "patient", 1.0, "fixture"),
                Span(rk3, 1, 0, 13, "UHID", "patient", 1.0, "fixture"),
            ),
            dates=("2025-02-04",),
            events=(
                SyntheticEvent(
                    event_id="ev_p2_opd",
                    patient_id="fx_p2",
                    input_file=rk3[0],
                    loc=rk3[1],
                    date_field="visit_date",
                    iso_date="2025-02-04",
                ),
            ),
        ),
        SyntheticRecordSpec(
            record_id="r3b_lab",
            patient_id="fx_p2",
            input_file=rk3b[0],
            loc=rk3b[1],
            format="csv",
            fields={"patient_name": "Sunita Devi", "uhid": "SNJ/26/064128", "sex": "F"},
            spans=(
                Span(rk3b, 0, 0, 11, "PATIENT_NAME", "patient", 1.0, "fixture"),
                Span(rk3b, 1, 0, 13, "UHID", "patient", 1.0, "fixture"),
            ),
            dates=("2025-02-06",),
            events=(
                SyntheticEvent(
                    event_id="ev_p2_lab",
                    patient_id="fx_p2",
                    input_file=rk3b[0],
                    loc=rk3b[1],
                    date_field="lab_date",
                    iso_date="2025-02-06",
                ),
            ),
        ),
        SyntheticRecordSpec(
            record_id="r4_orphan",
            patient_id=None,
            input_file="notes/note_unlinked.txt",
            loc="doc:0",
            format="txt",
            fields={"note": "Unlinked clinical note"},
            text="Unlinked clinical note.",
            spans=(),
            dates=("2025-03-01",),
            events=(),
        ),
    )
    return SyntheticCohort(patients=patients, records=records)


class _StaticDateProvider:
    """Minimal ``DateProvider`` returning fixed ISO dates for a fixture record."""

    def __init__(self, dates: tuple[str, ...]) -> None:
        self._dates = dates

    def find(self, rec: Record) -> list[dict[str, Any]]:
        return [{"iso": d} for d in self._dates]


class _SecretHmacKeys:
    """Test-double implementation of ``KeysProtocol`` backed by HMAC-SHA256."""

    def __init__(self, secret: str) -> None:
        self._key = hashlib.sha256(secret.encode("utf-8")).digest()

    def keyed(self, purpose: str, *fields: str) -> bytes:
        msg = "|".join([purpose, *fields]).encode("utf-8")
        return hmac.new(self._key, msg, hashlib.sha256).digest()


def build_release_runner_double() -> ReleaseRunner:
    """Return a deterministic runner double using ``kavachmed.evidence`` + ``linkage.resolve``."""

    def _run(
        cohort: SyntheticCohort,
        output_dir: Path,
        secret: str,
        config: FrozenReleaseConfig,
    ) -> ReleaseRunArtifacts:
        output_dir.mkdir(parents=True, exist_ok=True)
        sanitized_dir = output_dir / "sanitized"
        sanitized_dir.mkdir(parents=True, exist_ok=True)

        evidences: list[Evidence] = []
        file_digests: dict[str, str] = {}
        for spec in cohort.records:
            rk = (spec.input_file, spec.loc)
            if spec.text:
                rec_fields = [Field(0, "content", spec.text, "text")]
            else:
                rec_fields = [
                    Field(idx, f"field:{k}", str(v), "str")
                    for idx, (k, v) in enumerate(spec.fields.items())
                ]
            rec = Record(key=rk, fields=rec_fields, meta={"format": spec.format})
            ev = extract(
                rec,
                list(spec.spans),
                date_provider=_StaticDateProvider(spec.dates),
            )
            evidences.append(ev)
            file_digests[spec.input_file] = hashlib.sha256(
                spec.input_file.encode("utf-8")
            ).hexdigest()

        keys = _SecretHmacKeys(secret)
        resolution = resolve(
            evidences,
            keys,
            policy=config.to_policy_dict(),
            file_digests=file_digests,
        )

        out_records: list[ReleaseRunRecord] = []
        source_to_output: dict[tuple[str, str], tuple[str, str]] = {}
        jsonl_lines: list[str] = []

        for spec in cohort.records:
            rk = (spec.input_file, spec.loc)
            psn = resolution.patient_of.get(rk)
            fam = resolution.family_of.get(rk, "")
            delta = resolution.shift_days.get(fam, 0)

            shifted_dates: dict[str, str] = {}
            for ev in spec.events:
                parsed = _parse_iso_or_dmy_date(ev.iso_date)
                if parsed is not None:
                    shifted_dates[ev.date_field] = (
                        parsed + dt.timedelta(days=delta)
                    ).isoformat()

            out_rel = spec.input_file
            out_path = sanitized_dir / out_rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "loc": spec.loc,
                "patient": psn,
                "dates": shifted_dates,
            }
            out_path.write_text(
                json.dumps(payload, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            source_to_output[rk] = (out_rel, spec.loc)
            out_records.append(
                ReleaseRunRecord(
                    out_file=out_rel,
                    loc=spec.loc,
                    patient=psn,
                    dates=shifted_dates,
                )
            )
            jsonl_lines.append(
                json.dumps(
                    {
                        "file": spec.input_file,
                        "out": out_rel,
                        "loc": spec.loc,
                        "patient": psn,
                    },
                    sort_keys=True,
                )
            )

        (output_dir / "records.jsonl").write_text(
            "\n".join(jsonl_lines) + ("\n" if jsonl_lines else ""),
            encoding="utf-8",
        )
        return ReleaseRunArtifacts(
            output_dir=output_dir,
            records=tuple(out_records),
            source_to_output=source_to_output,
        )

    return _run


def build_default_a6_controls(cohort: SyntheticCohort) -> tuple[A6ControlVector, ...]:
    """Construct positive and negative A6 control vectors for ``cohort``."""
    if not cohort.patients:
        return ()
    p0 = cohort.patients[0]
    orig_id = p0.uhids[0] if p0.uhids else "SNJ/26/019672"
    orig_phone = p0.phones[0] if p0.phones else "9811100001"
    sha256_prefix = hashlib.sha256(orig_id.encode("utf-8")).hexdigest()[:8].upper()
    b64_id = base64.b64encode(orig_id.encode("utf-8")).decode("ascii")
    safe_psn = f"PSN-{CROCKFORD_ALPHABET[3]}7K9M2P4"
    safe_ext = f"{safe_psn}{CROCKFORD_ALPHABET[12]}"
    return (
        A6ControlVector("pos_raw_id", p0.patient_id, orig_id, True, "raw_id"),
        A6ControlVector("pos_wrapped_sha256_prefix", p0.patient_id, f"PSN-{sha256_prefix}", True, "sha256_prefix_8"),
        A6ControlVector("pos_phone_embedded", p0.patient_id, f"PSN-{orig_phone}", True, "embedded_phone"),
        A6ControlVector("pos_b64", p0.patient_id, b64_id, True, "base64"),
        A6ControlVector("neg_keyed_psn", p0.patient_id, safe_psn, False, "keyed_psn"),
        A6ControlVector("neg_collision_ext", p0.patient_id, safe_ext, False, "collision_ext"),
    )


def run_v10_verification(
    cohort: SyntheticCohort,
    runner: ReleaseRunner,
    work_dir: Path,
    *,
    secret_a: str = "synthetic-secret-alpha-01",
    secret_b: str = "synthetic-secret-bravo-02",
    config: FrozenReleaseConfig | None = None,
    controls: Sequence[A6ControlVector] | None = None,
    real_runner_wired: bool = False,
) -> V10VerificationSummary:
    """Run the 3-release V10 verification (A1, A2 same-secret, B rotated-secret)."""
    cfg = config or FrozenReleaseConfig()
    ctrl_vectors = tuple(controls) if controls is not None else build_default_a6_controls(cohort)

    run_a1 = runner(cohort, work_dir / "run_a1", secret_a, cfg)
    run_a2 = runner(cohort, work_dir / "run_a2", secret_a, cfg)
    run_b = runner(cohort, work_dir / "run_b", secret_b, cfg)

    det = verify_same_secret_determinism(cohort, run_a1, run_a2)
    a6_ctrl = evaluate_a6_controls(cohort, ctrl_vectors)
    a6_rel = evaluate_a6_release(cohort, run_a1)
    a7_rot = evaluate_a7_rotation(cohort, run_a1, run_b)

    pending: tuple[str, ...] = ()
    if not real_runner_wired:
        pending = ("real_keys_a3", "real_writer_a", "real_date_projection_d")

    return V10VerificationSummary(
        config=cfg,
        determinism=det,
        a6_controls=a6_ctrl,
        a6_release=a6_rel,
        a7_rotation=a7_rot,
        real_runner_wired=real_runner_wired,
        pending_checks=pending,
    )


def render_v10_summary(summary: V10VerificationSummary) -> str:
    """Render a strictly aggregate, secret-free and identifier-free V10 summary."""
    a6_rate_str = (
        f"{summary.a6_release.rate:.4f}"
        if summary.a6_release.rate is not None
        else "unavailable"
    )
    a7_rate_str = (
        f"{summary.a7_rotation.a7_rate:.4f}"
        if summary.a7_rotation.a7_rate is not None
        else "unavailable"
    )
    lines = [
        f"config: mode={summary.config.linkage_mode} family={summary.config.family_mode} "
        f"tau_high={summary.config.tau_high_bits} tau_low={summary.config.tau_low_bits}",
        f"determinism: passed={summary.determinism.passed} "
        f"files={summary.determinism.files_compared} "
        f"records={summary.determinism.records_compared} "
        f"intervals={summary.determinism.intervals_checked}",
        f"a6_controls: pos={summary.a6_controls.positive_flagged}/{summary.a6_controls.positive_total} "
        f"neg={summary.a6_controls.negative_passed}/{summary.a6_controls.negative_total}",
        f"a6_release: flagged={summary.a6_release.patients_flagged}/{summary.a6_release.patients_evaluated} "
        f"rate={a6_rate_str} unlinked_excluded={summary.a6_release.unlinked_records_excluded}",
        f"a7_rotation: psn={summary.a7_rotation.psn_reused}/{summary.a7_rotation.psn_comparable} "
        f"date={summary.a7_rotation.date_reused}/{summary.a7_rotation.date_comparable} "
        f"a7_rate={a7_rate_str} coverage_adequate={summary.a7_rotation.coverage.adequate} "
        f"unresolved={summary.a7_rotation.dropped_or_unresolved}",
        f"runner: real_wired={summary.real_runner_wired} pending={','.join(summary.pending_checks) or 'none'}",
    ]
    return "\n".join(lines)
