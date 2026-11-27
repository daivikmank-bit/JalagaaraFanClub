"""Edit resolution, priority ranking, protected-span carving, and numeric surrogates (A8)."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from kavachmed.contracts import Edit, PrivacyPlan, Record, Resolution, Span
from kavachmed.security import Keys, StageError

_IDENTIFIER_TYPES = frozenset({
    "AADHAAR", "AADHAAR_VID", "AADHAAR_MASKED", "ABHA_NUMBER", "ABHA_ADDRESS",
    "MRN", "UHID", "IPD", "OPD", "PAN", "EPIC", "DL", "PASSPORT", "GSTIN",
    "PHONE", "EMAIL", "UPI", "INSURANCE_ID", "POLICY_NO", "PMJAY_ID",
    "VEHICLE_PLATE", "DOCTOR_REG",
})

_NAME_TYPES = frozenset({
    "PATIENT_NAME", "RELATIVE_NAME", "THIRD_PARTY_NAME", "SENDER_NAME", "NAME",
})

_DATE_AGE_TYPES = frozenset({"DATE", "DOB", "AGE"})
_WHOLE_TOKEN_TYPES = frozenset({"DATE", "AGE"})


def overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def surrogate_integer(keys: Keys, namespace: str, source: str) -> str:
    """Deterministic width-preserving positive integer surrogate using k_surr."""
    if not source.isascii() or not source.isdecimal() or source.startswith("0"):
        raise StageError("edits", "E_IO")
    floor = 10 ** (len(source) - 1)
    size = 9 * floor
    digest = keys.keyed("surr", namespace, source)
    candidate = floor + int.from_bytes(digest, "big") % size
    if str(candidate) == source:
        candidate = floor + (candidate - floor + 1) % size
    return str(candidate)


def _span_tier(sp: Span) -> int:
    if sp.typ in _IDENTIFIER_TYPES:
        return 4
    if sp.typ in _NAME_TYPES:
        return 3
    if sp.typ in _DATE_AGE_TYPES:
        return 2
    return 1


def _subtract_intervals(start: int, end: int, blocked: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return sorted uncovered sub-intervals of [start, end) after removing `blocked`."""
    cur = [(start, end)]
    for bs, be in blocked:
        nxt: list[tuple[int, int]] = []
        for cs, ce in cur:
            if not overlaps(cs, ce, bs, be):
                nxt.append((cs, ce))
            else:
                if cs < bs:
                    nxt.append((cs, bs))
                if be < ce:
                    nxt.append((be, ce))
        cur = nxt
        if not cur:
            break
    return cur


def _replacement_for(
    sp: Span,
    field_vtype: str,
    field_path: str,
    field_text: str,
    psn: str | None,
    cluster_id: str | None,
    family_shift: int,
    plan: PrivacyPlan,
    keys: Keys | None,
    date_shifter: Any,
) -> tuple[str, str]:
    if sp.typ == "AGE":
        if cluster_id is not None and cluster_id in plan.age_band:
            return "BAND_AGE", plan.age_band[cluster_id]
        return "PLACEHOLDER", "[AGE]"

    if sp.typ == "DOB":
        return "PLACEHOLDER", "[DOB]"

    if sp.typ == "DATE":
        if date_shifter is not None and family_shift != 0:
            shifted = date_shifter(field_text[sp.start:sp.end], family_shift)
            if shifted:
                return "SHIFT_DATE", shifted
        return "PLACEHOLDER", "[DATE]"

    if field_vtype in ("int", "float"):
        if keys is None:
            raise StageError("edits", "E_SECRET")
        if field_vtype == "int" and field_text.isascii() and field_text.isdecimal() and not field_text.startswith("0"):
            return "NUM_SURROGATE", surrogate_integer(keys, field_path, field_text)
        return "NUM_SURROGATE", "0.0" if field_vtype == "float" else "0"

    if sp.typ == "PATIENT_NAME" and sp.role == "patient":
        if psn is not None:
            return "PSN", psn
        return "PLACEHOLDER", "[NAME]"

    if sp.typ == "RELATIVE_NAME" or sp.role == "relative":
        return "PLACEHOLDER", "[RELATIVE]"

    if sp.typ in _NAME_TYPES:
        return "PLACEHOLDER", "[NAME]"

    return "PLACEHOLDER", f"[{sp.typ}]"


def build(
    rec: Record,
    spans: list[Span],
    resolution: Resolution,
    plan: PrivacyPlan,
    *,
    keys: Keys | None = None,
    date_shifter: Any = None,
) -> list[Edit]:
    """Resolve spans into non-overlapping, priority-ranked Edits preserving protected text."""
    shield_by_field: dict[int, list[tuple[int, int]]] = defaultdict(list)
    cand_by_field: dict[int, list[Span]] = defaultdict(list)

    for sp in spans:
        if sp.key != rec.key:
            raise StageError("edits", "E_IO")
        if sp.field_idx < 0 or sp.field_idx >= len(rec.fields):
            raise StageError("edits", "E_IO")
        flen = len(rec.fields[sp.field_idx].text)
        if sp.start < 0 or sp.end > flen or sp.start >= sp.end:
            raise StageError("edits", "E_IO")

        if sp.protected:
            shield_by_field[sp.field_idx].append((sp.start, sp.end))
        else:
            cand_by_field[sp.field_idx].append(sp)

    psn = resolution.patient_of.get(rec.key)
    # Unlinked records use the orphan cluster id under which privacy.plan stores their band
    cluster_id = resolution.cluster_of.get(rec.key) or f"orphan:{rec.key[0]}|{rec.key[1]}"
    fam = resolution.family_of.get(rec.key, rec.key[0])
    fam_shift = resolution.shift_days.get(fam, 0)

    edits: list[Edit] = []
    for fidx in sorted(cand_by_field.keys()):
        field = rec.fields[fidx]
        blocked = list(shield_by_field.get(fidx, []))
        ranked = sorted(
            cand_by_field[fidx],
            key=lambda s: (-_span_tier(s), -s.score, -(s.end - s.start), s.start, s.typ),
        )
        field_edits: list[Edit] = []
        for sp in ranked:
            uncovered = _subtract_intervals(sp.start, sp.end, blocked)
            if not uncovered:
                continue
            is_whole = sp.typ in _WHOLE_TOKEN_TYPES or field.vtype in ("int", "float")
            if is_whole and uncovered != [(sp.start, sp.end)]:
                if sp.typ in _IDENTIFIER_TYPES:
                    raise StageError("edits", "E_IO")
                continue
            if field.vtype in ("int", "float") and (sp.start != 0 or sp.end != len(field.text)):
                continue

            kind, rep_text = _replacement_for(
                sp,
                field.vtype,
                field.path,
                field.text,
                psn,
                cluster_id,
                fam_shift,
                plan,
                keys,
                date_shifter,
            )
            for us, ue in uncovered:
                field_edits.append(Edit(
                    key=rec.key,
                    field_idx=fidx,
                    start=us,
                    end=ue,
                    kind=kind,  # type: ignore[arg-type]
                    text=rep_text,
                ))
                blocked.append((us, ue))

        field_edits.sort(key=lambda e: (e.start, e.end))
        edits.extend(field_edits)

    return edits
