"""Universal Record Model (URM) helpers for edit application and LocMap building."""
from __future__ import annotations

from kavachmed.contracts import Edit, LocEntry, RecKey, Span


def apply_field_edits(text: str, edits: list[Edit]) -> str:
    """Apply non-overlapping Edits (in input coordinates) to a single Field's text."""
    new_text, _ = apply_field_edits_with_loc(text, edits, ("", ""), 0, 0)
    return new_text


def _emit_unchanged_with_shields(
    text: str,
    seg_start: int,
    seg_end: int,
    out_cur: int,
    key: RecKey,
    field_idx: int,
    shields: list[Span],
    out_parts: list[str],
    entries: list[LocEntry],
) -> int:
    """Emit [seg_start, seg_end) split at protected shield boundaries."""
    pos = seg_start
    for sh in shields:
        if sh.end <= pos or sh.start >= seg_end:
            continue
        s_clamped = max(pos, sh.start)
        e_clamped = min(seg_end, sh.end)
        if s_clamped > pos:
            chunk = text[pos:s_clamped]
            out_parts.append(chunk)
            entries.append(LocEntry(out_cur, out_cur + len(chunk), key, field_idx, pos, s_clamped, None))
            out_cur += len(chunk)
        if e_clamped > s_clamped:
            chunk = text[s_clamped:e_clamped]
            out_parts.append(chunk)
            entries.append(LocEntry(out_cur, out_cur + len(chunk), key, field_idx, s_clamped, e_clamped, sh.exempt_kind))
            out_cur += len(chunk)
        pos = e_clamped

    if pos < seg_end:
        tail = text[pos:seg_end]
        out_parts.append(tail)
        entries.append(LocEntry(out_cur, out_cur + len(tail), key, field_idx, pos, seg_end, None))
        out_cur += len(tail)
    return out_cur


def apply_field_edits_with_loc(
    text: str,
    edits: list[Edit],
    key: RecKey,
    field_idx: int,
    out_offset: int,
    protected_spans: list[Span] | None = None,
) -> tuple[str, list[LocEntry]]:
    """Apply non-overlapping Edits and emit character-level LocEntry mappings."""
    shields = sorted(
        (s for s in (protected_spans or []) if s.protected and s.exempt_kind is not None),
        key=lambda s: (s.start, s.end),
    )
    if not edits:
        if not text:
            return "", []
        out_parts: list[str] = []
        entries: list[LocEntry] = []
        _emit_unchanged_with_shields(text, 0, len(text), out_offset, key, field_idx, shields, out_parts, entries)
        return "".join(out_parts), entries

    ordered = sorted(edits, key=lambda e: (e.start, e.end))
    out_parts = []
    entries = []
    cursor = 0
    out_cur = out_offset

    for ed in ordered:
        if ed.start < cursor:
            continue
        if ed.start > cursor:
            out_cur = _emit_unchanged_with_shields(
                text, cursor, ed.start, out_cur, key, field_idx, shields, out_parts, entries
            )
        if ed.text:
            out_parts.append(ed.text)
            entries.append(LocEntry(out_cur, out_cur + len(ed.text), key, field_idx, ed.start, ed.end, None))
            out_cur += len(ed.text)
        cursor = ed.end

    if cursor < len(text):
        _emit_unchanged_with_shields(
            text, cursor, len(text), out_cur, key, field_idx, shields, out_parts, entries
        )

    return "".join(out_parts), entries
