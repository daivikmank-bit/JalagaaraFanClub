"""CSV/TSV adapter: strict parsing, @header channel, type-signature row healer, LocMap (A5)."""
from __future__ import annotations

import csv
import io
import re
from collections import Counter, defaultdict

from kavachmed.contracts import Edit, Field, FileEntry, LocEntry, LocMap, ParsedFile, RecKey, Record, Span
from kavachmed.security import StageError
from kavachmed.urm import apply_field_edits_with_loc

_INT_RE = re.compile(r"^\s*\d+\s*$")
_DATE_RE = re.compile(r"^\s*\d{1,4}[-/]\d{1,2}[-/]\d{1,4}\s*$")


def _has_unclosed_quote(text: str) -> bool:
    in_quote = False
    i = 0
    n = len(text)
    while i < n:
        if text[i] == '"':
            if in_quote and i + 1 < n and text[i + 1] == '"':
                i += 2
                continue
            in_quote = not in_quote
        i += 1
    return in_quote


def _detect_newline(text: str) -> str:
    if "\r\n" in text:
        return "\r\n"
    if "\n" in text:
        return "\n"
    if "\r" in text:
        return "\r"
    return "\n"


def _cell_shape(val: str) -> str:
    if not val.strip():
        return "empty"
    if _INT_RE.match(val):
        return "int"
    if _DATE_RE.match(val):
        return "date"
    return "text"


def _build_col_profiles(body: list[list[str]], width: int) -> list[set[str]]:
    profiles: list[set[str]] = [set() for _ in range(width)]
    for row in body:
        if len(row) == width:
            for j, cell in enumerate(row):
                sh = _cell_shape(cell)
                if sh != "empty":
                    profiles[j].add(sh)
    return profiles


def _score_candidate(cand: list[str], profiles: list[set[str]]) -> int:
    score = 0
    for cell, prof in zip(cand, profiles):
        sh = _cell_shape(cell)
        if sh == "empty":
            continue
        if not prof:
            score += 1
        elif sh in prof:
            score += 2
        else:
            return -1
    return score


def _repair_overwide(row: list[str], width: int, delim: str, profiles: list[set[str]]) -> list[str]:
    extra = len(row) - width
    if extra > 4:
        raise StageError("parse", "E_CSV")

    best_score = -1
    best_cands: list[list[str]] = []

    def backtrack(idx: int, cur: list[str], merges_left: int) -> None:
        nonlocal best_score, best_cands
        if idx == len(row):
            if len(cur) == width:
                sc = _score_candidate(cur, profiles)
                if sc > best_score:
                    best_score = sc
                    best_cands = [cur]
                elif sc == best_score and sc >= 0:
                    best_cands.append(cur)
            return
        # Option 1: take row[idx] as its own cell
        if len(cur) < width:
            backtrack(idx + 1, [*cur, row[idx]], merges_left)
        # Option 2: merge with previous cell
        if merges_left > 0 and cur:
            merged = [*cur[:-1], cur[-1] + delim + row[idx]]
            backtrack(idx + 1, merged, merges_left - 1)

    backtrack(0, [], extra)
    if best_score < 0 or len(best_cands) != 1:
        raise StageError("parse", "E_CSV")
    return best_cands[0]


def _repair_short(row: list[str], width: int, profiles: list[set[str]]) -> list[str]:
    missing = width - len(row)
    if missing > 4:
        raise StageError("parse", "E_CSV")

    best_score = -1
    best_cands: list[list[str]] = []

    def backtrack(idx: int, cur: list[str], inserts_left: int) -> None:
        nonlocal best_score, best_cands
        if len(cur) == width:
            if idx == len(row):
                sc = _score_candidate(cur, profiles)
                if sc > best_score:
                    best_score = sc
                    best_cands = [cur]
                elif sc == best_score and sc >= 0:
                    best_cands.append(cur)
            return
        if idx < len(row):
            backtrack(idx + 1, [*cur, row[idx]], inserts_left)
        if inserts_left > 0:
            backtrack(idx, [*cur, ""], inserts_left - 1)

    backtrack(0, [], missing)
    if best_score < 0 or len(best_cands) != 1:
        raise StageError("parse", "E_CSV")
    return best_cands[0]


def _serialize_cell(
    text: str,
    edits: list[Edit],
    key: RecKey,
    field_idx: int,
    delim: str,
    out_cur: int,
    protected_spans: list[Span] | None = None,
) -> tuple[str, list[LocEntry]]:
    rep, raw_locs = apply_field_edits_with_loc(
        text, edits, key, field_idx, 0, protected_spans=protected_spans
    )
    needs_quote = any(c in rep for c in (delim, '"', "\n", "\r"))
    if not needs_quote:
        shifted = [
            LocEntry(out_cur + e.out_start, out_cur + e.out_end, e.key, e.field_idx, e.in_start, e.in_end, e.exempt_kind)
            for e in raw_locs
        ]
        return rep, shifted

    # Account for opening quote and doubled internal quotes
    escaped_parts: list[str] = ['"']
    # Map char index in `rep` -> char index in `escaped`
    rep_to_esc: list[int] = [0] * (len(rep) + 1)
    esc_pos = 1
    for i, ch in enumerate(rep):
        rep_to_esc[i] = esc_pos
        if ch == '"':
            escaped_parts.append('""')
            esc_pos += 2
        else:
            escaped_parts.append(ch)
            esc_pos += 1
    rep_to_esc[len(rep)] = esc_pos
    escaped_parts.append('"')

    shifted = [
        LocEntry(
            out_cur + rep_to_esc[e.out_start],
            out_cur + rep_to_esc[e.out_end],
            e.key,
            e.field_idx,
            e.in_start,
            e.in_end,
            e.exempt_kind,
        )
        for e in raw_locs
    ]
    return "".join(escaped_parts), shifted


def parse(entry: FileEntry) -> ParsedFile:
    assert entry.text is not None
    if _has_unclosed_quote(entry.text):
        raise StageError("parse", "E_CSV")

    delim = "\t" if entry.sniff_fmt == "tsv" else ","
    newline = _detect_newline(entry.text)
    trailing_nl = entry.text.endswith(("\r\n", "\n", "\r"))

    try:
        rows = list(csv.reader(io.StringIO(entry.text, newline=""), delimiter=delim, strict=True))
    except csv.Error as exc:
        raise StageError("parse", "E_CSV") from exc

    if not rows:
        raise StageError("parse", "E_EMPTY")

    header = rows[0]
    body = rows[1:]
    width = len(header)
    profiles = _build_col_profiles(body, width)

    hdr_key: RecKey = (entry.rel_path, "@header")
    hdr_rec = Record(
        key=hdr_key,
        fields=[Field(idx=j, path=f"@header/{j}", text=h, vtype="header") for j, h in enumerate(header)],
        meta={"encoding": entry.encoding, "delimiter": delim},
    )

    records: list[Record] = [hdr_rec]
    repaired_count = 0
    for i, raw_row in enumerate(body):
        if len(raw_row) > width:
            row = _repair_overwide(raw_row, width, delim, profiles)
            repaired_count += 1
        elif len(raw_row) < width:
            row = _repair_short(raw_row, width, profiles)
            repaired_count += 1
        else:
            row = raw_row

        key: RecKey = (entry.rel_path, f"row:{i}")
        fields = [
            Field(idx=j, path=header[j] if j < len(header) else f"col:{j}", text=cell, vtype="str")
            for j, cell in enumerate(row)
        ]
        records.append(Record(key=key, fields=fields, meta={"encoding": entry.encoding, "delimiter": delim}))

    def projector(
        edits: list[Edit],
        psn_map: dict[RecKey, str | None] | None = None,
        *,
        protected_spans: list[Span] | None = None,
    ) -> tuple[bytes, LocMap]:
        by_field: dict[tuple[RecKey, int], list[Edit]] = defaultdict(list)
        psn_edited_recs: set[RecKey] = set()
        for ed in edits:
            by_field[(ed.key, ed.field_idx)].append(ed)
            if ed.kind == "PSN":
                psn_edited_recs.add(ed.key)

        prot_by_field: dict[tuple[RecKey, int], list[Span]] = defaultdict(list)
        for sp in protected_spans or []:
            if sp.protected:
                prot_by_field[(sp.key, sp.field_idx)].append(sp)

        needs_patient_ref_col = any(
            (psn_map or {}).get(rec.key) is not None and rec.key not in psn_edited_recs
            for rec in records
            if not rec.key[1].startswith("@")
        )

        out_parts: list[str] = []
        loc_entries: list[LocEntry] = []
        out_cur = 0

        for r_idx, rec in enumerate(records):
            if r_idx > 0:
                out_parts.append(newline)
                out_cur += len(newline)
            for f_idx, f in enumerate(rec.fields):
                if f_idx > 0:
                    out_parts.append(delim)
                    out_cur += len(delim)
                cell_str, cell_locs = _serialize_cell(
                    f.text,
                    by_field.get((rec.key, f.idx), []),
                    rec.key,
                    f.idx,
                    delim,
                    out_cur,
                    protected_spans=prot_by_field.get((rec.key, f.idx)),
                )
                out_parts.append(cell_str)
                loc_entries.extend(cell_locs)
                out_cur += len(cell_str)
            if needs_patient_ref_col:
                out_parts.append(delim)
                out_cur += len(delim)
                extra_val = "patient_ref" if rec.key[1] == "@header" else ((psn_map or {}).get(rec.key) or "")
                out_parts.append(extra_val)
                out_cur += len(extra_val)

        if trailing_nl:
            out_parts.append(newline)

        return "".join(out_parts).encode("utf-8"), LocMap(loc_entries)

    return ParsedFile(
        rel_path=entry.rel_path,
        sha256=entry.sha256,
        fmt="tsv" if entry.sniff_fmt == "tsv" else "csv",
        encoding=entry.encoding or "utf-8",
        newline=newline,
        records=records,
        raw_bytes=entry.raw_bytes,
        projector=projector,
        extra={"repaired_rows": repaired_count},
    )
