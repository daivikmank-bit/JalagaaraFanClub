"""Plain text and WhatsApp chat adapter (A7)."""
from __future__ import annotations

import re
from collections import defaultdict

from kavachmed.contracts import Edit, Field, FileEntry, LocEntry, LocMap, ParsedFile, RecKey, Record, Span
from kavachmed.urm import apply_field_edits_with_loc

CHAT_PREFIX = re.compile(
    r"^\[(?P<date>[^,\]\r\n]+),[ \t]*(?P<time>[^\]\r\n]+)\]"
    r"[ \t]+(?P<sender>[^:\r\n]+):[ \t]*(?P<message>.*)$"
)


def _split_line_ending(line: str) -> tuple[str, str]:
    if line.endswith("\r\n"):
        return line[:-2], "\r\n"
    if line.endswith("\n"):
        return line[:-1], "\n"
    if line.endswith("\r"):
        return line[:-1], "\r"
    return line, ""


def parse(entry: FileEntry) -> ParsedFile:
    assert entry.text is not None
    key: RecKey = (entry.rel_path, "doc")
    fields: list[Field] = []
    # Each segment is ("lit", str) or ("field", field_idx)
    segments: list[tuple[str, str | int]] = []

    def add_field(path: str, text: str) -> None:
        idx = len(fields)
        fields.append(Field(idx=idx, path=path, text=text, vtype="str"))
        segments.append(("field", idx))

    detected_nl = "\n"
    for line_no, raw_line in enumerate(entry.text.splitlines(keepends=True)):
        body, eol = _split_line_ending(raw_line)
        if line_no == 0 and eol:
            detected_nl = eol

        m = CHAT_PREFIX.match(body)
        if m is not None:
            ds, de = m.span("date")
            ts, te = m.span("time")
            ss, se = m.span("sender")
            ms, me = m.span("message")
            segments.append(("lit", body[:ds]))
            add_field(f"lines/{line_no}/date", m.group("date"))
            segments.append(("lit", body[de:ts]))
            add_field(f"lines/{line_no}/time", m.group("time"))
            segments.append(("lit", body[te:ss]))
            add_field(f"lines/{line_no}/sender", m.group("sender"))
            segments.append(("lit", body[se:ms]))
            add_field(f"lines/{line_no}/message", m.group("message"))
        elif body:
            add_field(f"lines/{line_no}", body)

        if eol:
            segments.append(("lit", eol))

    rec = Record(key=key, fields=fields, meta={"encoding": entry.encoding, "newline": detected_nl})

    def projector(
        edits: list[Edit],
        psn_map: dict[RecKey, str | None] | None = None,
        *,
        protected_spans: list[Span] | None = None,
    ) -> tuple[bytes, LocMap]:
        by_field: dict[int, list[Edit]] = defaultdict(list)
        for ed in edits:
            if ed.key == key:
                by_field[ed.field_idx].append(ed)

        prot_by_field: dict[int, list[Span]] = defaultdict(list)
        for sp in protected_spans or []:
            if sp.key == key and sp.protected:
                prot_by_field[sp.field_idx].append(sp)

        parts: list[str] = []
        loc_entries: list[LocEntry] = []
        out_cur = 0

        psn = (psn_map or {}).get(key)
        has_psn_edit = any(ed.key == key and ed.kind == "PSN" for ed in edits)
        if psn is not None and not has_psn_edit:
            prefix = f"Patient: {psn}{detected_nl}"
            parts.append(prefix)
            out_cur += len(prefix)

        for kind, val in segments:
            if kind == "lit":
                lit = str(val)
                parts.append(lit)
                out_cur += len(lit)
            else:
                fidx = int(val)
                rep, f_locs = apply_field_edits_with_loc(
                    fields[fidx].text,
                    by_field.get(fidx, []),
                    key,
                    fidx,
                    out_cur,
                    protected_spans=prot_by_field.get(fidx),
                )
                parts.append(rep)
                loc_entries.extend(f_locs)
                out_cur += len(rep)

        return "".join(parts).encode("utf-8"), LocMap(loc_entries)

    return ParsedFile(
        rel_path=entry.rel_path,
        sha256=entry.sha256,
        fmt="text",
        encoding=entry.encoding or "utf-8",
        newline=detected_nl,
        records=[rec],
        raw_bytes=entry.raw_bytes,
        projector=projector,
    )
