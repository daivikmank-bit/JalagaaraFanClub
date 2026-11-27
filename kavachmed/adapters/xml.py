"""XML adapter — sanitising projector with safe quarantine fallback (A12).

A full Expat-based sanitising adapter is deferred pending Owner C/D residual
scanners.  This module provides the *safe quarantine* path:

  * Attempts to parse the file as XML using xml.parsers.expat with explicit
    DTD/entity rejection.
  * If the file is well-formed XML → quarantines with reason 'unsupported_format'
    (safe, documented).
  * If the file is malformed XML → quarantines with reason 'validation_failed'.
  * Never copies raw XML into sanitized/.

No PII surfaces are written to logs or errors; only format/structural codes.
"""
from __future__ import annotations

import re
import xml.parsers.expat as expat
from collections import defaultdict
from typing import Any

from kavachmed.contracts import Edit, Field, FileEntry, LocEntry, LocMap, ParsedFile, RecKey, Record, Span
from kavachmed.security import StageError
from kavachmed.urm import apply_field_edits_with_loc


def _reject_doctype(
    doctypeName: str,  # noqa: N803
    systemId: str,  # noqa: N803
    publicId: str,  # noqa: N803
    has_internal_subset: bool,
) -> None:
    raise StageError("parse", "E_XML_DECL")


def _reject_entity(context: str, base: str, systemId: str, publicId: str) -> bool:  # noqa: N803
    # Returning False causes the parser to treat the entity as an error.
    return False


def _probe_xml(raw: bytes) -> str:
    """Return 'well_formed', 'malformed', or 'decl_rejected'."""
    p = expat.ParserCreate()
    p.StartDoctypeDeclHandler = _reject_doctype  # type: ignore[assignment]
    p.ExternalEntityRefHandler = _reject_entity  # type: ignore[assignment]

    # Bounded element stack counter to guard against depth exhaustion
    depth: list[int] = [0]
    max_depth = 500

    def _start(name: str, attrs: dict[str, str]) -> None:
        depth[0] += 1
        if depth[0] > max_depth:
            raise StageError("parse", "E_DEPTH")

    def _end(name: str) -> None:
        depth[0] -= 1

    p.StartElementHandler = _start
    p.EndElementHandler = _end

    try:
        p.Parse(raw, True)
        return "well_formed"
    except StageError as exc:
        if exc.code == "E_XML_DECL":
            return "decl_rejected"
        raise  # re-raise E_DEPTH
    except expat.ExpatError:
        return "malformed"


# Structural attributes are copied verbatim (namespaces, code systems, OIDs, class codes);
# every other attribute value (extension="UHID-…", value="…", displayName) is a field.
_STRUCTURAL_ATTRS = frozenset({
    "xmlns", "version", "encoding", "standalone", "root", "codesystem", "codesystemname",
    "codesystemversion", "classcode", "moodcode", "typecode", "determinercode", "contextcontrolcode",
    "templateid", "xsi:type", "type", "use", "mediatype", "representation", "nullflavor",
    "inversionind", "negationind", "contextconductionind",
})
_RE_MARKUP = re.compile(r"<!--(?P<comment>.*?)-->|<!\[CDATA\[(?P<cdata>.*?)\]\]>|<\?.*?\?>|<(?P<tag>/?[^<>]*?)>", re.S)
_RE_ATTR = re.compile(r"""([A-Za-z_:][-A-Za-z0-9_:.]*)\s*=\s*(["'])(.*?)\2""", re.S)
_RE_TAGNAME = re.compile(r"/?\s*([A-Za-z_:][-A-Za-z0-9_:.]*)")


def parse(entry: FileEntry, policy: dict[str, Any]) -> ParsedFile:
    """Sanitise well-formed XML (A12): text nodes, comments, CDATA and non-structural attribute
    values become fields; markup is copied byte-for-byte. DTDs, external entities, excessive
    depth and malformed XML are quarantined (never copied raw)."""
    if entry.text is None or entry.quarantine_reason is not None:
        raise StageError("parse", "E_UNSUPPORTED")
    raw: bytes = entry.raw_bytes if entry.raw_bytes is not None else b""
    if not raw:
        raise StageError("parse", "E_EMPTY")
    try:
        result = _probe_xml(raw)
    except StageError:
        raise StageError("parse", "E_UNSUPPORTED")
    if result != "well_formed":
        raise StageError("parse", "E_UNSUPPORTED")

    src = entry.text
    key: RecKey = (entry.rel_path, "doc")
    fields: list[Field] = []
    segments: list[tuple[str, str | int]] = []
    tag_stack: list[str] = []

    def lit(t: str) -> None:
        if t:
            segments.append(("lit", t))

    def field(path: str, t: str) -> None:
        if not t.strip():
            lit(t)
            return
        idx = len(fields)
        fields.append(Field(idx=idx, path=path, text=t, vtype="str"))
        segments.append(("field", idx))

    pos = 0
    for m in _RE_MARKUP.finditer(src):
        field("/".join(tag_stack[-2:]) + "/text" if tag_stack else "text", src[pos:m.start()])
        if m.group("comment") is not None:
            lit("<!--"); field("comment", m.group("comment")); lit("-->")
        elif m.group("cdata") is not None:
            lit("<![CDATA["); field("/".join(tag_stack[-2:]) + "/cdata", m.group("cdata")); lit("]]>")
        elif m.group("tag") is not None:
            body = m.group("tag")
            nm = _RE_TAGNAME.match(body)
            name = nm.group(1) if nm else ""
            lit("<")
            cur = 0
            for am in _RE_ATTR.finditer(body):
                aname = am.group(1)
                vs, ve = am.span(3)
                lit(body[cur:vs])
                if aname.casefold() in _STRUCTURAL_ATTRS or aname.casefold().startswith("xmlns"):
                    lit(body[vs:ve])
                else:
                    field(f"{name}/@{aname}", body[vs:ve])
                cur = ve
            lit(body[cur:])
            lit(">")
            if body.startswith("/"):
                if tag_stack:
                    tag_stack.pop()
            elif not body.rstrip().endswith("/") and name:
                tag_stack.append(name)
        else:
            lit(m.group(0))
        pos = m.end()
    field("text", src[pos:])

    rec = Record(key=key, fields=fields, meta={"encoding": entry.encoding})

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
        prot: dict[int, list[Span]] = defaultdict(list)
        for sp in protected_spans or []:
            if sp.key == key and sp.protected:
                prot[sp.field_idx].append(sp)
        parts: list[str] = []
        locs: list[LocEntry] = []
        out_cur = 0
        for kind, val in segments:
            if kind == "lit":
                parts.append(str(val)); out_cur += len(str(val))
                continue
            fidx = int(val)
            rep, f_locs = apply_field_edits_with_loc(
                fields[fidx].text, by_field.get(fidx, []), key, fidx, out_cur, protected_spans=prot.get(fidx)
            )
            parts.append(rep); locs.extend(f_locs); out_cur += len(rep)
        out = "".join(parts).encode("utf-8")
        if _probe_xml(out) != "well_formed":
            raise StageError("edits", "E_IO")  # never emit broken XML (quarantine instead)
        return out, LocMap(locs)

    return ParsedFile(
        rel_path=entry.rel_path, sha256=entry.sha256, fmt="xml", encoding=entry.encoding or "utf-8",
        newline="\n", records=[rec], raw_bytes=raw, projector=projector,
    )
