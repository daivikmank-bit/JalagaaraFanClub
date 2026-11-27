"""Format adapter router."""
from __future__ import annotations

from typing import Any

from kavachmed.adapters import csv_tsv, fhir, json_iter, text, xml as xml_adapter
from kavachmed.contracts import FileEntry, ParsedFile
from kavachmed.security import StageError


def parse(entry: FileEntry, policy: dict[str, Any]) -> ParsedFile:
    if entry.quarantine_reason is not None or entry.text is None:
        raise StageError("parse", "E_UNSUPPORTED")
    if entry.sniff_fmt in ("csv", "tsv"):
        return csv_tsv.parse(entry)
    if entry.sniff_fmt == "json":
        max_depth = int(policy.get("runtime", {}).get("max_json_depth", 4000))
        pf = json_iter.parse(entry, max_depth=max_depth)
        if pf.fmt == "fhir" and getattr(fhir, "REGISTER_FHIR", False):
            return fhir.parse(entry, policy)
        return pf
    if entry.sniff_fmt == "txt":
        return text.parse(entry)
    if entry.sniff_fmt == "xml":
        return xml_adapter.parse(entry, policy)
    raise StageError("parse", "E_UNSUPPORTED")
