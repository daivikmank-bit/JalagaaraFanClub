"""Field-level dispositions and independent validation of the staged representation."""
from dataclasses import replace
from pathlib import Path, PurePosixPath

from kavachmed import adapters, credentials as cred, residual_a, residual_b
from kavachmed.contracts import Edit, FileEntry
from kavachmed.security import StageError


def field_policy(rec):
    edits = []
    view = []
    roles = rec.meta.get("release_roles", {})
    for fld in rec.fields:
        role = roles.get(fld.idx)
        ranges = []
        if fld.vtype not in {"key", "header", "path"}:
            leaf = fld.path.rsplit("/", 1)[-1].lstrip("@")
            if leaf in {"text", "cdata"} and "/" in fld.path:
                leaf = fld.path.split("/")[-2]
            if not role and cred.normalized(leaf) in cred.SECRET_FIELDS:
                role = "credential"
            if role in {"credential", "source", "attachment_meta"}:
                marker = {"credential": cred.MARKER, "source": cred.SOURCE, "attachment_meta": ""}[role]
                if fld.text and fld.text != marker:
                    if fld.vtype != "str":
                        raise StageError("parse", "E_FIELD_TYPE")
                    ranges.append((0, len(fld.text), marker))
            else:
                ranges.extend((s, e, cred.MARKER) for s, e in cred.inline_ranges(fld.text))
                if role == "attachment":
                    ranges.extend((*m.span(1), "[MRN]") for m in cred._MRN_LABEL.finditer(fld.text))
        masked = fld.text
        last = len(masked)
        for s, e, marker in sorted(set(ranges), reverse=True):
            if e > last:
                continue
            edits.append(Edit(rec.key, fld.idx, s, e, "PLACEHOLDER", marker))
            masked = masked[:s] + " " * (e - s) + masked[e:]
            last = s
        view.append(fld if masked == fld.text else replace(fld, text=masked))
    return replace(rec, fields=view), edits


def verify(data, rel, fmt, encoding, policy, leak_keys):
    from kavachmed.adapters import fhir
    from kavachmed.rename import contains_local_id
    path = PurePosixPath(rel)
    if any(contains_local_id(c) for c in (*path.parts[:-1], path.stem)):
        return False
    text = data.decode(encoding)
    if cred.key_material(text):
        return False
    sniff = "txt" if fmt == "text" else "json" if fmt == "fhir" else fmt
    parsed = adapters.parse(FileEntry(Path(rel), rel, "", data, sniff, encoding, text, None), policy)
    def decoded_ok(value):
        return not (cred.key_material(value) or list(cred.inline_ranges(value))
                    or cred._MRN_LABEL.search(value) or residual_a.scan_gate(value, rel, leak_keys)
                    or residual_b.scan(value, rel))
    if fmt == "fhir":
        import json
        return fhir.verify_channels(json.loads(text), decoded_ok)
    for rec in parsed.records:
        if field_policy(rec)[1]:
            return False
        for fld in rec.fields:
            if fld.vtype == "b64" and not decoded_ok(fld.text):
                return False
    return True

