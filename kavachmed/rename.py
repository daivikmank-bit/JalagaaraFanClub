"""Path, directory, key, and header renaming with collision-free allocation (A9)."""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import PurePosixPath

from kavachmed.local_ids import contains_local_id
from kavachmed.contracts import Edit, ParsedFile, Resolution, Span
from kavachmed.security import Keys

_SPLIT_RE = re.compile(r"[_\-.\s\\/]+")
_DIGIT_CORE_RE = re.compile(r"\d{5,}")
_SAFE_EXTS = frozenset({".csv", ".tsv", ".json", ".txt", ".xml"})

_SAFE_BOILERPLATE = frozenset({
    "registration", "opd", "discharge", "notes", "lab", "fhir", "billing",
    "clean", "misc", "report", "summary", "visits", "patients", "patient",
    "sanitized", "export", "legacy", "sanjeevani", "jeevan", "jyoti", "arogya",
    "hardoi", "ward", "vitals", "sheet", "helpdesk", "tickets", "ambulance",
    "log", "scanned", "consent", "form", "index", "deep", "note", "table",
    "data", "bundle", "claim", "claims", "rx", "prescription", "prescriptions",
    "protocol", "protocols", "formulary", "ranges", "ref", "doc", "file",
    "sample", "test", "copy", "final", "v1", "v2", "r4", "snj", "jjh", "ass", "dhh",
})


def tokenize_component(comp: str) -> list[str]:
    return [t for t in _SPLIT_RE.split(comp) if t]


def filename_token(keys: Keys, parent_id: str, component: str) -> str:
    digest = keys.keyed("fname", parent_id, component)
    return "f-" + digest.hex()[:16]


def extract_filename_signals(rel_path: str) -> dict[str, list[str]]:
    """Harvest digit cores and non-boilerplate tokens from a relative path BEFORE renaming."""
    p = PurePosixPath(rel_path)
    stem_and_dirs = list(p.parts[:-1]) + [p.stem]
    digits: list[str] = []
    tokens: list[str] = []
    for part in stem_and_dirs:
        for tok in tokenize_component(part):
            if _DIGIT_CORE_RE.fullmatch(tok):
                digits.append(tok)
            elif tok.lower() not in _SAFE_BOILERPLATE and not tok.isdigit():
                tokens.append(tok)
    return {"digits": digits, "tokens": tokens}


def _is_sensitive_component(comp: str, df_counts: Counter[str], is_stem: bool = False) -> bool:
    if contains_local_id(comp):
        return True
    tokens = tokenize_component(comp)
    for tok in tokens:
        if _DIGIT_CORE_RE.search(tok):
            return True
        low = tok.lower()
        if low in _SAFE_BOILERPLATE or (tok.isdigit() and len(tok) <= 4):
            continue
        # High corpus DF (>= 3 files) without digits is boilerplate unless Title-Case name
        if df_counts[low] >= 3 and not tok[0].isupper():
            continue
        if is_stem or any(ord(ch) > 127 for ch in tok) or tok[0].isupper():
            return True
    return False


def _validate_component(comp: str) -> str:
    if not comp or comp in (".", "..") or "/" in comp or "\\" in comp:
        raise ValueError("invalid renamed path component")
    return comp


def allocate_output_paths(
    rel_paths: list[str],
    keys: Keys,
    resolution: Resolution | None = None,
) -> dict[str, str]:
    """Allocate deterministic, collision-free output relative paths in UTF-8 sorted order."""
    sorted_paths = sorted(rel_paths, key=lambda p: p.encode("utf-8"))

    # Compute corpus document frequency of path tokens
    df_counts: Counter[str] = Counter()
    for rp in sorted_paths:
        p = PurePosixPath(rp)
        seen_in_doc: set[str] = set()
        for part in list(p.parts[:-1]) + [p.stem]:
            for tok in tokenize_component(part):
                seen_in_doc.add(tok.lower())
        for tok in seen_in_doc:
            df_counts[tok] += 1

    # Pre-reserve unchanged sibling names per parent directory (case-folded)
    reserved_per_parent: dict[str, set[str]] = defaultdict(set)
    dir_mapping: dict[tuple[str, str], str] = {}

    # Pass 1: reserve untouched directory/file names
    # Only reserve names that will NOT be renamed
    dir_is_sensitive: dict[tuple[str, str], bool] = {}
    for rp in sorted_paths:
        p = PurePosixPath(rp)
        parent_in = ""
        for d in p.parts[:-1]:
            key_d = (parent_in, d)
            if key_d not in dir_is_sensitive:
                dir_is_sensitive[key_d] = _is_sensitive_component(d, df_counts, is_stem=False)
            parent_in = f"{parent_in}/{d}" if parent_in else d

    # Reserve untouched dirs and files
    for rp in sorted_paths:
        p = PurePosixPath(rp)
        parent_out = ""
        parent_in = ""
        all_dirs_clean = True
        for d in p.parts[:-1]:
            if dir_is_sensitive[(parent_in, d)]:
                all_dirs_clean = False
                break
            reserved_per_parent[parent_out].add(d.lower())
            parent_in = f"{parent_in}/{d}" if parent_in else d
            parent_out = f"{parent_out}/{d}" if parent_out else d
        if all_dirs_clean:
            stem_sens = _is_sensitive_component(p.stem, df_counts, is_stem=True)
            if not stem_sens:
                ext = p.suffix.lower() if p.suffix.lower() in _SAFE_EXTS else ".txt"
                reserved_per_parent[parent_out].add((p.stem + ext).lower())

    # Pass 2: allocate in UTF-8 sorted order, tracking actually emitted names separately
    allocated_per_parent: dict[str, set[str]] = defaultdict(set)

    def alloc_unique(parent_out: str, base_name: str, ext: str = "") -> str:
        taken = reserved_per_parent[parent_out] | allocated_per_parent[parent_out]
        cand = base_name + ext
        if cand.lower() not in taken:
            allocated_per_parent[parent_out].add(cand.lower())
            return _validate_component(cand)
        idx = 1
        while f"{base_name}_{idx}{ext}".lower() in taken:
            idx += 1
        final = f"{base_name}_{idx}{ext}"
        allocated_per_parent[parent_out].add(final.lower())
        return _validate_component(final)

    out_map: dict[str, str] = {}
    for rp in sorted_paths:
        p = PurePosixPath(rp)
        parent_in = ""
        parent_out = ""
        out_parts: list[str] = []

        for d in p.parts[:-1]:
            key_d = (parent_in, d)
            if key_d not in dir_mapping:
                if dir_is_sensitive[key_d]:
                    tok = filename_token(keys, parent_out or "root", d)
                    dir_mapping[key_d] = alloc_unique(parent_out, tok)
                else:
                    allocated_per_parent[parent_out].add(d.lower())
                    dir_mapping[key_d] = d
            mapped_d = dir_mapping[key_d]
            out_parts.append(mapped_d)
            parent_in = f"{parent_in}/{d}" if parent_in else d
            parent_out = f"{parent_out}/{mapped_d}" if parent_out else mapped_d

        ext = p.suffix.lower() if p.suffix.lower() in _SAFE_EXTS else ".txt"
        if _is_sensitive_component(p.stem, df_counts, is_stem=True):
            tok = filename_token(keys, parent_out or "root", p.stem)
            fname = alloc_unique(parent_out, tok, ext)
        else:
            cand = _validate_component(p.stem + ext)
            if cand.lower() not in allocated_per_parent[parent_out]:
                allocated_per_parent[parent_out].add(cand.lower())
                fname = cand
            else:
                fname = alloc_unique(parent_out, p.stem, ext)
        out_parts.append(fname)
        out_map[rp] = "/".join(out_parts)

    return out_map


def rename_headers_and_keys(pf: ParsedFile, spans: list[Span]) -> list[Edit]:
    """Produce REPLACE edits for sensitive @header fields and JSON key fields."""
    edits: list[Edit] = []
    span_by_field: dict[tuple[tuple[str, str], int], list[Span]] = defaultdict(list)
    for sp in spans:
        if not sp.protected:
            span_by_field[(sp.key, sp.field_idx)].append(sp)

    col_counter = 0
    key_counter = 0
    for rec in pf.records:
        for f in rec.fields:
            if f.vtype not in ("header", "key"):
                continue
            f_spans = span_by_field.get((rec.key, f.idx), [])
            if not f_spans:
                continue
            if f.vtype == "header":
                label = f"col_{col_counter}"
                col_counter += 1
            else:
                label = f"key_{key_counter}"
                key_counter += 1
            edits.append(Edit(rec.key, f.idx, 0, len(f.text), "REPLACE", label))
    return edits
