"""Iterative RFC 8259 JSON parser, field assigner, and token-replay projector (A6).

Zero recursion in both parsing and serialization so deep_export.json (1,501+ levels)
and depth=4,000 round-trip cleanly without touching sys.setrecursionlimit.
"""
from __future__ import annotations

import base64
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from kavachmed import credentials
from kavachmed.contracts import Edit, Field, FileEntry, LocEntry, LocMap, ParsedFile, RecKey, Record, Span
from kavachmed.security import StageError
from kavachmed.urm import apply_field_edits_with_loc

_JSON_WS = frozenset(" \t\r\n")
_JSON_DELIMS = frozenset(" \t\r\n,}]")
_JSON_NUM_RE = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
_B64_KEY_HINTS = ("base64", "b64", "content", "attachment", "payload", "form_data")
_STRING_DECODER = json.JSONDecoder()


def pointer_segment(key: str) -> str:
    return key.replace("~", "~0").replace("/", "~1")


@dataclass
class _Token:
    start: int
    end: int
    vtype: str          # "key" | "str" | "int" | "float" | "b64"
    value: str
    top_key: str | None
    arr_idx: int | None
    rec_loc: str
    field_idx: int = -1
    obj_id: int = 0
    key_hint: str = ""


def _skip_ws(s: str, i: int, n: int) -> int:
    while i < n and s[i] in _JSON_WS:
        i += 1
    return i


def _scan_string(s: str, i: int, n: int) -> tuple[str, int]:
    """Consume one JSON string token starting at `i` ('\"') using stdlib scalar decoder."""
    try:
        val, end = _STRING_DECODER.raw_decode(s, i)
    except json.JSONDecodeError as exc:
        raise StageError("parse", "E_INVALID_JSON") from exc
    if not isinstance(val, str):
        raise StageError("parse", "E_INVALID_JSON")
    return val, end


def _tokenize_iterative(
    s: str, max_depth: int
) -> tuple[list[_Token], bool, dict[str, list[bool]], list[bool], list[tuple[str | None, int, int, int]], dict[int, str], dict[int, int]]:
    """Iterative RFC 8259 parser.

    Returns (tokens, is_bundle, top_array_dict_flags, root_array_dict_flags, candidate_obj_records).
    Each candidate_obj_record is (top_key_or_None, arr_idx, start_pos, end_pos).
    """
    n = len(s)
    i = _skip_ws(s, 0, n)
    if i >= n:
        raise StageError("parse", "E_INVALID_JSON")

    tokens: list[_Token] = []
    # Frame: [kind ("obj"|"arr"), state, obj_id, top_key, arr_idx, elem_count, start_pos]
    stack: list[list[Any]] = []
    owners: dict[int, str] = {}
    starts: dict[int, int] = {}
    next_obj_id = 1
    is_bundle = False
    top_array_dict_flags: dict[str, list[bool]] = {}
    root_array_dict_flags: list[bool] = []
    candidate_obj_records: list[tuple[str | None, int, int, int]] = []
    root_done = False

    def push_container(ch: str, pos: int) -> int:
        nonlocal next_obj_id
        depth = len(stack) + 1
        if depth > max_depth:
            raise StageError("parse", "E_DEPTH")
        top_key = None
        arr_idx = None
        if stack:
            parent = stack[-1]
            top_key = parent[3]
            arr_idx = parent[4]
            if len(stack) == 1 and parent[0] == "obj" and top_key is not None and ch == "[":
                top_array_dict_flags.setdefault(top_key, [])
            if len(stack) == 1 and parent[0] == "arr":
                root_array_dict_flags.append(ch == "{")
                arr_idx = parent[5]
            elif len(stack) == 2 and parent[0] == "arr" and parent[3] in top_array_dict_flags:
                top_array_dict_flags[parent[3]].append(ch == "{")
        oid = next_obj_id
        next_obj_id += 1
        starts[oid] = pos
        if stack:
            parent_id = stack[-1][2]
            owners[oid] = (tokens[-1].value if tokens and tokens[-1].vtype == "key" and tokens[-1].obj_id == parent_id else owners.get(parent_id, ""))
        stack.append(["obj" if ch == "{" else "arr", "START", oid, top_key, arr_idx, 0, pos])
        return pos + 1

    def pop_container(end_pos: int) -> None:
        nonlocal root_done
        popped = stack.pop()
        if popped[0] == "obj":
            if len(stack) == 1 and stack[0][0] == "arr":
                candidate_obj_records.append((None, int(popped[4]), int(popped[6]), end_pos))
            elif len(stack) == 2 and stack[0][0] == "obj" and stack[1][0] == "arr" and popped[3] is not None:
                candidate_obj_records.append((str(popped[3]), int(popped[4]), int(popped[6]), end_pos))
        if not stack:
            root_done = True
            return
        stack[-1][1] = "COMMA_OR_END"

    while True:
        i = _skip_ws(s, i, n)
        if not stack:
            if root_done:
                if i != n:
                    raise StageError("parse", "E_INVALID_JSON")
                break
            if i >= n:
                raise StageError("parse", "E_INVALID_JSON")
            ch = s[i]
            if ch in "{[":
                i = push_container(ch, i)
                continue
            if ch == '"':
                val, end = _scan_string(s, i, n)
                tokens.append(_Token(i, end, "str", val, None, None, "doc", obj_id=0))
                i = end
                root_done = True
                continue
            if s.startswith("true", i) and (i + 4 == n or s[i + 4] in _JSON_DELIMS):
                i += 4
                root_done = True
                continue
            if s.startswith("false", i) and (i + 5 == n or s[i + 5] in _JSON_DELIMS):
                i += 5
                root_done = True
                continue
            if s.startswith("null", i) and (i + 4 == n or s[i + 4] in _JSON_DELIMS):
                i += 4
                root_done = True
                continue
            m = _JSON_NUM_RE.match(s, i)
            if m and (m.end() == n or s[m.end()] in _JSON_DELIMS):
                raw_num = m.group(0)
                vtype = "float" if ("." in raw_num or "e" in raw_num or "E" in raw_num) else "int"
                tokens.append(_Token(i, m.end(), vtype, raw_num, None, None, "doc", obj_id=0))
                i = m.end()
                root_done = True
                continue
            raise StageError("parse", "E_INVALID_JSON")

        if i >= n:
            raise StageError("parse", "E_INVALID_JSON")

        frame = stack[-1]
        kind, state, oid, top_key, arr_idx, elem_count, _start_pos = frame
        ch = s[i]

        if kind == "obj":
            if state == "START":
                if ch == "}":
                    pop_container(i + 1)
                    i += 1
                    continue
                if ch != '"':
                    raise StageError("parse", "E_INVALID_JSON")
                kval, end = _scan_string(s, i, n)
                tk = kval if len(stack) == 1 else top_key
                frame[3] = tk
                tokens.append(_Token(i, end, "key", kval, tk, arr_idx, "doc", obj_id=oid))
                frame[1] = "COLON"
                i = end
                continue
            if state == "KEY_NEXT":
                if ch != '"':
                    raise StageError("parse", "E_INVALID_JSON")
                kval, end = _scan_string(s, i, n)
                tk = kval if len(stack) == 1 else top_key
                frame[3] = tk
                tokens.append(_Token(i, end, "key", kval, tk, arr_idx, "doc", obj_id=oid))
                frame[1] = "COLON"
                i = end
                continue
            if state == "COLON":
                if ch != ":":
                    raise StageError("parse", "E_INVALID_JSON")
                frame[1] = "VALUE"
                i += 1
                continue
            if state == "COMMA_OR_END":
                if ch == ",":
                    frame[1] = "KEY_NEXT"
                    i += 1
                    continue
                if ch == "}":
                    pop_container(i + 1)
                    i += 1
                    continue
                raise StageError("parse", "E_INVALID_JSON")
            if len(stack) == 1 and top_key == "resourceType" and ch == '"':
                val, end = _scan_string(s, i, n)
                if val == "Bundle":
                    is_bundle = True
                tokens.append(_Token(i, end, "str", val, top_key, arr_idx, "doc", obj_id=oid, key_hint=top_key or ""))
                i = end
                pop_or_cont = stack[-1]
                pop_or_cont[1] = "COMMA_OR_END"
                continue

        elif kind == "arr":
            if state == "START" and ch == "]":
                pop_container(i + 1)
                i += 1
                continue
            if state == "COMMA_OR_END":
                if ch == ",":
                    frame[1] = "VALUE"
                    frame[5] += 1
                    if len(stack) == 2 and stack[0][0] == "obj":
                        frame[4] = frame[5]
                    i += 1
                    continue
                if ch == "]":
                    pop_container(i + 1)
                    i += 1
                    continue
                raise StageError("parse", "E_INVALID_JSON")
            if len(stack) == 2 and stack[0][0] == "obj" and state == "START":
                frame[4] = 0
                arr_idx = 0

        last_key = ""
        if tokens and tokens[-1].vtype == "key" and tokens[-1].obj_id == oid:
            last_key = tokens[-1].value

        role = credentials.json_role(owners.get(oid, ""), last_key)
        if role and role != "attachment_meta" and ch != '"' and not (role == "attachment" and ch in "{["):
            raise StageError("parse", "E_ATTACHMENT" if role.startswith("attachment") else "E_FIELD_TYPE")
        if ch in "{[":
            i = push_container(ch, i)
            continue
        if ch == '"':
            val, end = _scan_string(s, i, n)
            if len(stack) == 1 and frame[0] == "arr":
                root_array_dict_flags.append(False)
            elif len(stack) == 2 and frame[0] == "arr" and frame[3] in top_array_dict_flags:
                top_array_dict_flags[frame[3]].append(False)
            tokens.append(_Token(i, end, "str", val, top_key, arr_idx, "doc", obj_id=oid, key_hint=last_key))
            i = end
            stack[-1][1] = "COMMA_OR_END"
            continue
        if s.startswith("true", i) and (i + 4 == n or s[i + 4] in _JSON_DELIMS):
            if len(stack) == 1 and frame[0] == "arr":
                root_array_dict_flags.append(False)
            elif len(stack) == 2 and frame[0] == "arr" and frame[3] in top_array_dict_flags:
                top_array_dict_flags[frame[3]].append(False)
            i += 4
            stack[-1][1] = "COMMA_OR_END"
            continue
        if s.startswith("false", i) and (i + 5 == n or s[i + 5] in _JSON_DELIMS):
            if len(stack) == 1 and frame[0] == "arr":
                root_array_dict_flags.append(False)
            elif len(stack) == 2 and frame[0] == "arr" and frame[3] in top_array_dict_flags:
                top_array_dict_flags[frame[3]].append(False)
            i += 5
            stack[-1][1] = "COMMA_OR_END"
            continue
        if s.startswith("null", i) and (i + 4 == n or s[i + 4] in _JSON_DELIMS):
            if len(stack) == 1 and frame[0] == "arr":
                root_array_dict_flags.append(False)
            elif len(stack) == 2 and frame[0] == "arr" and frame[3] in top_array_dict_flags:
                top_array_dict_flags[frame[3]].append(False)
            i += 4
            stack[-1][1] = "COMMA_OR_END"
            continue
        m = _JSON_NUM_RE.match(s, i)
        if m and (m.end() == n or s[m.end()] in _JSON_DELIMS):
            if len(stack) == 1 and frame[0] == "arr":
                root_array_dict_flags.append(False)
            elif len(stack) == 2 and frame[0] == "arr" and frame[3] in top_array_dict_flags:
                top_array_dict_flags[frame[3]].append(False)
            raw_num = m.group(0)
            vtype = "float" if ("." in raw_num or "e" in raw_num or "E" in raw_num) else "int"
            tokens.append(_Token(i, m.end(), vtype, raw_num, top_key, arr_idx, "doc", obj_id=oid, key_hint=last_key))
            i = m.end()
            stack[-1][1] = "COMMA_OR_END"
            continue
        raise StageError("parse", "E_INVALID_JSON")

    return tokens, is_bundle, top_array_dict_flags, root_array_dict_flags, candidate_obj_records, owners, starts


_PATIENT_LEVEL_KEYS = frozenset({
    "patient", "patient_name", "patientname", "name", "uhid", "mrn", "patient_id", "patientid",
    "abha", "abha_number", "aadhaar", "aadhar_no", "mobile", "phone",
})


def parse(entry: FileEntry, max_depth: int = 4000) -> ParsedFile:
    assert entry.text is not None
    src = entry.text
    tokens, is_bundle, top_array_dict_flags, root_array_dict_flags, cand_recs, owners, starts = _tokenize_iterative(src, max_depth)

    record_array_keys = {
        k for k, flags in top_array_dict_flags.items()
        if not is_bundle and flags and all(flags)
    }
    is_root_record_array = bool(root_array_dict_flags) and all(root_array_dict_flags)

    # A single-patient document (top-level "patient"/"name"/"uhid" next to e.g. "panels")
    # is ONE record ("doc", PRD §6.3); only exports without patient-level keys at the top
    # (OPD {"_export_meta", "visits": [...]}) are record arrays.
    top_level_keys = {
        str(tok.top_key).casefold() for tok in tokens
        if tok.top_key is not None and tok.top_key not in record_array_keys
    }
    if record_array_keys and top_level_keys & _PATIENT_LEVEL_KEYS:
        record_array_keys = set()

    records_by_loc: dict[str, Record] = {}
    obj_bounds: dict[str, tuple[int, int]] = {}

    def get_rec(loc: str) -> Record:
        if loc not in records_by_loc:
            records_by_loc[loc] = Record(
                key=(entry.rel_path, loc),
                fields=[],
                meta={"encoding": entry.encoding},
            )
        return records_by_loc[loc]

    # Pre-create public records from object boundaries so empty {} items keep their index
    if is_root_record_array:
        for tk, aidx, s_pos, e_pos in sorted(cand_recs, key=lambda x: x[1]):
            if tk is None:
                loc = f"/{aidx}"
                get_rec(loc)
                obj_bounds[loc] = (s_pos, e_pos)
    elif record_array_keys:
        for tk, aidx, s_pos, e_pos in sorted(cand_recs, key=lambda x: (str(x[0]), x[1])):
            if tk in record_array_keys:
                loc = f"/{pointer_segment(tk)}/{aidx}"
                get_rec(loc)
                obj_bounds[loc] = (s_pos, e_pos)

    budget = credentials.AttachmentBudget()
    for tok in tokens:
        if is_root_record_array and tok.arr_idx is not None:
            loc = f"/{tok.arr_idx}"
        elif tok.top_key in record_array_keys and tok.arr_idx is not None:
            loc = f"/{pointer_segment(tok.top_key)}/{tok.arr_idx}"
        elif record_array_keys:
            loc = "@meta"
        else:
            loc = "doc"
        tok.rec_loc = loc
        rec = get_rec(loc)

        vtype = tok.vtype
        field_text = tok.value
        tok.field_idx = len(rec.fields)
        role = credentials.json_role(owners.get(tok.obj_id, ""), tok.key_hint)
        if role:
            rec.meta.setdefault("release_roles", {})[tok.field_idx] = role
        if not is_bundle and role == "attachment" and vtype == "str":
            try:
                field_text = budget.decode(tok.value)
            except StageError:
                # An optional untyped string attachment can be cleared without changing its schema.
                if credentials.normalized(tok.key_hint) != "attachment":
                    raise
                rec.meta["release_roles"][tok.field_idx] = "attachment_meta"
            else:
                vtype = tok.vtype = "b64"
                tok.value = field_text
        if tok.key_hint and obj_bounds.get(loc, (None,))[0] == starts.get(tok.obj_id):
            rec.meta.setdefault("direct_fields", set()).add(tok.field_idx)
        elif tok.key_hint and owners.get(tok.obj_id, "").casefold() in {"pt", "patient", "beneficiary", "member"}:
            rec.meta.setdefault("patient_subfields", set()).add(tok.field_idx)
        rec.fields.append(Field(
            idx=tok.field_idx,
            path=f"{loc}/{pointer_segment(tok.key_hint or tok.vtype)}",
            text=field_text,
            vtype=vtype,  # type: ignore[arg-type]
        ))

    if not records_by_loc:
        get_rec("doc")

    records = list(records_by_loc.values())

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

        # Track records that already have a "patient_ref" key so we never emit duplicates
        recs_with_patient_ref: set[RecKey] = {
            (entry.rel_path, tok.rec_loc)
            for tok in tokens
            if tok.vtype == "key" and tok.value == "patient_ref"
        }

        # Build insertion points before closing '}' of linked records lacking a PSN edit
        insert_before: dict[int, list[tuple[int, str]]] = defaultdict(list)
        for loc, (s_pos, e_pos) in obj_bounds.items():
            rk: RecKey = (entry.rel_path, loc)
            psn = (psn_map or {}).get(rk)
            if psn is not None and rk not in psn_edited_recs and rk not in recs_with_patient_ref:
                is_empty_obj = src[s_pos + 1:e_pos - 1].strip() == ""
                snippet = f'"patient_ref":{json.dumps(psn)}' if is_empty_obj else f',"patient_ref":{json.dumps(psn)}'
                insert_before[e_pos - 1].append((s_pos, snippet))
        if not obj_bounds and (psn_map or {}).get((entry.rel_path, "doc")) is not None:
            doc_rk: RecKey = (entry.rel_path, "doc")
            if doc_rk not in psn_edited_recs and doc_rk not in recs_with_patient_ref and src.lstrip().startswith("{"):
                rbrace = src.rfind("}")
                if rbrace != -1:
                    psn = (psn_map or {})[doc_rk]
                    is_empty_obj = src[src.find("{") + 1:rbrace].strip() == ""
                    snippet = f'"patient_ref":{json.dumps(psn)}' if is_empty_obj else f',"patient_ref":{json.dumps(psn)}'
                    insert_before[rbrace].append((0, snippet))

        # Reserve ALL unchanged keys in each object before allocating edited keys
        reserved_keys_per_obj: dict[int, set[str]] = defaultdict(set)
        for tok in tokens:
            if tok.vtype == "key":
                rk = (entry.rel_path, tok.rec_loc)
                if not by_field.get((rk, tok.field_idx)):
                    reserved_keys_per_obj[tok.obj_id].add(tok.value)

        sorted_inserts = sorted(insert_before.items())
        ins_idx = 0

        def _emit_gap(gap_start: int, gap_end: int) -> None:
            nonlocal out_cur, ins_idx
            pos = gap_start
            while ins_idx < len(sorted_inserts) and sorted_inserts[ins_idx][0] <= gap_end:
                ins_pos, ins_list = sorted_inserts[ins_idx]
                if ins_pos >= pos:
                    if ins_pos > pos:
                        seg = src[pos:ins_pos]
                        out_parts.append(seg)
                        out_cur += len(seg)
                    for _, snip in ins_list:
                        out_parts.append(snip)
                        out_cur += len(snip)
                    pos = ins_pos
                ins_idx += 1
            if pos < gap_end:
                seg = src[pos:gap_end]
                out_parts.append(seg)
                out_cur += len(seg)

        out_parts: list[str] = []
        loc_entries: list[LocEntry] = []
        src_cur = 0
        out_cur = 0

        for tok in tokens:
            if tok.start > src_cur:
                _emit_gap(src_cur, tok.start)

            rec_key: RecKey = (entry.rel_path, tok.rec_loc)
            f_edits = by_field.get((rec_key, tok.field_idx), [])
            f_shields = prot_by_field.get((rec_key, tok.field_idx))

            if not f_edits and not f_shields:
                raw_tok = src[tok.start:tok.end]
                out_parts.append(raw_tok)
                if tok.vtype in ("str", "key", "b64"):
                    loc_entries.append(LocEntry(out_cur + 1, out_cur + len(raw_tok) - 1, rec_key, tok.field_idx, 0, len(tok.value)))
                else:
                    loc_entries.append(LocEntry(out_cur, out_cur + len(raw_tok), rec_key, tok.field_idx, 0, len(tok.value)))
                out_cur += len(raw_tok)
                src_cur = tok.end
                continue

            if tok.vtype in ("int", "float"):
                if any(ed.kind == "BAND_AGE" for ed in f_edits):
                    band_str, _ = apply_field_edits_with_loc(tok.value, f_edits, rec_key, tok.field_idx, 0)
                    dumped = json.dumps(band_str, ensure_ascii=False)
                    out_parts.append(dumped)
                    loc_entries.append(LocEntry(out_cur + 1, out_cur + len(dumped) - 1, rec_key, tok.field_idx, 0, len(tok.value)))
                    out_cur += len(dumped)
                else:
                    new_num, _ = apply_field_edits_with_loc(tok.value, f_edits, rec_key, tok.field_idx, 0)
                    if not _JSON_NUM_RE.fullmatch(new_num):
                        raise StageError("edits", "E_IO")
                    out_parts.append(new_num)
                    loc_entries.append(LocEntry(out_cur, out_cur + len(new_num), rec_key, tok.field_idx, 0, len(tok.value)))
                    out_cur += len(new_num)
            elif tok.vtype == "b64":
                rep_plain, _ = apply_field_edits_with_loc(tok.value, f_edits, rec_key, tok.field_idx, 0)
                b64_out = base64.b64encode(rep_plain.encode("utf-8")).decode("ascii")
                dumped = json.dumps(b64_out, ensure_ascii=False)
                out_parts.append(dumped)
                loc_entries.append(LocEntry(out_cur + 1, out_cur + len(dumped) - 1, rec_key, tok.field_idx, 0, len(tok.value)))
                out_cur += len(dumped)
            else:
                rep_str, raw_locs = apply_field_edits_with_loc(
                    tok.value, f_edits, rec_key, tok.field_idx, 0, protected_spans=f_shields
                )
                if tok.vtype == "key":
                    used = reserved_keys_per_obj[tok.obj_id]
                    if rep_str in used:
                        suffix = 1
                        while f"{rep_str}_{suffix}" in used:
                            suffix += 1
                        rep_str = f"{rep_str}_{suffix}"
                    used.add(rep_str)
                dumped = json.dumps(rep_str, ensure_ascii=False)
                # Map each char of rep_str -> escaped offset in dumped[1:-1]
                rep_to_esc = [0] * (len(rep_str) + 1)
                epos = 1
                for idx_c, ch in enumerate(rep_str):
                    rep_to_esc[idx_c] = epos
                    epos += len(json.dumps(ch, ensure_ascii=False)) - 2
                rep_to_esc[len(rep_str)] = epos
                out_parts.append(dumped)
                for le in raw_locs:
                    loc_entries.append(LocEntry(
                        out_cur + rep_to_esc[le.out_start],
                        out_cur + rep_to_esc[le.out_end],
                        le.key,
                        le.field_idx,
                        le.in_start,
                        le.in_end,
                        le.exempt_kind,
                    ))
                out_cur += len(dumped)

            src_cur = tok.end

        if src_cur < len(src):
            _emit_gap(src_cur, len(src))

        return "".join(out_parts).encode("utf-8"), LocMap(loc_entries)

    fmt = "fhir" if is_bundle else "json"
    return ParsedFile(
        rel_path=entry.rel_path,
        sha256=entry.sha256,
        fmt=fmt,
        encoding=entry.encoding or "utf-8",
        newline="\n",
        records=records,
        raw_bytes=entry.raw_bytes,
        projector=projector,
        extra={"obj_bounds": obj_bounds},
    )
