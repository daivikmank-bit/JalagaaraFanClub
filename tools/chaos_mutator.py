#!/usr/bin/env python3
"""Offline chaos dataset mutator (A13 severity 1-2, A14 severity 3 + LOHO).

Rewrites BOTH input files and truth.json consistently into a sibling directory
(never inside the kit/submission repo). Keeps dev/truth isolated from runtime.
"""
from __future__ import annotations

import argparse
import base64
import copy
import csv
import hashlib
import io
import json
import random
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

# Unseen synthetic name pools (completely disjoint from dev/)
_UNSEEN_FIRST_LATIN = (
    "Aarav", "Bhavna", "Chirag", "Devika", "Eshan", "Falguni",
    "Gaurav", "Himani", "Ishaan", "Jhanvi", "Kshitij", "Lavanya",
    "Mridul", "Niharika", "Omkar", "Pranavi", "Raghav", "Suhani",
    "Tanmay", "Urvashi", "Vedant", "Yashika", "Zoravar", "Anirudh",
)
_UNSEEN_LAST_LATIN = (
    "Agnihotri", "Bhardwaj", "Chaturvedi", "Deshmukh", "Gokhale", "Hegde",
    "Iyengar", "Jaiswal", "Kulkarni", "Mahajan", "Nambiar", "Oza",
    "Parikh", "Qureshi", "Rathore", "Sengupta", "Tripathi", "Upadhyay",
    "Vaidya", "Wagle", "Zaveri", "Bhattacharya", "Chatterjee", "Nair",
)
_UNSEEN_FIRST_DEVA = (
    "आरव", "भावना", "चिराग", "देविका", "ईशान", "फाल्गुनी",
    "गौरव", "हिमानी", "क्षितिज", "लावण्या", "मृदुल", "निहारिका",
    "ओंकार", "प्रणवी", "राघव", "सुहानी", "तन्मय", "उर्वशी",
)
_UNSEEN_LAST_DEVA = (
    "अग्निहोत्री", "भारद्वाज", "चतुर्वेदी", "देशमुख", "गोखले", "हेगड़े",
    "अयंगर", "जायसवाल", "कुलकर्णी", "महाजन", "नाम्बियार", "ओझा",
    "पारिख", "राठौर", "सेनगुप्ता", "त्रिपाठी", "उपाध्याय", "वैद्य",
)

_LABEL_SYNONYMS = {
    "Patient Name": ("Pt. Name", "Rogi ka naam", "Patient's Name"),
    "Name": ("Pt. Name", "Rogi ka naam"),
    "Father/Husband Name": ("Guardian", "Attendant", "Informant"),
    "S/o": ("C/O", "Guardian:"),
    "Address": ("Residential Addr", "Pata"),
}


def case_rng(seed: int, family: str, identity: str) -> random.Random:
    message = json.dumps([seed, family, identity], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(message).digest(), "big"))


def verify_file_inventory(case_root: Path, truth: dict[str, Any]) -> None:
    input_root = case_root / "input"
    disk_paths = {p.relative_to(input_root).as_posix() for p in input_root.rglob("*") if p.is_file()}
    assert disk_paths == set(truth["files"]), f"inventory mismatch: {disk_paths ^ set(truth['files'])}"
    for rel, entry in truth["files"].items():
        raw = (input_root / rel).read_bytes()
        assert entry["path"] == rel
        assert entry["size"] == len(raw)
        assert entry["sha256"] == hashlib.sha256(raw).hexdigest()


def _decode_with_hint(raw: bytes, enc_hint: str) -> tuple[str, str] | None:
    for enc in (enc_hint, "utf-8-sig", "utf-8", "utf-16", "cp1252"):
        if not enc:
            continue
        try:
            return raw.decode(enc, errors="strict"), enc
        except (UnicodeDecodeError, LookupError):
            continue
    return None


def _encode_with_hint(text: str, enc: str) -> bytes:
    return text.encode(enc, errors="strict")


def _build_name_sub_table(truth: dict[str, Any], seed: int) -> dict[tuple[str, str, str], str]:
    """Build deterministic unseen name substitutions keyed by (patient_or_person, script, surface)."""
    table: dict[tuple[str, str, str], str] = {}
    for rel in sorted(truth.get("files", {})):
        finfo = truth["files"][rel]
        for uloc in sorted(finfo.get("units", {})):
            unit = finfo["units"][uloc]
            patient_id = str(unit.get("patient") or uloc)
            for ent in unit.get("entities", []):
                if ent.get("type") not in ("PATIENT_NAME", "RELATIVE_NAME"):
                    continue
                surf = str(ent.get("surface", ""))
                script = str(ent.get("script", "latin"))
                role = str(ent.get("role", "patient"))
                if not surf or len(surf) < 2:
                    continue
                key = (f"{patient_id}:{role}", script, surf)
                if key in table:
                    continue
                rng = case_rng(seed, f"name:{script}:{role}", f"{patient_id}:{surf}")
                if script == "deva":
                    f_pool, l_pool = _UNSEEN_FIRST_DEVA, _UNSEEN_LAST_DEVA
                else:
                    f_pool, l_pool = _UNSEEN_FIRST_LATIN, _UNSEEN_LAST_LATIN
                parts = surf.split()
                if len(parts) >= 2:
                    table[key] = f"{rng.choice(f_pool)} {rng.choice(l_pool)}"
                else:
                    table[key] = rng.choice(f_pool)
    return table


def _apply_surface_replacements(
    text: str,
    pairs: list[tuple[str, str]],
    preserve_surfaces: list[str] | None = None,
) -> tuple[str, list[tuple[str, str]]]:
    """Replace exact non-overlapping surfaces from longest to shortest, never touching preserve spans."""
    applied: list[tuple[str, str]] = []
    spans: list[tuple[int, int, str, str]] = []
    occupied: list[tuple[int, int]] = []

    # Protect all clinical preserve spans from accidental substring replacement
    for ps in preserve_surfaces or ():
        if not ps:
            continue
        start = 0
        while True:
            idx = text.find(ps, start)
            if idx == -1:
                break
            occupied.append((idx, idx + len(ps)))
            start = idx + len(ps)

    for old, new in sorted(pairs, key=lambda p: (-len(p[0]), p[0])):
        if not old or old == new:
            continue
        start = 0
        while True:
            idx = text.find(old, start)
            if idx == -1:
                break
            end = idx + len(old)
            if not any(os < end and idx < oe for os, oe in occupied):
                spans.append((idx, end, old, new))
                occupied.append((idx, end))
                applied.append((old, new))
            start = end
    if not spans:
        return text, []
    spans.sort(key=lambda x: x[0], reverse=True)
    out = text
    for s, e, _old, new in spans:
        out = out[:s] + new + out[e:]
    return out, applied


def _mutate_csv_by_unit(
    text: str,
    delim: str,
    unit_pairs: dict[str, tuple[list[tuple[str, str]], list[str]]],
    label_pairs: list[tuple[str, str]],
) -> tuple[str, dict[str, list[tuple[str, str]]], list[tuple[str, str]]]:
    """Apply unit pairs strictly to each unit's 0-based data row:N (never across rows)."""
    rows = list(csv.reader(io.StringIO(text, newline=""), delimiter=delim, strict=True))
    applied_by_unit: dict[str, list[tuple[str, str]]] = {}
    applied_labels: list[tuple[str, str]] = []
    if label_pairs and rows:
        new_hdr: list[str] = []
        for cell in rows[0]:
            c2, app = _apply_surface_replacements(cell, label_pairs)
            new_hdr.append(c2)
            applied_labels.extend(app)
        rows[0] = new_hdr
    for data_idx in range(len(rows) - 1):
        uloc = f"row:{data_idx}"
        if uloc not in unit_pairs:
            continue
        pairs, preserves = unit_pairs[uloc]
        new_row: list[str] = []
        u_applied: list[tuple[str, str]] = []
        for cell in rows[data_idx + 1]:
            c2, app = _apply_surface_replacements(cell, pairs, preserves)
            new_row.append(c2)
            u_applied.extend(app)
        rows[data_idx + 1] = new_row
        if u_applied:
            applied_by_unit[uloc] = u_applied
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=delim, lineterminator="\n")
    w.writerows(rows)
    return buf.getvalue(), applied_by_unit, applied_labels


def _resolve_json_pointer(doc: Any, pointer: str) -> Any:
    if pointer in ("", "/"):
        return doc
    node = doc
    for part in pointer.split("/")[1:]:
        key = part.replace("~1", "/").replace("~0", "~")
        node = node[int(key)] if isinstance(node, list) else node[key]
    return node


def _mutate_json_node(
    node: Any,
    pairs: list[tuple[str, str]],
    preserves: list[str],
    applied: list[tuple[str, str]],
) -> Any:
    if isinstance(node, str):
        s2, app = _apply_surface_replacements(node, pairs, preserves)
        applied.extend(app)
        return s2
    if isinstance(node, list):
        return [_mutate_json_node(item, pairs, preserves, applied) for item in node]
    if isinstance(node, dict):
        return {k: _mutate_json_node(v, pairs, preserves, applied) for k, v in node.items()}
    return node


def _mutate_json_by_unit(
    text: str,
    unit_pairs: dict[str, tuple[list[tuple[str, str]], list[str]]],
) -> tuple[str, dict[str, list[tuple[str, str]]]]:
    """Apply unit pairs strictly within each annotated JSON pointer."""
    doc = json.loads(text)
    applied_by_unit: dict[str, list[tuple[str, str]]] = {}
    for uloc, (pairs, preserves) in unit_pairs.items():
        u_applied: list[tuple[str, str]] = []
        if uloc in ("", "/"):
            doc = _mutate_json_node(doc, pairs, preserves, u_applied)
        else:
            parts = uloc.split("/")[1:]
            parent = _resolve_json_pointer(doc, "/" + "/".join(parts[:-1])) if len(parts) > 1 else doc
            last = parts[-1].replace("~1", "/").replace("~0", "~")
            idx_or_key: int | str = int(last) if isinstance(parent, list) else last
            parent[idx_or_key] = _mutate_json_node(parent[idx_or_key], pairs, preserves, u_applied)
        if u_applied:
            applied_by_unit[uloc] = u_applied
    out = json.dumps(doc, ensure_ascii=False, indent=2)
    if text.endswith("\n"):
        out += "\n"
    return out, applied_by_unit


def _reformat_phone(surf: str, seed: int) -> str:
    digits = re.sub(r"\D", "", surf)
    if len(digits) < 10:
        return surf
    core = digits[-10:]
    rng = case_rng(seed, "phone_fmt", core)
    choice = rng.randint(0, 2)
    if choice == 0:
        return f"+91-{core[:5]}-{core[5:]}"
    if choice == 1:
        return f"0{core[:5]} {core[5:]}"
    return core


def _reformat_date(surf: str, seed: int) -> str:
    m = re.fullmatch(r"(\d{2})/(\d{2})/(\d{4})", surf.strip())
    if not m:
        return surf
    dd, mm, yyyy = m.groups()
    rng = case_rng(seed, "date_fmt", surf)
    return f"{yyyy}-{mm}-{dd}" if rng.randint(0, 1) == 0 else f"{dd}-{mm}-{yyyy}"


def mutate_case(
    kit_dir: Path,
    dest_dir: Path,
    seed: int,
    severity: int,
    holdout: str | None = None,
) -> dict[str, int]:
    kit_dir = kit_dir.resolve()
    dest_dir = dest_dir.resolve()
    if dest_dir == kit_dir or kit_dir in dest_dir.parents or dest_dir in kit_dir.parents:
        raise ValueError("dest must not overlap kit directory")
    if dest_dir.exists():
        raise ValueError(f"dest already exists: {dest_dir}")

    truth_src = kit_dir / "dev" / "truth" / "truth.json"
    input_src = kit_dir / "dev" / "input"
    truth = json.loads(truth_src.read_text(encoding="utf-8"))

    # Optional A14 LOHO filter
    if holdout:
        hl = holdout.lower()
        truth["files"] = {
            rel: info for rel, info in truth["files"].items()
            if hl in rel.lower()
        }
        if not truth["files"]:
            raise ValueError(f"holdout hospital '{holdout}' matched 0 files")

    staging = Path(tempfile.mkdtemp(prefix=f".staging_{dest_dir.name}_{seed}_", dir=str(dest_dir.parent)))
    (staging / "input").mkdir(parents=True)
    (staging / "truth").mkdir(parents=True)

    counts: Counter[str] = Counter()
    name_table = _build_name_sub_table(truth, seed) if severity >= 1 else {}

    new_files: dict[str, Any] = {}
    for rel in sorted(truth["files"]):
        finfo = copy.deepcopy(truth["files"][rel])
        src_path = input_src / rel
        raw = src_path.read_bytes()

        # Copy non-sanitizable / clean reference controls unchanged
        if finfo.get("expect") == "either" or finfo.get("clean"):
            out_path = staging / "input" / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(raw)
            finfo["sha256"] = hashlib.sha256(raw).hexdigest()
            finfo["size"] = len(raw)
            new_files[rel] = finfo
            counts["preserved_unchanged"] += 1
            continue

        decoded = _decode_with_hint(raw, str(finfo.get("encoding", "utf-8")))
        if decoded is None:
            out_path = staging / "input" / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(raw)
            new_files[rel] = finfo
            counts["skipped_undecodable"] += 1
            continue

        text, enc = decoded

        if severity >= 1:
            # Build per-unit surface replacement pairs from annotated content entities
            # and collect preserve surfaces to protect from accidental replacement
            unit_pairs: dict[str, tuple[list[tuple[str, str]], list[str]]] = {}
            global_pairs: list[tuple[str, str]] = []
            all_preserve: list[str] = []
            surface_map: dict[str, str] = {}
            for uloc in sorted(finfo.get("units", {})):
                unit = finfo["units"][uloc]
                pid = str(unit.get("patient") or uloc)
                u_pairs: list[tuple[str, str]] = []
                u_preserves: list[str] = []
                for ent in unit.get("entities", []):
                    if ent.get("channel") != "content":
                        continue
                    etype = ent.get("type")
                    surf = str(ent.get("surface", ""))
                    if etype in ("PATIENT_NAME", "RELATIVE_NAME"):
                        counts["names_eligible"] += 1
                        role = str(ent.get("role", "patient"))
                        script = str(ent.get("script", "latin"))
                        new_val = name_table.get((f"{pid}:{role}", script, surf))
                        if new_val:
                            u_pairs.append((surf, new_val))
                            global_pairs.append((surf, new_val))
                            surface_map[surf] = new_val
                    elif severity >= 2 and etype == "PHONE":
                        counts["phone_eligible"] += 1
                        new_val = _reformat_phone(surf, seed)
                        if new_val != surf:
                            u_pairs.append((surf, new_val))
                            global_pairs.append((surf, new_val))
                            surface_map[surf] = new_val
                        else:
                            counts["phone_skipped"] += 1
                    elif severity >= 2 and etype == "DOB":
                        counts["dob_eligible"] += 1
                        new_val = _reformat_date(surf, seed)
                        if new_val != surf:
                            u_pairs.append((surf, new_val))
                            global_pairs.append((surf, new_val))
                            surface_map[surf] = new_val
                        else:
                            counts["dob_skipped"] += 1
                # Collect preserve surfaces for this unit
                raw_preserves = unit.get("preserve", {})
                if isinstance(raw_preserves, dict):
                    preserve_items = raw_preserves.values()
                elif isinstance(raw_preserves, list):
                    preserve_items = raw_preserves
                else:
                    preserve_items = []
                for pres in preserve_items:
                    if isinstance(pres, dict):
                        val = pres.get("text") or pres.get("surface")
                        if val:
                            u_preserves.append(str(val))
                            all_preserve.append(str(val))
                    elif isinstance(pres, str):
                        u_preserves.append(pres)
                        all_preserve.append(pres)
                if u_pairs:
                    unit_pairs[uloc] = (u_pairs, u_preserves)

            # Label synonym swaps for prose/headers
            label_pairs: list[tuple[str, str]] = []
            for lbl, syns in _LABEL_SYNONYMS.items():
                if f"{lbl}:" in text:
                    rng = case_rng(seed, "label", f"{rel}:{lbl}")
                    label_pairs.append((f"{lbl}:", f"{rng.choice(syns)}:"))

            # Apply replacements channel-targeted, protecting preserve surfaces
            applied_by_unit: dict[str, list[tuple[str, str]]] = {}
            applied_flat: list[tuple[str, str]] = []
            if finfo.get("format") in ("csv", "tsv") and unit_pairs:
                delim_ch = "\t" if finfo.get("format") == "tsv" else ","
                text, applied_by_unit, applied_labels = _mutate_csv_by_unit(text, delim_ch, unit_pairs, label_pairs)
                for apps in applied_by_unit.values():
                    applied_flat.extend(apps)
                applied_flat.extend(applied_labels)
            elif finfo.get("format") in ("json", "fhir") and unit_pairs:
                try:
                    text, applied_by_unit = _mutate_json_by_unit(text, unit_pairs)
                    for apps in applied_by_unit.values():
                        applied_flat.extend(apps)
                except (ValueError, KeyError, IndexError, TypeError):
                    text, applied_flat = _apply_surface_replacements(
                        text, global_pairs + label_pairs, preserve_surfaces=all_preserve
                    )
            else:
                text, applied_flat = _apply_surface_replacements(
                    text, global_pairs + label_pairs, preserve_surfaces=all_preserve
                )

            if applied_flat:
                counts["names_and_labels_mutated"] += len(applied_flat)
                global_applied_dict = dict(applied_flat)
                for uloc, unit in finfo.get("units", {}).items():
                    unit_applied = dict(applied_by_unit[uloc]) if uloc in applied_by_unit else global_applied_dict
                    for ent in unit.get("entities", []):
                        if ent.get("channel") == "content" and ent.get("surface") in unit_applied:
                            ent["surface"] = unit_applied[ent["surface"]]
                            etype = ent.get("type")
                            if etype in ("PATIENT_NAME", "RELATIVE_NAME"):
                                counts["names_applied"] += 1
                            elif etype == "PHONE":
                                counts["phone_applied"] += 1
                            elif etype == "DOB":
                                counts["dob_applied"] += 1
            counts["names_skipped"] = max(0, counts["names_eligible"] - counts["names_applied"])

        if severity >= 2 and finfo.get("format") == "text":
            # Inject canary + unlabelled name narrative on eligible text files
            rng = case_rng(seed, "sev2_txt", rel)
            units = finfo.get("units", {})
            if "doc" in units:
                counts["canary_eligible"] += 1
                doc_unit = units["doc"]
                canary_val = f"CANARY-{seed:04d}-{hashlib.sha256(f'{seed}:{rel}'.encode()).hexdigest()[:8].upper()}"
                unlabelled_name = f"{rng.choice(_UNSEEN_FIRST_LATIN)} {rng.choice(_UNSEEN_LAST_LATIN)}"
                sep = "" if text.endswith("\n") else "\n"
                text = f"{text}{sep}Note: {unlabelled_name} ({canary_val}) reviewed.\n"
                doc_unit.setdefault("entities", []).append({
                    "type": "PATIENT_NAME", "surface": unlabelled_name,
                    "person": doc_unit.get("patient", ""), "role": "patient",
                    "script": "latin", "channel": "content", "match": "exact",
                    "key": False, "critical": True,
                })
                doc_unit["entities"].append({
                    "type": "CANARY", "surface": canary_val,
                    "person": doc_unit.get("patient", ""), "role": "patient",
                    "script": "latin", "channel": "content", "match": "exact",
                    "key": True, "critical": True,
                })
                truth.setdefault("canaries", []).append({
                    "surface": canary_val, "file": rel, "loc": "doc",
                })
                counts["canaries_injected"] += 1
                counts["canary_applied"] += 1
                counts["unlabelled_narratives"] += 1

        if severity >= 2 and finfo.get("format") in ("csv", "tsv"):
            counts["csv_perm_eligible"] += 1
            delim = "\t" if finfo.get("format") == "tsv" else ","
            try:
                rows = list(csv.reader(io.StringIO(text, newline=""), delimiter=delim, strict=True))
                if len(rows) >= 2 and len(rows[0]) >= 3 and all(len(r) == len(rows[0]) for r in rows):
                    perm = list(range(len(rows[0])))
                    case_rng(seed, "csv_perm", rel).shuffle(perm)
                    buf = io.StringIO()
                    w = csv.writer(buf, delimiter=delim, lineterminator="\n")
                    for r in rows:
                        w.writerow([r[idx] for idx in perm])
                    text = buf.getvalue()
                    counts["csv_reordered"] += 1
                    counts["csv_perm_applied"] += 1
                else:
                    counts["csv_perm_skipped"] += 1
            except csv.Error:
                counts["skipped_csv_perm"] += 1
                counts["csv_perm_skipped"] += 1

        out_bytes = _encode_with_hint(text, enc)
        out_path = staging / "input" / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(out_bytes)
        finfo["sha256"] = hashlib.sha256(out_bytes).hexdigest()
        finfo["size"] = len(out_bytes)
        new_files[rel] = finfo

    truth["files"] = new_files
    truth["chaos"] = True
    truth["seed"] = seed
    (staging / "truth" / "truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")

    verify_file_inventory(staging, truth)
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(staging), str(dest_dir))
    return dict(sorted(counts.items()))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="KavachMed chaos dataset mutator (A13/A14)")
    ap.add_argument("--kit", required=True, help="Kit root containing dev/")
    ap.add_argument("--dest", required=True, help="Destination chaos case directory (outside repo)")
    ap.add_argument("--seed", type=int, default=104)
    ap.add_argument("--severity", type=int, choices=(1, 2, 3), default=2)
    ap.add_argument("--holdout", default=None, help="Optional hospital substring for LOHO (A14)")
    args = ap.parse_args(argv)

    counts = mutate_case(Path(args.kit), Path(args.dest), args.seed, args.severity, args.holdout)
    print("CHAOS OK " + " ".join(f"{k}={v}" for k, v in counts.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
