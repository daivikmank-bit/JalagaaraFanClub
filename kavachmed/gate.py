"""Read-back residual gate with bounded repair and write-fault detection (A10)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from kavachmed.contracts import Edit, Hit, LocMap, RecKey, Record
from kavachmed.security import StageError


def scan_attempt(
    path: Path,
    label: str,
    encoding: str,
    loc_map: LocMap,
    leak_keys: dict[str, Any],
    scan_a: Callable[..., list[Hit]],
    scan_b: Callable[..., list[Hit]],
) -> tuple[bytes, list[Hit]]:
    """Read actual disk bytes, decode strictly, and run Layer A + Layer B."""
    disk_bytes = path.read_bytes()
    text = disk_bytes.decode(encoding, errors="strict")
    ranges = loc_map.clinical_ranges()
    hits = list(scan_a(text, label, leak_keys, ranges))
    hits.extend(scan_b(text, label, ranges))
    return disk_bytes, hits


def run(
    dest_path: Path,
    out_bytes: bytes,
    rel_out: str,
    loc_map: LocMap,
    leak_keys: dict[str, Any],
    scan_a: Callable[[str, str, dict[str, Any], list[tuple[int, int, str]]], list[Hit]],
    scan_b: Callable[[str, str, list[tuple[int, int, str]]], list[Hit]],
    max_passes: int = 3,
) -> tuple[bool, dict[str, int]]:
    """Backward-compatible single-buffer gate wrapper."""
    ok, _bytes, _loc, counts = run_with_repair(
        dest_path,
        out_bytes,
        rel_out,
        loc_map,
        leak_keys,
        scan_a,
        scan_b,
        reproject_fn=None,
        records_by_key={},
        max_passes=max_passes,
    )
    return ok, counts


def run_with_repair(
    dest_path: Path,
    initial_bytes: bytes,
    rel_out: str,
    initial_loc_map: LocMap,
    leak_keys: dict[str, Any],
    scan_a: Callable[..., list[Hit]],
    scan_b: Callable[..., list[Hit]],
    reproject_fn: Callable[[list[Edit]], tuple[bytes, LocMap]] | None,
    records_by_key: dict[RecKey, Record],
    *,
    encoding: str = "utf-8",
    max_passes: int = 3,
    writer_fn: Callable[[Path, bytes], None] | None = None,
) -> tuple[bool, bytes | None, LocMap, dict[str, int]]:
    """Write -> close -> re-read rb -> decode -> scan A & B -> repair (<=3 passes)."""
    attempts = min(max(1, max_passes), 3)
    check_counts: dict[str, int] = {"A": 0, "B1": 0, "B2": 0, "B3": 0, "B4": 0}
    cur_bytes = initial_bytes
    cur_loc_map = initial_loc_map
    accumulated_repairs: list[Edit] = []
    seen_repairs: set[tuple[RecKey, int, int, int]] = set()

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(attempts):
        try:
            if writer_fn is not None:
                writer_fn(dest_path, cur_bytes)
            else:
                dest_path.write_bytes(cur_bytes)
            disk_bytes, hits = scan_attempt(
                dest_path, rel_out, encoding, cur_loc_map, leak_keys, scan_a, scan_b
            )
        except (OSError, UnicodeDecodeError, StageError, Exception):  # noqa: BLE001
            dest_path.unlink(missing_ok=True)
            return False, None, cur_loc_map, check_counts

        for h in hits:
            check_counts[h.check] = check_counts.get(h.check, 0) + 1

        # Detect write fault (disk bytes differ from projected bytes)
        if disk_bytes != cur_bytes:
            if attempt + 1 >= attempts:
                dest_path.unlink(missing_ok=True)
                return False, None, cur_loc_map, check_counts
            continue

        if not hits:
            return True, disk_bytes, cur_loc_map, check_counts

        if attempt + 1 >= attempts or reproject_fn is None:
            dest_path.unlink(missing_ok=True)
            return False, None, cur_loc_map, check_counts

        new_repairs: list[Edit] = []
        for h in hits:
            if h.channel == "path":
                dest_path.unlink(missing_ok=True)
                return False, None, cur_loc_map, check_counts
            entries = cur_loc_map.lookup(h.start, h.end)
            if not entries:
                dest_path.unlink(missing_ok=True)
                return False, None, cur_loc_map, check_counts
            for le in entries:
                if le.exempt_kind is not None:
                    dest_path.unlink(missing_ok=True)
                    return False, None, cur_loc_map, check_counts
                in_s, in_e = le.in_start, le.in_end
                # Verbatim segment (same length in and out): redact only the flagged token,
                # not the whole line/cell. Whole-segment repair wiped clinical text around a
                # single hit (e.g. a complete prescription line for one unknown word).
                rec0 = records_by_key.get(le.key)
                if (
                    rec0 is not None and le.field_idx < len(rec0.fields)
                    and le.out_end - le.out_start == le.in_end - le.in_start
                ):
                    os_, oe_ = max(h.start, le.out_start), min(h.end, le.out_end)
                    cand_s = le.in_start + (os_ - le.out_start)
                    cand_e = le.in_start + (oe_ - le.out_start)
                    disk_text = disk_bytes.decode(encoding, errors="strict")
                    if (
                        cand_e > cand_s
                        and rec0.fields[le.field_idx].text[cand_s:cand_e] == disk_text[os_:oe_]
                    ):
                        in_s, in_e = cand_s, cand_e
                rkey = (le.key, le.field_idx, in_s, in_e)
                # Escalate when an earlier, narrower repair of this spot did not clear the hit:
                # token -> whole segment -> whole field, before giving up to quarantine.
                if rkey in seen_repairs:
                    rkey = (le.key, le.field_idx, le.in_start, le.in_end)
                    in_s, in_e = le.in_start, le.in_end
                if rkey in seen_repairs and rec0 is not None and le.field_idx < len(rec0.fields):
                    in_s, in_e = 0, len(rec0.fields[le.field_idx].text)
                    rkey = (le.key, le.field_idx, in_s, in_e)
                if rkey not in seen_repairs:
                    seen_repairs.add(rkey)
                    rec = records_by_key.get(le.key)
                    if rec is None or le.field_idx >= len(rec.fields):
                        dest_path.unlink(missing_ok=True)
                        return False, None, cur_loc_map, check_counts
                    fld = rec.fields[le.field_idx]
                    if fld.vtype == "int":
                        rep_kind, rep_txt = "NUM_SURROGATE", "0"
                    elif fld.vtype == "float":
                        rep_kind, rep_txt = "NUM_SURROGATE", "0.0"
                    else:
                        rep_kind, rep_txt = "PLACEHOLDER", "[REDACTED]"
                    new_repairs.append(Edit(
                        key=le.key,
                        field_idx=le.field_idx,
                        start=in_s,
                        end=in_e,
                        kind=rep_kind,  # type: ignore[arg-type]
                        text=rep_txt,
                    ))

        if not new_repairs:
            dest_path.unlink(missing_ok=True)
            return False, None, cur_loc_map, check_counts

        accumulated_repairs.extend(new_repairs)
        try:
            cur_bytes, cur_loc_map = reproject_fn(accumulated_repairs)
        except (StageError, ValueError, UnicodeError, OSError):
            dest_path.unlink(missing_ok=True)
            return False, None, cur_loc_map, check_counts

    dest_path.unlink(missing_ok=True)
    return False, None, cur_loc_map, check_counts
