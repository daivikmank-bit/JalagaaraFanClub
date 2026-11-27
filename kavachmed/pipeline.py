"""End-to-end KavachMed pipeline orchestrator."""
from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kavachmed import (
    adapters,
    dates,
    detect,
    edits,
    emit,
    evidence,
    gate,
    inventory,
    linkage,
    privacy,
    propagate,
    rename,
    residual_a,
    residual_b,
    shield,
    typing as col_typing,
)
from kavachmed.adapters import fhir
from kavachmed.privacy import spans as priv_spans
from kavachmed.contracts import Evidence, LocMap, ParsedFile, RecKey, Span
from kavachmed.linkage.filename import FilenameSignal
from kavachmed.security import Keys, StageError, log_counts

PARSE_REASON: dict[str, str] = {
    "E_INVALID_JSON": "invalid_json",
    "E_DEPTH": "excessive_depth",
    "E_CSV": "validation_failed",
    "E_MIXED_ENC": "mixed_encoding",
    "E_UNDECODABLE": "undecodable",
    "E_EMPTY": "empty_file",
    "E_UNSUPPORTED": "unsupported_format",
    "E_IO": "validation_failed",
}

_INTERNAL_LOC_PREFIXES = ("@",)


def _is_public_loc(loc: str) -> bool:
    return not loc.startswith(_INTERNAL_LOC_PREFIXES)


def _drop_path_boilerplate_keys(leak_keys: Any, rel_paths: list[str]) -> Any:
    """Drop Layer A name keys that are input-path boilerplate (hospital names, "export").

    A token appearing in the paths of >= 2 different input files names a system or a
    folder, not a person; as a key it would flag the renamed output path of every file.
    """
    names = getattr(leak_keys, "name_tokens", None)
    if not names:
        return leak_keys
    import dataclasses
    import re as _re
    df: Counter[str] = Counter()
    for rp in rel_paths:
        df.update({t.casefold() for t in _re.findall(r"[^\W_\d]+", rp)})
    keep = tuple(n for n in names if df.get(n, 0) < 2)
    if len(keep) == len(names):
        return leak_keys
    return dataclasses.replace(leak_keys, name_tokens=keep)


@dataclass(slots=True)
class _FileMeta:
    abs_path: Path
    rel_path: str
    sha256: str
    fmt: str
    encoding: str
    extra: dict[str, Any] = field(default_factory=dict)


def _preflight_output_ownership(
    output_dir: Path,
    san_dir: Path,
    expected_san_paths: set[Path],
) -> set[Path]:
    """Validate ownership of any existing files in sanitized/ before writing."""
    if not san_dir.is_dir():
        return set()
    existing_files = [p.resolve() for p in san_dir.rglob("*") if p.is_file()]
    if not existing_files:
        return set()
    prev_report_path = output_dir / "report.json"
    if not prev_report_path.is_file():
        # Any pre-existing file in sanitized/ without a valid prior manifest is unowned.
        raise StageError("emit", "E_IO")
    try:
        prev = json.loads(prev_report_path.read_text(encoding="utf-8"))
        prev_files = prev.get("files")
        if prev.get("schema") != "raksha-report/1" or not isinstance(prev_files, list):
            raise ValueError("invalid_manifest")
        prev_owned = {
            (output_dir / f["output"]).resolve()
            for f in prev_files
            if isinstance(f, dict) and f.get("output")
        }
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        raise StageError("emit", "E_IO") from exc

    # Fail closed before any mutation if a planned output collides with an unowned file.
    for p in existing_files:
        if p in expected_san_paths and p not in prev_owned:
            raise StageError("emit", "E_IO")
    return prev_owned


def _cleanup_stale_outputs(
    san_dir: Path,
    prev_owned: set[Path],
    expected_san_paths: set[Path],
) -> None:
    """Remove only stale files owned by the prior manifest; preserve unowned files."""
    if not san_dir.is_dir():
        return
    san_resolved = san_dir.resolve()
    for p in sorted(prev_owned - expected_san_paths, reverse=True):
        if san_resolved in p.parents and p.is_file():
            p.unlink(missing_ok=True)
    for d in sorted(san_dir.rglob("*"), reverse=True):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()


_REFERENCE_SCHEMA_TOKENS = frozenset({
    "drug", "medicine", "formulary", "protocol", "guideline",
    "parameter", "range", "reference", "normal", "unit", "test", "code",
})


def _is_reference_candidate(
    pf: ParsedFile,
    all_file_spans: list[Span],
    orig_rel: str,
    mapped_rel: str,
) -> bool:
    """Classify reference document by affirmative schema + zero detections (never folder name)."""
    if mapped_rel != orig_rel:
        return False
    if any(not sp.protected for sp in all_file_spans):
        return False
    for rec in pf.records:
        for f in rec.fields:
            if "patient" in f.path.lower():
                return False
    if pf.fmt in ("csv", "tsv", "json"):
        header_text = " ".join(
            f.text.lower()
            for r in pf.records
            if r.key[1] == "@header"
            for f in r.fields
        )
        path_text = " ".join(f.path.lower() for r in pf.records for f in r.fields)
        combined = header_text + " " + path_text
        return any(tok in combined for tok in _REFERENCE_SCHEMA_TOKENS)
    return False


def run(input_dir: Path, output_dir: Path, keys: Keys, policy: dict[str, Any]) -> int:
    t0 = time.monotonic()
    t_stage: dict[str, float] = {
        "inventory": 0.0,
        "parse": 0.0,
        "detect": 0.0,
        "linkage": 0.0,
        "compile_keys": 0.0,
        "propagate": 0.0,
        "project": 0.0,
        "gate": 0.0,
    }
    san_dir = output_dir / "sanitized"
    log_dir = output_dir / "logs"
    san_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    ctx = detect.build_context(policy, keys=keys)
    all_rel_paths: list[str] = []
    file_metas: list[_FileMeta] = []
    light_pfs: list[ParsedFile] = []
    quarantined_map: dict[str, dict[str, Any]] = {}
    evidences: list[Evidence] = []
    all_spans_flat: list[Span] = []
    all_filename_signals: list[FilenameSignal] = []

    # Pass 1: stream inventory -> parse -> detect -> extract compact evidence, then release buffers.
    _t_inv0 = time.monotonic()
    for entry in inventory.iter_scan(input_dir):
        t_stage["inventory"] += time.monotonic() - _t_inv0
        all_rel_paths.append(entry.rel_path)

        if entry.quarantine_reason is not None:
            quarantined_map[entry.rel_path] = {
                "input_sha256": entry.sha256,
                "output": None,
                "status": "quarantined",
                "reason": entry.quarantine_reason,
            }
            _t_inv0 = time.monotonic()
            continue

        _tp = time.monotonic()
        try:
            pf = adapters.parse(entry, policy)
            ctx.coltypes = col_typing.type_columns(pf)
        except StageError as exc:
            t_stage["parse"] += time.monotonic() - _tp
            quarantined_map[entry.rel_path] = {
                "input_sha256": entry.sha256,
                "output": None,
                "status": "quarantined",
                "reason": PARSE_REASON.get(exc.code, "validation_failed"),
            }
            _t_inv0 = time.monotonic()
            continue
        except (ValueError, UnicodeError, OSError):
            t_stage["parse"] += time.monotonic() - _tp
            quarantined_map[entry.rel_path] = {
                "input_sha256": entry.sha256,
                "output": None,
                "status": "quarantined",
                "reason": "validation_failed",
            }
            _t_inv0 = time.monotonic()
            continue
        t_stage["parse"] += time.monotonic() - _tp

        _td = time.monotonic()
        sig = rename.extract_filename_signals(pf.rel_path)
        extra = dict(pf.extra)
        extra["filename_signals"] = sig
        # Build FilenameSignal objects from digit/token signals for C's linkage module.
        for digit_val in sig.get("digits", []):
            all_filename_signals.append(
                FilenameSignal(input_file=pf.rel_path, typ="ID", value=digit_val, source="filename")
            )
        for tok_val in sig.get("tokens", []):
            # Names in exported filenames are Title-Case (Firstname_Lastname) or Devanagari;
            # lower-case tokens (hospital codes, "ds", "whatsapp") are boilerplate and
            # would become Layer A leak keys matching every occurrence in the corpus.
            if not (tok_val[:1].isupper() or any("ऀ" <= ch <= "ॿ" for ch in tok_val)):
                continue
            all_filename_signals.append(
                FilenameSignal(input_file=pf.rel_path, typ="NAME", value=tok_val, source="filename")
            )
        # Owner D: per-file date order + PII-free QI summaries for privacy.plan (D1-D4)
        if pf.fmt != "fhir":
            extra["date_order"] = priv_spans.file_date_order(pf.records, policy)
            extra["record_qi"] = {
                r.key[1]: priv_spans.record_summary(r) for r in pf.records if priv_spans.summarizable(r.key[1])
            }
        for rec in pf.records:
            sh_spans = shield.protect(rec, ctx)
            det_spans = detect.detect(rec, ctx, sh_spans)
            all_spans = sh_spans + det_spans
            if pf.fmt != "fhir":
                all_spans += priv_spans.privacy_spans(rec, policy, extra["date_order"])
            all_spans_flat.extend(all_spans)
            if _is_public_loc(rec.key[1]):
                evidences.append(evidence.extract(rec, all_spans))
        t_stage["detect"] += time.monotonic() - _td

        file_metas.append(
            _FileMeta(
                abs_path=entry.abs_path,
                rel_path=pf.rel_path,
                sha256=pf.sha256,
                fmt=pf.fmt,
                encoding=pf.encoding,
                extra=extra,
            )
        )
        light_pfs.append(
            ParsedFile(
                rel_path=pf.rel_path,
                fmt=pf.fmt,
                sha256=pf.sha256,
                encoding=pf.encoding,
                newline=pf.newline,
                records=[],
                raw_bytes=b"",
                extra=extra,
            )
        )
        _t_inv0 = time.monotonic()

    _ts = time.monotonic()
    file_digests = {m.rel_path: m.sha256 for m in file_metas}
    resolution = linkage.resolve(
        evidences,
        keys,
        policy,
        file_digests=file_digests,
        filename_signals=all_filename_signals if all_filename_signals else None,
    )
    plan = privacy.plan(resolution, evidences, light_pfs, policy)
    t_stage["linkage"] = time.monotonic() - _ts

    out_rel_map = rename.allocate_output_paths([m.rel_path for m in file_metas], keys, resolution)

    # Preflight output ownership BEFORE any gate write to sanitized/.
    planned_san_paths = {(san_dir / out_rel_map[m.rel_path]).resolve() for m in file_metas}
    prev_owned = _preflight_output_ownership(output_dir, san_dir, planned_san_paths)

    _ts = time.monotonic()
    leak_keys = residual_a.compile_keys(
        all_spans_flat,
        evidences,
        filename_signals=all_filename_signals if all_filename_signals else None,
    )
    leak_keys = _drop_path_boilerplate_keys(leak_keys, all_rel_paths)
    t_stage["compile_keys"] = time.monotonic() - _ts
    all_spans_flat.clear()

    max_passes = int(policy.get("runtime", {}).get("residual_passes", 3))
    ref_byte_copy = bool(policy.get("runtime", {}).get("reference_doc_byte_copy", True))

    record_rows: list[dict[str, Any]] = []
    sanitized_map: dict[str, dict[str, Any]] = {}
    accepted_entity_counts: Counter[str] = Counter()
    gate_totals: Counter[str] = Counter({"A": 0, "B1": 0, "B2": 0, "B3": 0, "B4": 0})
    records_quarantined = 0
    generalized_set = set(plan.generalized)

    # Pass 2: stream one file at a time, verify digest, project, gate, and release buffers.
    for meta in file_metas:
        entry = inventory.read_entry(meta.abs_path, meta.rel_path)
        if entry.sha256 != meta.sha256 or entry.quarantine_reason is not None:
            raise StageError("inventory", "E_IO")

        pf = adapters.parse(entry, policy)
        ctx.coltypes = col_typing.type_columns(pf)
        pf.extra["filename_signals"] = meta.extra.get("filename_signals", {})

        mapped_rel = out_rel_map[pf.rel_path]
        rel_out = f"sanitized/{mapped_rel}"
        dest = san_dir / mapped_rel
        public_recs = [r for r in pf.records if _is_public_loc(r.key[1])]

        spans_by_rec: dict[RecKey, list[Span]] = {}
        all_file_spans: list[Span] = []
        _tp = time.monotonic()
        date_order = meta.extra.get("date_order", "dmy")
        for rec in pf.records:
            sh_spans = shield.protect(rec, ctx)
            det_spans = detect.detect(rec, ctx, sh_spans)
            if pf.fmt != "fhir":
                det_spans = det_spans + priv_spans.privacy_spans(rec, policy, date_order)
            swept = propagate.sweep(rec, resolution, sh_spans + det_spans)
            spans_by_rec[rec.key] = swept
            all_file_spans.extend(swept)
        t_stage["propagate"] += time.monotonic() - _tp

        # Reference-document byte-copy path (A11)
        if ref_byte_copy and _is_reference_candidate(pf, all_file_spans, pf.rel_path, mapped_rel):
            _tg = time.monotonic()
            ref_ok, _, _, ref_counts = gate.run_with_repair(
                dest,
                pf.raw_bytes,
                rel_out,
                LocMap([]),
                leak_keys,
                residual_a.scan_gate,
                residual_b.scan,
                reproject_fn=None,
                records_by_key={},
                encoding=pf.encoding,
                max_passes=1,
            )
            t_stage["gate"] += time.monotonic() - _tg
            gate_totals.update(ref_counts)
            if ref_ok:
                sanitized_map[pf.rel_path] = {
                    "input_sha256": pf.sha256,
                    "output": rel_out,
                    "status": "sanitized",
                }
                continue

        # Linked FHIR requires registered FHIR handler (fail closed otherwise)
        if pf.fmt == "fhir" and not getattr(fhir, "REGISTER_FHIR", False):
            if any(resolution.patient_of.get(r.key) is not None for r in public_recs):
                sanitized_map[pf.rel_path] = {
                    "input_sha256": pf.sha256,
                    "output": None,
                    "status": "quarantined",
                    "reason": "unsupported_format",
                }
                records_quarantined += len(public_recs)
                continue

        try:
            base_edits = []
            protected_file_spans: list[Span] = []
            for rec in pf.records:
                rec_spans = spans_by_rec.get(rec.key, [])
                protected_file_spans.extend(s for s in rec_spans if s.protected)
                base_edits.extend(
                    edits.build(
                        rec,
                        rec_spans,
                        resolution,
                        plan,
                        keys=keys,
                        date_shifter=dates.make_shifter(date_order),
                    )
                )
            base_edits.extend(rename.rename_headers_and_keys(pf, all_file_spans))
            assert pf.projector is not None
            records_by_key = {r.key: r for r in pf.records}
            if pf.fmt == "fhir" and "fhir_ctx" in pf.extra:
                pf.extra["fhir_ctx"].update({
                    "resolution": resolution,
                    "plan": plan,
                    "keys": keys,
                    "policy": policy,
                })

            def reproject(extra_repairs: list[Any]) -> tuple[bytes, LocMap]:
                if pf.fmt == "fhir":
                    edits_arg = extra_repairs
                elif extra_repairs:
                    rep_by_field: dict[tuple[Any, int], list[Any]] = {}
                    for er in extra_repairs:
                        rep_by_field.setdefault((er.key, er.field_idx), []).append(er)
                    filtered_base = [
                        be for be in base_edits
                        if not any(
                            be.start < er.end and er.start < be.end
                            for er in rep_by_field.get((be.key, be.field_idx), ())
                        )
                    ]
                    edits_arg = [*filtered_base, *extra_repairs]
                else:
                    edits_arg = base_edits
                return pf.projector(  # type: ignore[misc]
                    edits_arg,
                    resolution.patient_of,
                    protected_spans=protected_file_spans,
                )

            _tproj = time.monotonic()
            out_bytes, loc_map = reproject([])
            t_stage["project"] += time.monotonic() - _tproj
        except (StageError, ValueError, UnicodeError, OSError):
            sanitized_map[pf.rel_path] = {
                "input_sha256": pf.sha256,
                "output": None,
                "status": "quarantined",
                "reason": "validation_failed",
            }
            records_quarantined += len(public_recs)
            continue

        _tg2 = time.monotonic()
        ok, final_bytes, _final_loc, g_counts = gate.run_with_repair(
            dest,
            out_bytes,
            rel_out,
            loc_map,
            leak_keys,
            residual_a.scan_gate,
            residual_b.scan,
            reproject_fn=reproject,
            records_by_key=records_by_key,
            encoding="utf-8",
            max_passes=max_passes,
        )
        t_stage["gate"] += time.monotonic() - _tg2
        gate_totals.update(g_counts)
        if not ok or final_bytes is None:
            sanitized_map[pf.rel_path] = {
                "input_sha256": pf.sha256,
                "output": None,
                "status": "quarantined",
                "reason": "residual_pii",
            }
            records_quarantined += len(public_recs)
            continue

        # Commit accepted entities and public record rows ONLY after passing gate
        for sp in all_file_spans:
            if not sp.protected:
                accepted_entity_counts[sp.typ] += 1

        sanitized_map[pf.rel_path] = {
            "input_sha256": pf.sha256,
            "output": rel_out,
            "status": "sanitized",
        }
        for rec in public_recs:
            row: dict[str, Any] = {
                "file": rel_out,
                "loc": rec.key[1],
                "patient": resolution.patient_of.get(rec.key),
            }
            if rec.key in generalized_set:
                row["status"] = "generalized"
            record_rows.append(row)

    file_reports: list[dict[str, Any]] = []
    for rel_path in all_rel_paths:
        if rel_path in quarantined_map:
            file_reports.append(quarantined_map[rel_path])
        elif rel_path in sanitized_map:
            file_reports.append(sanitized_map[rel_path])

    # Remove stale files owned by prior manifest; preserve unrelated unowned files
    expected_san_paths = {
        (output_dir / f["output"]).resolve()
        for f in file_reports
        if f.get("status") == "sanitized" and f.get("output")
    }
    _cleanup_stale_outputs(san_dir, prev_owned, expected_san_paths)

    elapsed = time.monotonic() - t0
    identifiers_total = sum(accepted_entity_counts.values())
    t_stage["total"] = elapsed
    emit.write(
        output_dir,
        file_reports=file_reports,
        record_rows=record_rows,
        entity_counts=dict(accepted_entity_counts),
        identifiers_detected=identifiers_total,
        records_generalized=sum(1 for r in record_rows if r.get("status") == "generalized"),
        records_suppressed=0,
        elapsed_seconds=elapsed,
        records_quarantined=records_quarantined,
        stage_seconds={k: round(v, 4) for k, v in t_stage.items()},
    )

    n_san = sum(1 for f in file_reports if f["status"] == "sanitized")
    log_counts(log_dir, "gate", **dict(gate_totals))
    log_counts(
        log_dir,
        "emit",
        files=len(file_reports),
        sanitized=n_san,
        quarantined=len(file_reports) - n_san,
        records=len(record_rows),
        identifiers=identifiers_total,
    )
    print(f"files {len(file_reports)} sanitized {n_san} quarantined {len(file_reports) - n_san} records {len(record_rows)}")
    return 0
