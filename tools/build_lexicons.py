"""Offline reproducible seed-lexicon, MANIFEST, and term-frequency builder."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import gzip
import io
import json
from pathlib import Path
import sys
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kavachmed.indic import lexicon_tokens, normalize_name  # noqa: E402


@dataclass(frozen=True, slots=True)
class SourceProvenance:
    """Provenance metadata for a single lexicon input source."""

    name: str
    url: str
    licence: str
    retrieved: str

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "url": self.url,
            "licence": self.licence,
            "retrieved": self.retrieved,
        }


@dataclass(frozen=True, slots=True)
class LocalNameSource:
    """A local file supplying name strings plus its provenance."""

    path: Path
    provenance: SourceProvenance
    columns: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class TermFreqResult:
    """Aggregate term-frequency counts and total rows scanned."""

    total_rows: int
    counts: dict[str, int]


def _is_ascii_alpha_token(tok: str) -> bool:
    return bool(tok) and all("a" <= ch <= "z" for ch in tok)


def _is_devanagari_token(tok: str) -> bool:
    return bool(tok) and all(0x0900 <= ord(ch) <= 0x097F for ch in tok)


def _load_json_no_duplicates(path: Path) -> dict[str, object]:
    """Load JSON object while rejecting duplicate keys at any level."""

    def _check_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        obj: dict[str, object] = {}
        for key, val in pairs:
            if key in obj:
                raise ValueError(f"Duplicate JSON key in {path}: {key!r}")
            obj[key] = val
        return obj

    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw, object_pairs_hook=_check_pairs)
    if not isinstance(data, dict):
        raise ValueError(f"Manifest {path} must be a JSON object")
    return data


def _count_artifact_rows(path: Path) -> int:
    """Count data rows in a lexicon artifact (excluding CSV/TSV headers)."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            lines = [line for line in fh if line.strip()]
        inner_suffix = Path(path.stem).suffix
        if inner_suffix in {".csv", ".tsv"}:
            return max(0, len(lines) - 1)
        return len(lines)

    with path.open("r", encoding="utf-8") as fh:
        lines = [line for line in fh if line.strip()]
    if path.suffix in {".csv", ".tsv"}:
        return max(0, len(lines) - 1)
    return len(lines)


def validate_manifest(lexicon_dir: Path, manifest_path: Path | None = None) -> dict[str, object]:
    """Validate MANIFEST.json schema, provenance fields, file coverage, and row counts."""
    lexicon_dir = Path(lexicon_dir)
    manifest_path = Path(manifest_path) if manifest_path is not None else lexicon_dir / "MANIFEST.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    manifest = _load_json_no_duplicates(manifest_path)

    # Verify every artifact file under lexicon_dir has a manifest entry.
    emitted_files = sorted(
        p.relative_to(lexicon_dir).as_posix()
        for p in lexicon_dir.rglob("*")
        if p.is_file() and p.resolve() != manifest_path.resolve()
    )
    missing_entries = [rel for rel in emitted_files if rel not in manifest]
    if missing_entries:
        raise ValueError(f"Emitted lexicon file(s) missing from MANIFEST.json: {missing_entries}")

    for rel_path, entry in manifest.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Manifest entry {rel_path!r} must be an object")

        sources = entry.get("sources")
        if not isinstance(sources, list) or not sources:
            raise ValueError(f"Manifest entry {rel_path!r} must have a non-empty 'sources' list")
        for idx, src in enumerate(sources):
            if not isinstance(src, dict):
                raise ValueError(f"Manifest entry {rel_path!r} source[{idx}] must be an object")
            for req in ("name", "url", "licence", "retrieved"):
                val = src.get(req)
                if not isinstance(val, str) or not val.strip():
                    raise ValueError(
                        f"Manifest entry {rel_path!r} source[{idx}] missing non-empty {req!r}"
                    )

        build_script = entry.get("build_script")
        if not isinstance(build_script, str) or not build_script.strip():
            raise ValueError(f"Manifest entry {rel_path!r} missing non-empty 'build_script'")

        declared_count = entry.get("count")
        if not isinstance(declared_count, int) or isinstance(declared_count, bool) or declared_count < 0:
            raise ValueError(f"Manifest entry {rel_path!r} must have integer 'count' >= 0")

        artifact_path = lexicon_dir / rel_path
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Manifest entry {rel_path!r} points to missing file: {artifact_path}")

        actual_count = _count_artifact_rows(artifact_path)
        if actual_count != declared_count:
            raise ValueError(
                f"Count mismatch for {rel_path!r}: manifest={declared_count}, actual={actual_count}"
            )

    return manifest


def _write_text_artifact(path: Path, content: str, compress: bool) -> None:
    """Write UTF-8 text or deterministic gzip (mtime=0, no filename) to path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = content.encode("utf-8")
    if not compress:
        path.write_bytes(raw)
        return

    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0) as gz:
        gz.write(raw)
    path.write_bytes(buf.getvalue())


def _read_local_name_source(src: LocalNameSource) -> list[str]:
    """Read raw name strings from a local .txt, .csv, or .jsonl file."""
    for field_name in ("name", "url", "licence", "retrieved"):
        val = getattr(src.provenance, field_name)
        if not isinstance(val, str) or not val.strip():
            raise ValueError(f"Missing required source metadata {field_name!r} for {src.path}")

    if not src.path.is_file():
        raise FileNotFoundError(f"Declared local name source not found: {src.path}")

    suffix = src.path.suffix.lower()
    values: list[str] = []
    if suffix == ".csv":
        with src.path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                return values
            cols = src.columns or tuple(reader.fieldnames)
            missing_cols = [c for c in cols if c not in reader.fieldnames]
            if missing_cols:
                raise ValueError(f"Missing column(s) {missing_cols} in {src.path}")
            for row in reader:
                for col in cols:
                    val = row.get(col)
                    if val:
                        values.append(val)
    elif suffix == ".jsonl":
        with src.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                cols = src.columns or tuple(sorted(obj.keys()))
                for col in cols:
                    val = obj.get(col)
                    if isinstance(val, str) and val:
                        values.append(val)
    else:
        with src.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    values.append(line)
    return values


def _generate_faker_names(seed: int, budget: int) -> tuple[list[str], SourceProvenance]:
    """Generate en_IN and hi_IN names from pinned Faker if requested."""
    try:
        import faker  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "Faker is not installed; install faker or run with --faker-budget 0 and local sources"
        ) from exc

    fake = faker.Faker(["en_IN", "hi_IN"])
    fake.seed_instance(seed)
    names: list[str] = []
    for _ in range(budget):
        names.append(fake.first_name())
        names.append(fake.last_name())
    prov = SourceProvenance(
        name=f"Faker en_IN/hi_IN (faker=={faker.__version__}, seed={seed}, budget={budget})",
        url="https://pypi.org/project/Faker/",
        licence="MIT",
        retrieved="generated-offline",
    )
    return names, prov


def _update_manifest(lexicon_dir: Path, updates: dict[str, dict[str, object]]) -> Path:
    """Merge updates into MANIFEST.json while preserving other owners' entries."""
    manifest_path = lexicon_dir / "MANIFEST.json"
    existing: dict[str, object] = {}
    if manifest_path.exists():
        existing = _load_json_no_duplicates(manifest_path)

    merged: dict[str, object] = dict(existing)
    merged.update(updates)
    ordered = {key: merged[key] for key in sorted(merged.keys())}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(ordered, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest_path


def build_seed_lexicon(
    lexicon_dir: Path,
    local_sources: Sequence[LocalNameSource] = (),
    *,
    faker_seed: int = 42,
    faker_budget: int = 0,
    compress: bool = False,
    build_script: str = "tools/build_lexicons.py",
) -> dict[str, Path]:
    """Build sorted, deduplicated Latin and Devanagari seed name lexicons and update MANIFEST.json."""
    lexicon_dir = Path(lexicon_dir)
    raw_names: list[str] = []
    provenances: list[SourceProvenance] = []

    if faker_budget > 0:
        f_names, f_prov = _generate_faker_names(faker_seed, faker_budget)
        raw_names.extend(f_names)
        provenances.append(f_prov)

    for src in sorted(local_sources, key=lambda s: str(s.path)):
        raw_names.extend(_read_local_name_source(src))
        provenances.append(src.provenance)

    if not provenances:
        raise ValueError("At least one source (Faker budget > 0 or local_sources) is required")

    latin_tokens: set[str] = set()
    deva_tokens: set[str] = set()
    for raw in raw_names:
        for tok in lexicon_tokens(raw):
            if _is_ascii_alpha_token(tok):
                latin_tokens.add(tok)
            elif _is_devanagari_token(tok):
                deva_tokens.add(tok)

    ext = ".txt.gz" if compress else ".txt"
    lat_rel = f"names/latin{ext}"
    dev_rel = f"names/devanagari{ext}"

    sorted_lat = sorted(latin_tokens)
    sorted_dev = sorted(deva_tokens)

    lat_path = lexicon_dir / lat_rel
    dev_path = lexicon_dir / dev_rel

    _write_text_artifact(lat_path, "\n".join(sorted_lat) + ("\n" if sorted_lat else ""), compress)
    _write_text_artifact(dev_path, "\n".join(sorted_dev) + ("\n" if sorted_dev else ""), compress)

    prov_dicts = [p.to_dict() for p in provenances]
    _update_manifest(
        lexicon_dir,
        {
            lat_rel: {
                "sources": prov_dicts,
                "build_script": build_script,
                "count": len(sorted_lat),
            },
            dev_rel: {
                "sources": prov_dicts,
                "build_script": build_script,
                "count": len(sorted_dev),
            },
        },
    )
    validate_manifest(lexicon_dir)
    return {"latin": lat_path, "devanagari": dev_path}


def _iter_shard_name_pairs(
    shard_path: Path, first_col: str, last_col: str
) -> Iterable[tuple[str | None, str | None]]:
    """Yield (first, last) pairs from a CSV, JSONL, or Parquet shard with strict column checks."""
    if not shard_path.is_file():
        raise FileNotFoundError(f"Declared shard not found: {shard_path}")

    suffix = shard_path.suffix.lower()
    if suffix == ".csv":
        with shard_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            fields = set(reader.fieldnames or ())
            if first_col not in fields or last_col not in fields:
                raise ValueError(
                    f"Shard {shard_path} missing required columns ({first_col!r}, {last_col!r}); "
                    f"found {sorted(fields)}"
                )
            for row in reader:
                yield row.get(first_col), row.get(last_col)
        return

    if suffix == ".jsonl":
        with shard_path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if first_col not in obj or last_col not in obj:
                    raise ValueError(
                        f"Shard {shard_path} line {line_no} missing required columns "
                        f"({first_col!r}, {last_col!r})"
                    )
                yield obj.get(first_col), obj.get(last_col)
        return

    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "pyarrow is required to read .parquet shards offline"
            ) from exc

        pf = pq.ParquetFile(shard_path)
        schema_names = set(pf.schema_arrow.names)
        if first_col not in schema_names or last_col not in schema_names:
            raise ValueError(
                f"Shard {shard_path} missing required columns ({first_col!r}, {last_col!r}); "
                f"found {sorted(schema_names)}"
            )
        for batch in pf.iter_batches(columns=[first_col, last_col]):
            col_first = batch.column(0).to_pylist()
            col_last = batch.column(1).to_pylist()
            for f_val, l_val in zip(col_first, col_last):
                yield f_val, l_val
        return

    raise ValueError(f"Unsupported shard format: {shard_path}")


def build_term_frequencies(
    shard_paths: Sequence[Path],
    lexicon_dir: Path,
    provenance: SourceProvenance,
    *,
    first_col: str = "firstName",
    last_col: str = "lastName",
    build_script: str = "tools/build_lexicons.py",
) -> TermFreqResult:
    """Compute per-row distinct identifying token frequencies across shards and register in MANIFEST.json."""
    if not shard_paths:
        raise ValueError("shard_paths must not be empty")

    for field_name in ("name", "url", "licence", "retrieved"):
        val = getattr(provenance, field_name)
        if not isinstance(val, str) or not val.strip():
            raise ValueError(f"Missing required provenance field {field_name!r}")

    ordered_shards = sorted((Path(p) for p in shard_paths), key=lambda p: p.as_posix())
    for sp in ordered_shards:
        if not sp.is_file():
            raise FileNotFoundError(f"Declared shard not found: {sp}")

    total_rows = 0
    counts: dict[str, int] = {}

    for sp in ordered_shards:
        for first_val, last_val in _iter_shard_name_pairs(sp, first_col, last_col):
            total_rows += 1
            row_tokens: set[str] = set()
            for raw in (first_val, last_val):
                if not isinstance(raw, str) or not raw.strip():
                    continue
                for tok in normalize_name(raw):
                    if tok.kind == "identifying" and tok.latin:
                        row_tokens.add(tok.latin)
            for ident in sorted(row_tokens):
                counts[ident] = counts.get(ident, 0) + 1

    lexicon_dir = Path(lexicon_dir)
    tf_rel = "names/term_freq.csv"
    tf_path = lexicon_dir / tf_rel
    tf_path.parent.mkdir(parents=True, exist_ok=True)

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["token", "rows_with_token"])
    for token in sorted(counts.keys()):
        writer.writerow([token, counts[token]])
    tf_path.write_bytes(buf.getvalue().encode("utf-8"))

    _update_manifest(
        lexicon_dir,
        {
            tf_rel: {
                "sources": [provenance.to_dict()],
                "build_script": build_script,
                "count": len(counts),
                "total_rows": total_rows,
            }
        },
    )
    validate_manifest(lexicon_dir)
    return TermFreqResult(total_rows=total_rows, counts=counts)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build and validate KavachMed name lexicons.")
    parser.add_argument("--lexicon-dir", type=Path, default=Path("config/lexicons"))
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--faker-seed", type=int, default=42)
    parser.add_argument("--faker-budget", type=int, default=0)
    parser.add_argument("--compress", action="store_true")
    parser.add_argument("--local-list", type=Path, action="append", default=[])
    parser.add_argument("--source-name", default="")
    parser.add_argument("--source-url", default="")
    parser.add_argument("--source-licence", default="")
    parser.add_argument("--retrieved", default="")
    args = parser.parse_args(argv)

    if args.validate_only:
        validate_manifest(args.lexicon_dir)
        return 0

    local_sources = [
        LocalNameSource(
            path=p,
            provenance=SourceProvenance(
                name=args.source_name,
                url=args.source_url,
                licence=args.source_licence,
                retrieved=args.retrieved,
            ),
        )
        for p in args.local_list
    ]
    build_seed_lexicon(
        args.lexicon_dir,
        local_sources=local_sources,
        faker_seed=args.faker_seed,
        faker_budget=args.faker_budget,
        compress=args.compress,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
