"""Input inventory: byte-sorted enumeration, raw SHA-256, sniffing, and decoding."""
from __future__ import annotations

from collections.abc import Iterator
import hashlib
from pathlib import Path

from kavachmed.contracts import FileEntry
from kavachmed.sniff import decode_bytes, is_binary_magic, sniff_content


def read_entry(path: Path, rel: str) -> FileEntry:
    """Read, hash, sniff, and decode a single input file."""
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()

    if not raw.strip():
        return FileEntry(path, rel, sha, raw, "unknown", None, None, "empty_file")

    if is_binary_magic(raw):
        return FileEntry(path, rel, sha, raw, "binary", None, None, "unsupported_format")

    text, enc, err_reason = decode_bytes(raw)
    if err_reason is not None or text is None:
        return FileEntry(path, rel, sha, raw, "unknown", None, None, err_reason or "undecodable")

    fmt = sniff_content(path, text)
    return FileEntry(path, rel, sha, raw, fmt, enc, text, None)


def iter_scan(input_dir: Path) -> Iterator[FileEntry]:
    """Stream all regular files in UTF-8 sorted path order."""
    candidates = [
        (p, p.relative_to(input_dir).as_posix())
        for p in input_dir.rglob("*")
        if p.is_file()
    ]
    candidates.sort(key=lambda item: item[1].encode("utf-8"))

    root = input_dir.resolve()
    for path, rel in candidates:
        if path.is_symlink():
            target = path.resolve()
            if target != root and root not in target.parents:
                # A link escaping the input folder is never parsed or released (it could pull in
                # any file on the host); hash it for the report and quarantine.
                raw = path.read_bytes()
                yield FileEntry(path, rel, hashlib.sha256(raw).hexdigest(), b"", "unknown", None, None, "unsupported_format")
                continue
        yield read_entry(path, rel)


def scan(input_dir: Path) -> list[FileEntry]:
    """Enumerate all regular files in UTF-8 sorted path order."""
    return list(iter_scan(input_dir))
