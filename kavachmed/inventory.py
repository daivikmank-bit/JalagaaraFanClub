"""Input inventory: byte-sorted enumeration, raw SHA-256, sniffing, and decoding."""
from __future__ import annotations

from collections.abc import Iterator
import hashlib
import os
import stat
from contextlib import ExitStack

from kavachmed.security import StageError
from kavachmed.credentials import key_material
from pathlib import Path

from kavachmed.contracts import FileEntry
from kavachmed.sniff import decode_bytes, is_binary_magic, sniff_content


def _read_regular(root: Path, rel: str) -> bytes:
    parts = Path(rel).parts
    if not parts or Path(rel).is_absolute() or ".." in parts:
        raise StageError("inventory", "E_IO")
    with ExitStack() as handles:
        if os.name == "nt":
            import ctypes
            import msvcrt
            import _winapi
            from ctypes import wintypes
            class AttributeTag(ctypes.Structure):
                _fields_ = [("attributes", wintypes.DWORD), ("tag", wintypes.DWORD)]
            info = ctypes.windll.kernel32.GetFileInformationByHandleEx
            current = root.absolute()
            for part in ("", *parts):
                current /= part
                flags = 0x02000000 | 0x00200000  # BACKUP_SEMANTICS | OPEN_REPARSE_POINT
                handle = _winapi.CreateFile(str(current), 0x80000000, 1, 0, 3, flags, 0)
                fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
                handles.callback(os.close, fd)
                attrs = AttributeTag()
                if not info(wintypes.HANDLE(handle), 9, ctypes.byref(attrs), ctypes.sizeof(attrs)):
                    raise StageError("inventory", "E_IO")
                if attrs.attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                    raise StageError("inventory", "E_IO")
        else:
            directory = os.O_DIRECTORY | os.O_NOFOLLOW | os.O_RDONLY
            parent = os.open(root, directory)
            handles.callback(os.close, parent)
            for part in parts[:-1]:
                parent = os.open(part, directory, dir_fd=parent)
                handles.callback(os.close, parent)
            fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
            handles.callback(os.close, fd)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise StageError("inventory", "E_IO")
        with open(fd, "rb", closefd=False) as stream:
            return stream.read()


def read_entry(path: Path, rel: str, root: Path | None = None) -> FileEntry:
    try:
        raw = _read_regular(root or path.parent, rel if root else path.name)
    except OSError:
        raise StageError("inventory", "E_IO") from None
    sha = hashlib.sha256(raw).hexdigest()

    if not raw.strip():
        return FileEntry(path, rel, sha, raw, "unknown", None, None, "empty_file")

    if is_binary_magic(raw) or path.suffix.lower() in {".pdf", ".jpg", ".jpeg", ".png", ".zip", ".xlsx", ".docx", ".exe", ".sqlite", ".dcm"}:
        return FileEntry(path, rel, sha, raw, "binary", None, None, "unsupported_format")

    text, enc, err_reason = decode_bytes(raw)
    if err_reason is not None or text is None:
        return FileEntry(path, rel, sha, raw, "unknown", None, None, err_reason or "undecodable")

    if key_material(text):
        return FileEntry(path, rel, sha, raw, "unknown", None, None, "key_material")

    fmt = sniff_content(path, text)
    return FileEntry(path, rel, sha, raw, fmt, enc, text, None)


def iter_scan(input_dir: Path) -> Iterator[FileEntry]:
    """Stream all regular files in UTF-8 sorted path order."""
    candidates = []
    for path in input_dir.rglob("*"):
        st = path.lstat()
        if stat.S_ISLNK(st.st_mode) or getattr(st, "st_file_attributes", 0) & 0x400:
            raise StageError("inventory", "E_IO")
        if not stat.S_ISDIR(st.st_mode):
            candidates.append((path.relative_to(input_dir).as_posix(), path))
    for rel, path in sorted(candidates):
        yield read_entry(path, rel, input_dir)


def scan(input_dir: Path) -> list[FileEntry]:
    """Enumerate all regular files in UTF-8 sorted path order."""
    return list(iter_scan(input_dir))
