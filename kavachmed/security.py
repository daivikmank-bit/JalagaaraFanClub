"""Secret gate, PBKDF2-HMAC-SHA256 + RFC 5869 HKDF key derivation, and process hardening.

Note on secret scrubbing:
    `os.environ.pop("RAKSHA_SECRET", None)` removes the variable from Python's
    `os.environ` mapping, but on Linux it does NOT overwrite `/proc/self/environ`
    (which reflects the initial process stack). We mitigate core-dump exposure
    via `RLIMIT_CORE=0` and `prctl(PR_SET_DUMPABLE, 0)` on Linux, and we never
    log, print, or serialize the secret or derived subkeys.
"""
from __future__ import annotations

import hashlib
import hmac
import sys
import warnings
from pathlib import Path
from typing import Mapping

PURPOSES: tuple[str, ...] = ("psn", "shift", "fname", "surr")
CROCKFORD_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"  # 32 chars, no I/L/O/U

STAGE_IDS = frozenset({
    "security", "inventory", "parse", "detect",
    "linkage", "privacy", "edits", "gate", "emit",
})

ERROR_CODES = frozenset({
    "E_SECRET", "E_UNSUPPORTED", "E_EMPTY", "E_UNDECODABLE",
    "E_MIXED_ENC", "E_INVALID_JSON", "E_DEPTH", "E_CSV",
    "E_RESIDUAL_PII", "E_IO",
})


class StageError(Exception):
    """Closed-vocabulary exception that never carries PII or secret material."""

    def __init__(self, stage: str, code: str) -> None:
        self.stage = stage if stage in STAGE_IDS else "security"
        self.code = code if code in ERROR_CODES else "E_IO"
        super().__init__(f"{self.stage}:{self.code}")

    def __str__(self) -> str:
        return f"{self.stage}:{self.code}"


def _lp(data: bytes) -> bytes:
    """4-byte big-endian length prefix so ('a|b','c') != ('a','b|c')."""
    return len(data).to_bytes(4, "big") + data


def hkdf_extract(salt: bytes, ikm: bytes) -> bytes:
    """RFC 5869 Section 2.2: HKDF-Extract with SHA-256."""
    actual_salt = salt if salt else b"\x00" * 32
    return hmac.new(actual_salt, ikm, hashlib.sha256).digest()


def hkdf_expand(prk: bytes, info: bytes, length: int = 32) -> bytes:
    """RFC 5869 Section 2.3: HKDF-Expand with SHA-256."""
    if length < 0 or length > 255 * 32:
        raise ValueError("invalid HKDF output length")
    okm = bytearray()
    t = b""
    counter = 1
    while len(okm) < length:
        t = hmac.new(prk, t + info + bytes([counter]), hashlib.sha256).digest()
        okm.extend(t)
        counter += 1
    return bytes(okm[:length])


class Keys:
    """Holds derived 32-byte subkeys per purpose with a redacted repr."""

    __slots__ = ("_subkeys",)

    def __init__(self, subkeys: dict[str, bytes]) -> None:
        self._subkeys = dict(subkeys)

    def __repr__(self) -> str:
        return "Keys(<redacted>)"

    def __str__(self) -> str:
        return "Keys(<redacted>)"

    def keyed(self, purpose: str, *fields: str | bytes) -> bytes:
        """Compute HMAC-SHA256(k_purpose, lp(f1) || ... || lp(fn))."""
        if purpose not in self._subkeys:
            raise StageError("security", "E_SECRET")
        payload = bytearray()
        for f in fields:
            raw = f.encode("utf-8") if isinstance(f, str) else f
            payload.extend(_lp(raw))
        return hmac.new(self._subkeys[purpose], bytes(payload), hashlib.sha256).digest()

    def psn(self, canonical_key: str, chars: int = 8) -> str:
        """Derive deterministic 40-bit (8 Crockford chars) pseudonym: PSN-XXXXXXXX."""
        digest = self.keyed("psn", canonical_key)
        bits = chars * 5
        n_bytes = (bits + 7) // 8
        val = int.from_bytes(digest[:n_bytes], "big") & ((1 << bits) - 1)
        code = "".join(
            CROCKFORD_ALPHABET[(val >> (5 * (chars - 1 - i))) & 31]
            for i in range(chars)
        )
        return f"PSN-{code}"


def load_keys(
    env: Mapping[str, str],
    *,
    pbkdf2_iterations: int = 600_000,
    pbkdf2_salt: bytes = b"kavachmed/v5/stretch",
    hkdf_version: bytes = b"v5",
) -> Keys:
    """Fail-closed secret gate (Q12: unset or blank -> E_SECRET) + PBKDF2 -> HKDF."""
    raw = env.get("RAKSHA_SECRET", "")
    if not raw or not raw.strip():
        raise StageError("security", "E_SECRET")

    master = hashlib.pbkdf2_hmac(
        "sha256",
        raw.encode("utf-8"),
        pbkdf2_salt,
        pbkdf2_iterations,
        dklen=32,
    )
    prk = hkdf_extract(b"", master)
    subkeys: dict[str, bytes] = {}
    for purpose in PURPOSES:
        info = _lp(b"kavachmed") + _lp(hkdf_version) + _lp(purpose.encode("ascii"))
        subkeys[purpose] = hkdf_expand(prk, info, 32)
    return Keys(subkeys)


def harden_process() -> None:
    """Suppress warnings, disable core dumps, and set PR_SET_DUMPABLE=0 on Linux."""
    warnings.filterwarnings("ignore")
    try:
        import resource  # Unix only
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ImportError, OSError, ValueError):
        pass

    if sys.platform.startswith("linux"):
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            libc.prctl(4, 0, 0, 0, 0)  # PR_SET_DUMPABLE = 4
        except OSError:
            pass


def install_excepthook() -> None:
    """Install a PII-free exception hook that prints only exception class & stage."""
    def _hook(exc_type: type[BaseException], exc_val: BaseException, _tb: object) -> None:
        stage = getattr(exc_val, "stage", "unknown")
        code = getattr(exc_val, "code", "")
        suffix = f" code={code}" if code else ""
        sys.stderr.write(f"FATAL {exc_type.__name__} stage={stage}{suffix}\n")

    sys.excepthook = _hook


def log_counts(log_dir: Path | None, stage: str, **counts: int) -> None:
    """Write strictly integer counts (never strings or PII) to logs/pipeline.log."""
    safe_stage = stage if stage in STAGE_IDS else "pipeline"
    parts = " ".join(f"{k}={int(v)}" for k, v in sorted(counts.items()))
    line = f"{safe_stage} {parts}\n" if parts else f"{safe_stage}\n"
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        with (log_dir / "pipeline.log").open("a", encoding="utf-8") as fh:
            fh.write(line)

