"""Keyed Crockford Base32 pseudonym allocation and date-shift derivation (Plan C V4)."""

from __future__ import annotations

from typing import Protocol

CROCKFORD_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

# Ordered offsets: [-365..-30] (336) followed by [30..365] (336) = 672 values.
_SHIFT_OFFSETS: tuple[int, ...] = tuple(range(-365, -29)) + tuple(range(30, 366))
_NUM_SHIFT_OFFSETS = len(_SHIFT_OFFSETS)  # 672


class KeysProtocol(Protocol):
    """Protocol for A's ``security.Keys``."""

    def keyed(self, purpose: str, *fields: str) -> bytes: ...


def encode_crockford_40bit(digest: bytes) -> str:
    """Encode the first 5 bytes (40 bits) of ``digest`` into 8 Crockford Base32 chars."""
    if len(digest) < 5:
        raise ValueError("digest_too_short")
    val = int.from_bytes(digest[:5], "big")
    chars = ["0"] * 8
    for idx in range(7, -1, -1):
        chars[idx] = CROCKFORD_ALPHABET[val & 0x1F]
        val >>= 5
    return "".join(chars)


def assign_pseudonyms(
    canonical_keys: list[tuple[int, str, str]],
    keys: KeysProtocol,
) -> dict[tuple[int, str, str], str]:
    """Assign deterministic, collision-free ``PSN-XXXXXXXX`` labels in sorted-key order."""
    from kavachmed.linkage import LinkageError

    sorted_keys = sorted(canonical_keys)
    used_labels: set[str] = set()
    result: dict[tuple[int, str, str], str] = {}

    for ckey in sorted_keys:
        kind, ns, core = ckey
        if kind == 0:
            fields = ("id", ns, core)
        elif kind == 1:
            fields = ("adm", core)
        else:
            fields = ("scored", core)
        digest = keys.keyed("psn", *fields)
        base_label = f"PSN-{encode_crockford_40bit(digest)}"
        if base_label not in used_labels:
            used_labels.add(base_label)
            result[ckey] = base_label
            continue

        # Collision: probe 32 1-char extensions starting at next 5 digest bits.
        if len(digest) < 6:
            raise LinkageError("pseudonym_bucket_exhausted")
        start_idx = (digest[5] >> 3) & 0x1F
        assigned: str | None = None
        for offset in range(32):
            ext_char = CROCKFORD_ALPHABET[(start_idx + offset) % 32]
            candidate = f"{base_label}{ext_char}"
            if candidate not in used_labels:
                assigned = candidate
                break
        if assigned is None:
            raise LinkageError("pseudonym_bucket_exhausted")
        used_labels.add(assigned)
        result[ckey] = assigned

    return result


def derive_shift_days(keys: KeysProtocol, *fields: str) -> int:
    """Derive a deterministic nonzero date-shift offset in ±[30, 365]."""
    digest = keys.keyed("shift", *fields)
    if not digest:
        raise ValueError("empty_shift_digest")
    prefix = digest[:8] if len(digest) >= 8 else digest
    idx = int.from_bytes(prefix, "big") % _NUM_SHIFT_OFFSETS
    return _SHIFT_OFFSETS[idx]
