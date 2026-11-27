"""Unit tests for kavachmed.security: RFC 5869 A.1-A.3 vectors, fail-closed gate, domain separation."""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from kavachmed.security import (
    CROCKFORD_ALPHABET,
    StageError,
    hkdf_expand,
    hkdf_extract,
    load_keys,
)


def test_rfc5869_a1() -> None:
    ikm = bytes.fromhex("0b" * 22)
    salt = bytes.fromhex("000102030405060708090a0b0c")
    info = bytes.fromhex("f0f1f2f3f4f5f6f7f8f9")
    prk = hkdf_extract(salt, ikm)
    assert prk.hex() == "077709362c2e32df0ddc3f0dc47bba6390b6c73bb50f9c3122ec844ad7c2b3e5"
    okm = hkdf_expand(prk, info, 42)
    assert okm.hex() == (
        "3cb25f25faacd57a90434f64d0362f2a"
        "2d2d0a90cf1a5a4c5db02d56ecc4c5bf"
        "34007208d5b887185865"
    )


def test_rfc5869_a2() -> None:
    ikm = bytes(range(0x00, 0x50))
    salt = bytes(range(0x60, 0xB0))
    info = bytes(range(0xB0, 0x100))
    prk = hkdf_extract(salt, ikm)
    assert prk.hex() == "06a6b88c5853361a06104c9ceb35b45cef760014904671014a193f40c15fc244"
    okm = hkdf_expand(prk, info, 82)
    assert okm.hex() == (
        "b11e398dc80327a1c8e7f78c596a4934"
        "4f012eda2d4efad8a050cc4c19afa97c"
        "59045a99cac7827271cb41c65e590e09"
        "da3275600c2f09b8367793a9aca3db71"
        "cc30c58179ec3e87c14c01d5c1f3434f"
        "1d87"
    )


def test_rfc5869_a3() -> None:
    ikm = bytes.fromhex("0b" * 22)
    prk = hkdf_extract(b"", ikm)
    assert prk.hex() == "19ef24a32c717b167f33a91d6f648bdf96596776afdb6377ac434c1c293ccb04"
    okm = hkdf_expand(prk, b"", 42)
    assert okm.hex() == (
        "8da4e775a563c18f715f802a063c5a31"
        "b8a11f5c5ee1879ec3454e5f3c738d2d"
        "9d201395faa4b61a96c8"
    )


def test_fail_closed_secret_gate(tmp_path: Path) -> None:
    with pytest.raises(StageError) as exc_info:
        load_keys({})
    assert exc_info.value.stage == "security"
    assert exc_info.value.code == "E_SECRET"

    with pytest.raises(StageError):
        load_keys({"RAKSHA_SECRET": "   \t\n "})

    # CLI check: unset secret exits non-zero and never creates output/
    out_dir = tmp_path / "should_not_exist"
    inp_dir = tmp_path / "empty_inp"
    inp_dir.mkdir()
    env = dict(os.environ)
    env.pop("RAKSHA_SECRET", None)
    proc = subprocess.run(
        [sys.executable, "main.py", "--input", str(inp_dir), "--output", str(out_dir)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert not out_dir.exists()


def test_keyed_domain_and_purpose_separation() -> None:
    keys = load_keys({"RAKSHA_SECRET": "practice"}, pbkdf2_iterations=1000)
    assert repr(keys) == "Keys(<redacted>)"
    assert str(keys) == "Keys(<redacted>)"
    assert "practice" not in repr(keys)

    # Length-prefixed fields prevent concatenation ambiguity
    assert keys.keyed("psn", "a|b", "c") != keys.keyed("psn", "a", "b|c")
    # Distinct purposes yield distinct keys
    assert keys.keyed("psn", "patient-1") != keys.keyed("shift", "patient-1")
    assert keys.keyed("fname", "doc.txt") != keys.keyed("surr", "doc.txt")


def test_psn_format_and_unlinkability() -> None:
    k1 = load_keys({"RAKSHA_SECRET": "secret-one"}, pbkdf2_iterations=1000)
    k2 = load_keys({"RAKSHA_SECRET": "secret-two"}, pbkdf2_iterations=1000)
    psn1 = k1.psn("cluster:42")
    psn1_repeat = k1.psn("cluster:42")
    psn2 = k2.psn("cluster:42")

    assert psn1 == psn1_repeat
    assert psn1 != psn2
    assert re.fullmatch(rf"PSN-[{CROCKFORD_ALPHABET}]{{8}}", psn1)

