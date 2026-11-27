import hashlib
import hmac
import unittest

from kavachmed.linkage import LinkageError
from kavachmed.linkage.pseudo import assign_pseudonyms, encode_crockford_40bit


class _SpyKeys:
    def __init__(self, secret: bytes = b"test-secret", fixed: dict[tuple[str, ...], bytes] | None = None) -> None:
        self._secret = secret
        self._fixed = fixed or {}
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def keyed(self, purpose: str, *fields: str) -> bytes:
        self.calls.append((purpose, fields))
        key = (purpose, *fields)
        if key in self._fixed:
            return self._fixed[key]
        msg = b"".join(f"{len(f)}:{f}".encode("utf-8") for f in fields)
        return hmac.new(self._secret, purpose.encode("utf-8") + b":" + msg, hashlib.sha256).digest()


class TestPseudonyms(unittest.TestCase):
    def test_crockford_extremes_and_purpose_isolation(self) -> None:
        self.assertEqual(encode_crockford_40bit(b"\x00" * 5), "00000000")
        self.assertEqual(encode_crockford_40bit(b"\xff" * 5), "ZZZZZZZZ")

        keys = _SpyKeys(
            fixed={
                ("psn", "id", "aadhaar", "0000"): b"\x00" * 32,
                ("psn", "id", "aadhaar", "1111"): b"\xff" * 32,
            }
        )
        ck1 = (0, "aadhaar", "0000")
        ck2 = (0, "aadhaar", "1111")
        assigned = assign_pseudonyms([ck2, ck1], keys)
        self.assertEqual(assigned[ck1], "PSN-00000000")
        self.assertEqual(assigned[ck2], "PSN-ZZZZZZZZ")
        self.assertEqual(
            keys.calls,
            [
                ("psn", ("id", "aadhaar", "0000")),
                ("psn", ("id", "aadhaar", "1111")),
            ],
        )

    def test_collision_extension_exhaustion_and_secret_rotation(self) -> None:
        # Three clusters colliding on the same 40-bit prefix, two starting at suffix index 0.
        base = b"\x00" * 5
        fixed = {
            ("psn", "id", "pan", "A"): base + b"\x00" + b"\x00" * 26,
            ("psn", "id", "pan", "B"): base + b"\x00" + b"\x00" * 26,
            ("psn", "id", "pan", "C"): base + b"\x00" + b"\x00" * 26,
        }
        keys = _SpyKeys(fixed=fixed)
        cks = [(0, "pan", "C"), (0, "pan", "A"), (0, "pan", "B")]
        assigned = assign_pseudonyms(cks, keys)
        self.assertEqual(assigned[(0, "pan", "A")], "PSN-00000000")
        self.assertEqual(assigned[(0, "pan", "B")], "PSN-000000000")
        self.assertEqual(assigned[(0, "pan", "C")], "PSN-000000001")

        # Bucket larger than 33 (1 base + 32 extensions) raises safe exhaustion error.
        exhaust_fixed = {
            ("psn", "id", "pan", f"K{i:02d}"): base + b"\x00" * 27 for i in range(34)
        }
        exhaust_keys = _SpyKeys(fixed=exhaust_fixed)
        with self.assertRaises(LinkageError) as ctx:
            assign_pseudonyms([(0, "pan", f"K{i:02d}") for i in range(34)], exhaust_keys)
        self.assertEqual(ctx.exception.code, "pseudonym_bucket_exhausted")

        # Key rotation produces distinct labels with no raw IDs embedded.
        k1 = _SpyKeys(secret=b"secret-one")
        k2 = _SpyKeys(secret=b"secret-two")
        real_cks = [(0, "uhid:snj", "019672"), (1, "adm", "snj:ipd09")]
        a1 = assign_pseudonyms(real_cks, k1)
        a2 = assign_pseudonyms(real_cks, k2)
        self.assertNotEqual(a1[real_cks[0]], a2[real_cks[0]])
        for label in list(a1.values()) + list(a2.values()):
            self.assertNotIn("019672", label)
            self.assertNotIn("ipd09", label)


if __name__ == "__main__":
    unittest.main()
