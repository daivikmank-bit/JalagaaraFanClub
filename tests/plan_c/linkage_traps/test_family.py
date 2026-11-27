import hashlib
import hmac
import random
import unittest

from kavachmed.evidence import Evidence, Field, Record, Span, extract
from kavachmed.indic import normalize_name
from kavachmed.linkage import LinkageError, Resolution, resolve


class _SpyKeys:
    def __init__(self, secret: bytes = b"family-secret") -> None:
        self._secret = secret
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def keyed(self, purpose: str, *fields: str) -> bytes:
        self.calls.append((purpose, fields))
        msg = b"".join(f"{len(f)}:{f}".encode("utf-8") for f in fields)
        return hmac.new(self._secret, purpose.encode("utf-8") + b":" + msg, hashlib.sha256).digest()


def _ev(
    rk: tuple[str, str],
    name: str,
    *,
    sex: str | None = None,
    birth_years: tuple[int, int] | None = None,
    kin: str | None = None,
    ids: dict[str, list[str]] | None = None,
    phones: list[str] | None = None,
    phone_shared: bool = False,
    adm_key: str | None = None,
    hospital: str | None = None,
) -> Evidence:
    return Evidence(
        key=rk,
        names=normalize_name(name),
        kin=normalize_name(kin) if kin else [],
        sex=sex,
        birth_years=birth_years,
        ids=ids or {},
        phones=phones or [],
        phone_shared=phone_shared,
        adm_key=adm_key,
        hospital=hospital,
    )


class TestTemporalFamily(unittest.TestCase):
    def test_phase1_empty_single_graph_parity_identity_invariance_and_invalid_mode(self) -> None:
        keys = _SpyKeys()
        self.assertEqual(resolve([], keys, policy={"family_mode": "temporal_family"}), Resolution({}, {}, {}, {}))

        with self.assertRaises(LinkageError) as ctx:
            resolve([], keys, policy={"family_mode": "invalid_mode"})
        self.assertEqual(ctx.exception.code, "invalid_linkage_policy")

        evs = [
            _ev(("a.csv", "row:0"), "Ramesh Kumar", sex="M", birth_years=(1980, 1980), ids={"uhid": ["SNJ/26/000001"]}),
            _ev(("b.txt", "doc"), "रमेश कुमार", sex="M", birth_years=(1980, 1980), ids={"uhid": ["SNJ/26/000001"]}),
            _ev(("c.txt", "doc"), "Ramesh Kumar", sex="M", birth_years=(1980, 1980)),  # unlinked
            _ev(("d.txt", "doc"), "Sita Devi", sex="F", birth_years=(1975, 1975)),  # unlinked
        ]
        digests = {"a.csv": "da", "b.txt": "db", "c.txt": "dc", "d.txt": "dd"}

        res_single = resolve(
            evs,
            _SpyKeys(),
            policy={"mode": "scored", "family_mode": "single_graph"},
            file_digests=digests,
        )
        res_default = resolve(
            evs,
            _SpyKeys(),
            policy={"mode": "scored"},
            file_digests=digests,
        )
        self.assertEqual(res_single, res_default)

        res_fam = resolve(
            evs,
            _SpyKeys(),
            policy={"mode": "scored", "family_mode": "temporal_family"},
            file_digests=digests,
        )
        # V5 identity maps are strictly invariant across family modes.
        self.assertEqual(res_fam.patient_of, res_single.patient_of)
        self.assertEqual(res_fam.cluster_of, res_single.cluster_of)
        self.assertIsNone(res_fam.patient_of[("c.txt", "doc")])
        self.assertIsNone(res_fam.patient_of[("d.txt", "doc")])
        # Unlinked Ramesh (birth_years overlap) joins Ramesh's strict cluster family.
        self.assertEqual(res_fam.family_of[("c.txt", "doc")], res_fam.family_of[("a.csv", "row:0")])
        self.assertNotEqual(res_fam.family_of[("d.txt", "doc")], res_fam.family_of[("a.csv", "row:0")])

    def test_phase2_threshold_corroboration_vetoes_and_bucket_caps(self) -> None:
        digests = {f"f{i}.txt": f"d{i}" for i in range(10)}
        digests["same.txt"] = "dsame"

        # Cross-script skeleton + overlapping birth year passes between tau_low and tau_high.
        a = _ev(("f0.txt", "doc"), "Ramesh Kumar", sex="M", birth_years=(1980, 1982))
        b = _ev(("f1.txt", "doc"), "रमेश कुमार", sex="M", birth_years=(1981, 1983))
        # Exact boundary vs just below tau_low_bits:
        # With name=skeleton (~3.2 bits) + birth_interval=overlaps (~2.9 bits) + sex=agrees (~0.98 bits) + prior(-4.0) ~ 3.1 bits.
        res_pass = resolve(
            [a, b],
            _SpyKeys(),
            policy={"family_mode": "temporal_family", "tau_low_bits": 1.0, "tau_high_bits": 8.0},
            file_digests=digests,
        )
        self.assertIsNone(res_pass.patient_of[a.key])
        self.assertEqual(res_pass.family_of[a.key], res_pass.family_of[b.key])

        res_below = resolve(
            [a, b],
            _SpyKeys(),
            policy={"family_mode": "temporal_family", "tau_low_bits": 6.0, "tau_high_bits": 8.0},
            file_digests=digests,
        )
        self.assertNotEqual(res_below.family_of[a.key], res_below.family_of[b.key])

        # Same-file homonym without encounter context stays separate; with shared adm_key joins.
        h1 = _ev(("same.txt", "r0"), "Ramesh Kumar")
        h2 = _ev(("same.txt", "r1"), "Ramesh Kumar")
        res_hom = resolve(
            [h1, h2],
            _SpyKeys(),
            policy={"family_mode": "temporal_family", "tau_low_bits": -1.0, "tau_high_bits": 8.0},
            file_digests=digests,
        )
        self.assertNotEqual(res_hom.family_of[h1.key], res_hom.family_of[h2.key])

        h1_ctx = _ev(("same.txt", "r0"), "Ramesh Kumar", adm_key="ADM-9")
        h2_ctx = _ev(("same.txt", "r1"), "Ramesh Kumar", adm_key="ADM-9")
        res_ctx = resolve(
            [h1_ctx, h2_ctx],
            _SpyKeys(),
            policy={"family_mode": "temporal_family", "tau_low_bits": -1.0, "tau_high_bits": 8.0},
            file_digests=digests,
        )
        self.assertEqual(res_ctx.family_of[h1_ctx.key], res_ctx.family_of[h2_ctx.key])

        # Reject fuzzy-only name (no shared skeleton), shared-phone-only, conflicting relative, sex conflict.
        fz1 = _ev(("f2.txt", "doc"), "Ramesh Kumar", birth_years=(1980, 1980))
        fz2 = _ev(("f3.txt", "doc"), "Ramesha Kumar", birth_years=(1980, 1980))
        # Ensure we test distinct skeletons (e.g. "Sita" vs "Rita" or "Kiran" vs "Karan").
        fz1 = _ev(("f2.txt", "doc"), "Kiran", birth_years=(1980, 1980))
        fz2 = _ev(("f3.txt", "doc"), "Karan", birth_years=(1980, 1980))
        sp1 = _ev(("f4.txt", "doc"), "Asha", phones=["9876543210"], phone_shared=True)
        sp2 = _ev(("f5.txt", "doc"), "Asha", phones=["9876543210"], phone_shared=True)
        cr1 = _ev(("f6.txt", "doc"), "Mohan", kin="Sohan", birth_years=(1980, 1980))
        cr2 = _ev(("f7.txt", "doc"), "Mohan", kin="Rohan", birth_years=(1980, 1980))
        sx1 = _ev(("f8.txt", "doc"), "Kiran", sex="M", birth_years=(1980, 1980))
        sx2 = _ev(("f9.txt", "doc"), "Kiran", sex="F", birth_years=(1980, 1980))

        for x, y in [(fz1, fz2), (sp1, sp2), (cr1, cr2), (sx1, sx2)]:
            res_rej = resolve(
                [x, y],
                _SpyKeys(),
                policy={"family_mode": "temporal_family", "tau_low_bits": -2.0, "tau_high_bits": 8.0},
                file_digests=digests,
            )
            self.assertNotEqual(res_rej.family_of[x.key], res_rej.family_of[y.key])

        # Oversized same-file bucket respects max_bucket_size and is shuffle-invariant.
        big = [_ev(("same.txt", f"r{i}"), "Ramesh", adm_key="ADM-1") for i in range(6)]
        diag: dict[str, int] = {}
        r1 = resolve(
            big,
            _SpyKeys(),
            policy={"family_mode": "temporal_family", "max_bucket_size": 3, "tau_low_bits": -2.0, "tau_high_bits": 8.0},
            file_digests=digests,
            diagnostics=diag,
        )
        shuffled = list(big)
        random.Random(42).shuffle(shuffled)
        r2 = resolve(
            shuffled,
            _SpyKeys(),
            policy={"family_mode": "temporal_family", "max_bucket_size": 3, "tau_low_bits": -2.0, "tau_high_bits": 8.0},
            file_digests=digests,
        )
        self.assertEqual(r1, r2)
        self.assertGreater(diag["family_candidates_capped"], 0)
        # All stayed singletons because the oversized bucket was capped/abstained.
        self.assertEqual(len(set(r1.family_of.values())), 6)

    def test_phase3_veto_chain_budgets_anchors_and_duplicate_orphans(self) -> None:
        # A-B and B-C pass temporal gate, but A-C has a hard sex veto -> all three cannot join.
        a = _ev(("f1.txt", "doc"), "Kiran", sex="M", birth_years=(1980, 1980), kin="Mohan")
        b = _ev(("f2.txt", "doc"), "Kiran", birth_years=(1980, 1980), kin="Mohan")
        c = _ev(("f3.txt", "doc"), "Kiran", sex="F", birth_years=(1980, 1980), kin="Mohan")
        digests = {"f1.txt": "d1", "f2.txt": "d2", "f3.txt": "d3", "dup1.txt": "same", "dup2.txt": "same"}
        diag: dict[str, int] = {}
        res_chain = resolve(
            [c, b, a],
            _SpyKeys(),
            policy={"family_mode": "temporal_family", "tau_low_bits": 1.0, "tau_high_bits": 10.0},
            file_digests=digests,
            diagnostics=diag,
        )
        # A-B and B-C cannot all three be in the same family.
        self.assertNotEqual(res_chain.family_of[a.key], res_chain.family_of[c.key])
        self.assertGreaterEqual(diag["family_joins_skipped_veto"], 1)

        # Component cap = 2 prevents merging 3rd member into a 2-member strict cluster.
        sc1 = _ev(("f1.txt", "doc"), "Ramesh", sex="M", birth_years=(1980, 1980), ids={"uhid:h": ["1"]})
        sc2 = _ev(("f2.txt", "doc"), "Ramesh", sex="M", birth_years=(1980, 1980), ids={"uhid:h": ["1"]})
        sc3 = _ev(("f3.txt", "doc"), "Ramesh", sex="M", birth_years=(1980, 1980))
        diag_cap: dict[str, int] = {}
        res_cap = resolve(
            [sc1, sc2, sc3],
            _SpyKeys(),
            policy={"family_mode": "temporal_family", "max_component_size": 2},
            file_digests=digests,
            diagnostics=diag_cap,
        )
        self.assertEqual(res_cap.cluster_of[sc1.key], res_cap.cluster_of[sc2.key])
        self.assertNotEqual(res_cap.family_of[sc3.key], res_cap.family_of[sc1.key])
        self.assertGreaterEqual(diag_cap["family_joins_skipped_cap"], 1)

        # Pair-check budget exhaustion skips unverified joins.
        diag_bud: dict[str, int] = {}
        keys_merged = _SpyKeys()
        res_merged = resolve(
            [sc1, sc2, sc3],
            keys_merged,
            policy={
                "family_mode": "temporal_family",
                "tau_high_bits": 6.0,
                "tau_low_bits": 1.0,
                "max_pair_checks": 10,
            },
            file_digests=digests,
            diagnostics=diag_bud,
        )
        self.assertEqual(res_merged.family_of[sc1.key], res_merged.family_of[sc3.key])
        self.assertIsNone(res_merged.patient_of[sc3.key])  # Never upgrades null PSN
        self.assertIn(("shift", ("family", "id", "uhid:h", "1")), keys_merged.calls)
        for off in res_merged.shift_days.values():
            self.assertTrue(30 <= abs(off) <= 365)

        # Duplicate-byte orphan files retain distinct family_of even if offsets coincide.
        dup1 = _ev(("dup1.txt", "doc"), "Asha", birth_years=(1990, 1990))
        dup2 = _ev(("dup2.txt", "doc"), "Bina", birth_years=(1970, 1970))
        res_dup = resolve(
            [dup1, dup2],
            _SpyKeys(),
            policy={"family_mode": "temporal_family"},
            file_digests=digests,
        )
        self.assertNotEqual(res_dup.family_of[dup1.key], res_dup.family_of[dup2.key])
        self.assertEqual(
            res_dup.shift_days[res_dup.family_of[dup1.key]],
            res_dup.shift_days[res_dup.family_of[dup2.key]],
        )
