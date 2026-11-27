import unittest

from kavachmed.evidence import Evidence
from kavachmed.indic import normalize_name
from kavachmed.linkage import LinkageError
from kavachmed.linkage.strict import build_strict_clusters


class TestStrictLinkage(unittest.TestCase):
    def test_bridge_eligibility_and_determinism(self) -> None:
        # Same scoped UHID across CSV row and text visit links; different issuer stays separate.
        ev_csv = Evidence(
            key=("reg/a.csv", "row:1"),
            names=list(normalize_name("Ramesh Kumar")),
            sex="M",
            ids={"uhid:snj": ["019672"]},
        )
        ev_txt = Evidence(
            key=("notes/v1.txt", "doc"),
            names=list(normalize_name("रमेश कुमार")),
            sex="M",
            ids={"uhid:snj": ["019672"]},
        )
        ev_other_issuer = Evidence(
            key=("notes/v2.txt", "doc"),
            names=list(normalize_name("Ramesh Kumar")),
            sex="M",
            ids={"uhid:apollo": ["019672"]},
        )
        # Masked ID fragment, unscoped insurance ID, and year-reused UHID must not bridge.
        ev_masked_1 = Evidence(
            key=("m1.txt", "doc"),
            ids={"aadhaar_last4": ["1234"], "ins": ["POL999"], "uhid:yrhosp": ["0001"]},
        )
        ev_masked_2 = Evidence(
            key=("m2.txt", "doc"),
            ids={"aadhaar_last4": ["1234"], "ins": ["POL999"], "uhid:yrhosp": ["0001"]},
        )

        items = [ev_txt, ev_other_issuer, ev_masked_1, ev_csv, ev_masked_2]
        policy = {"year_reused_uhid_issuers": ["yrhosp"]}
        res1 = build_strict_clusters(items, policy=policy)
        res2 = build_strict_clusters(list(reversed(items)), policy=policy)

        self.assertEqual(res1, res2)
        self.assertEqual(len(res1.clusters), 1)
        self.assertEqual(
            res1.clusters[0].members,
            (("notes/v1.txt", "doc"), ("reg/a.csv", "row:1")),
        )
        self.assertEqual(res1.clusters[0].canonical_key, (0, "uhid:snj", "019672"))

        # Duplicate RecKey rejected with text-free reason code.
        with self.assertRaises(LinkageError) as ctx:
            build_strict_clusters([ev_csv, ev_csv])
        self.assertEqual(ctx.exception.code, "duplicate_record_key")
        self.assertNotIn("reg/a.csv", str(ctx.exception))

    def test_admission_bridge_conflicts_cross_bucket_and_budget(self) -> None:
        # Trusted admission-only bridge
        adm_a = Evidence(key=("a.txt", "doc"), adm_key="snj:ipd01", sex="F")
        adm_b = Evidence(key=("b.txt", "doc"), adm_key="snj:ipd01", birth_years=(1990, 1991))

        # Old-card / contested ID (sex + dissimilar names)
        old_1 = Evidence(
            key=("o1.txt", "doc"),
            names=list(normalize_name("Harish Pandey")),
            sex="M",
            birth_years=(1965, 1966),
            ids={"aadhaar": ["111122223333"]},
        )
        old_2 = Evidence(
            key=("o2.txt", "doc"),
            names=list(normalize_name("Sunita Devi")),
            sex="F",
            birth_years=(1998, 1999),
            ids={"aadhaar": ["111122223333"]},
        )

        # A-B-C chain where A-B share PAN and B-C share ABHA, but A and C conflict in birth_years.
        chain_a = Evidence(
            key=("c1.txt", "doc"),
            birth_years=(1960, 1961),
            ids={"pan": ["ABCDE1234F"]},
        )
        chain_b = Evidence(
            key=("c2.txt", "doc"),
            ids={"pan": ["ABCDE1234F"], "abha": ["12345678901234"]},
        )
        chain_c = Evidence(
            key=("c3.txt", "doc"),
            birth_years=(1995, 1996),
            ids={"abha": ["12345678901234"]},
        )

        diag: dict[str, int] = {}
        res = build_strict_clusters(
            [adm_a, adm_b, old_1, old_2, chain_a, chain_b, chain_c],
            diagnostics=diag,
        )
        self.assertEqual(len(res.clusters), 1)
        self.assertEqual(res.clusters[0].canonical_key, (1, "adm", "snj:ipd01"))
        self.assertEqual(diag["contested_bridges"], 1)
        self.assertEqual(diag["cross_bucket_veto_abstentions"], 1)

        # Exact groups may exceed the fuzzy pair budget, but not the component cap.
        over_diag: dict[str, int] = {}
        many = [
            Evidence(key=(f"f{i}.txt", "doc"), ids={"aadhaar": ["999988887777"]})
            for i in range(80)
        ]
        large = build_strict_clusters(many, policy={"max_pair_checks": 5})
        self.assertEqual(len(large.clusters[0].members), 80)
        over_res = build_strict_clusters(many, policy={"max_component_size": 79}, diagnostics=over_diag)
        self.assertEqual(over_res.clusters, ())
        self.assertEqual(over_diag["over_budget_abstentions"], 1)
        self.assertEqual(over_res.fuzzy_ineligible, {ev.key for ev in many})

        for attr, left, right in (("sex", "M", "F"), ("birth_years", (1980, 1981), (1990, 1992))):
            with self.subTest(attr=attr):
                setattr(many[0], attr, left)
                setattr(many[-1], attr, right)
                vetoed = build_strict_clusters(many, policy={"max_pair_checks": 5})
                self.assertFalse(vetoed.clusters)
                self.assertEqual(vetoed.fuzzy_ineligible, over_res.fuzzy_ineligible)
                setattr(many[0], attr, None)
                setattr(many[-1], attr, None)

    def test_shared_phone_never_bridges_or_vetoes_and_never_mutates(self) -> None:
        p1 = Evidence(
            key=("p1.txt", "doc"),
            names=list(normalize_name("Asha Sharma")),
            phones=["9876543210"],
            ids={"uhid:snj": ["100"]},
        )
        p2 = Evidence(
            key=("p2.txt", "doc"),
            names=list(normalize_name("Asha Sharma")),
            phones=["9123456789"],  # phone changed, valid exact bridge still links
            ids={"uhid:snj": ["100"]},
        )
        p3 = Evidence(
            key=("p3.txt", "doc"),
            names=list(normalize_name("Kamal Verma")),
            phones=["9876543210"],  # shares phone with Asha, incompatible name
        )

        diag: dict[str, int] = {}
        res = build_strict_clusters([p1, p2, p3], diagnostics=diag)
        self.assertEqual(len(res.clusters), 1)
        self.assertEqual(res.clusters[0].members, (("p1.txt", "doc"), ("p2.txt", "doc")))
        self.assertEqual(res.shared_phones, ("9876543210",))
        self.assertEqual(res.phone_shared_records, (("p1.txt", "doc"), ("p3.txt", "doc")))
        self.assertEqual(diag["shared_phones"], 1)
        # Input Evidence objects remain unmutated.
        self.assertFalse(p1.phone_shared)
        self.assertFalse(p3.phone_shared)


if __name__ == "__main__":
    unittest.main()
