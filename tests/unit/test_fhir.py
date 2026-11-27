"""Unit tests for ABDM-aware FHIR adapter, attachment handling, and validator (D7, D8, D12)."""
from __future__ import annotations

import base64
import json
import unittest
from pathlib import Path

from kavachmed.adapters import fhir
from kavachmed.contracts import FileEntry, PrivacyPlan, Resolution
from kavachmed.policy import load_policy
from kavachmed.security import StageError, load_keys


def _sample_bundle() -> dict:
    att_text = "Referral letter for Asha Devi, phone 9876543210, visited on 2026-03-10."
    return {
        "resourceType": "Bundle",
        "id": "b-001",
        "meta": {
            "lastUpdated": "2026-03-12T10:15:30+05:30",
            "profile": ["https://nrces.in/ndhm/fhir/r4/StructureDefinition/DocumentBundle"],
            "source": "https://hospital.local/export/Asha_Devi_1234",
        },
        "type": "document",
        "timestamp": "2026-03-12T10:15:30+05:30",
        "entry": [
            {
                "fullUrl": "urn:uuid:comp-1",
                "resource": {
                    "resourceType": "Composition",
                    "id": "comp-1",
                    "status": "final",
                    "type": {"coding": [{"system": "http://snomed.info/sct", "code": "371530004"}]},
                    "subject": {"reference": "urn:uuid:pat-1", "display": "Asha Devi"},
                    "encounter": {"reference": "urn:uuid:enc-1"},
                    "date": "2026-03-12T10:00:00+05:30",
                    "author": [{"reference": "urn:uuid:prac-1", "display": "Dr. Ramesh Srivastava"}],
                    "title": "Discharge Summary",
                    "section": [{"title": "Diagnosis", "entry": [{"reference": "urn:uuid:cond-1"}]}],
                },
            },
            {
                "fullUrl": "urn:uuid:pat-1",
                "resource": {
                    "resourceType": "Patient",
                    "id": "pat-1",
                    "text": {"status": "generated", "div": "<div xmlns=\"http://www.w3.org/1999/xhtml\">Asha Devi, 45 Y/F, Hardoi 241001, 9876543210</div>"},
                    "identifier": [{"system": "https://hosp.example/uhid", "value": "SNJ-26-001234"}],
                    "name": [{"text": "Asha Devi", "given": ["Asha"], "family": "Devi"}],
                    "telecom": [{"system": "phone", "value": "9876543210", "use": "mobile"}],
                    "gender": "female",
                    "birthDate": "1981-05-14",
                    "address": [{"line": ["H.No 12"], "city": "Sandila", "district": "Hardoi", "state": "Uttar Pradesh", "postalCode": "241001", "country": "IN"}],
                    "contact": [{"name": {"text": "Kamal Rajput"}}],
                    "extension": [{"url": "https://raksha.example/fhir/StructureDefinition/mothers-name", "valueString": "Savitri Devi"}],
                },
            },
            {
                "fullUrl": "urn:uuid:prac-1",
                "resource": {
                    "resourceType": "Practitioner",
                    "id": "prac-1",
                    "name": [{"text": "Dr. Ramesh Srivastava"}],
                    "identifier": [{"system": "https://nmc.org.in", "value": "UPMC-55123"}],
                },
            },
            {
                "fullUrl": "urn:uuid:enc-1",
                "resource": {
                    "resourceType": "Encounter",
                    "id": "enc-1",
                    "status": "finished",
                    "class": {"system": "http://terminology.hl7.org/CodeSystem/v3-ActCode", "code": "IMP"},
                    "subject": {"reference": "urn:uuid:pat-1"},
                    "period": {"start": "2026-03-03T08:00:00+05:30", "end": "2026-03-12T10:00:00+05:30"},
                },
            },
            {
                "fullUrl": "urn:uuid:cond-1",
                "resource": {
                    "resourceType": "Condition",
                    "id": "cond-1",
                    "code": {"text": "Dengue fever"},
                    "subject": {"reference": "urn:uuid:pat-1"},
                    "recordedDate": "2026-03-04",
                },
            },
            {
                "fullUrl": "urn:uuid:doc-1",
                "resource": {
                    "resourceType": "DocumentReference",
                    "id": "doc-1",
                    "status": "current",
                    "subject": {"reference": "urn:uuid:pat-1"},
                    "content": [
                        {
                            "attachment": {
                                "contentType": "text/plain",
                                "data": base64.b64encode(att_text.encode("utf-8")).decode("ascii"),
                                "title": "Referral letter",
                            }
                        }
                    ],
                },
            },
            {
                "fullUrl": "urn:uuid:bin-1",
                "resource": {
                    "resourceType": "Binary",
                    "id": "bin-1",
                    "contentType": "image/jpeg",
                    "data": base64.b64encode(b"\xff\xd8\xff\xe0RAWJPEG").decode("ascii"),
                },
            },
        ],
    }


class TestFhirAdapter(unittest.TestCase):
    def test_projection_and_validator(self) -> None:
        pol = load_policy()
        keys = load_keys({"RAKSHA_SECRET": "test-secret-fhir"}, pbkdf2_iterations=100)
        raw = _sample_bundle()
        txt = json.dumps(raw)
        entry = FileEntry(Path("fhir/b.json"), "fhir/b.json", "abc", txt.encode("utf-8"), "json", "utf-8", txt)
        pf = fhir.parse(entry, pol, raw_obj=raw)
        rk = ("fhir/b.json", "doc")
        res = Resolution(
            patient_of={rk: "PSN-ABCDEF12"},
            cluster_of={rk: "C1"},
            family_of={rk: "F1"},
            shift_days={"F1": 30},
        )
        plan = PrivacyPlan(age_band={"C1": "40-49"}, location_level={rk: "district"}, pin_level={rk: "none"})
        assert pf.projector is not None
        out_bytes, locmap = pf.projector({"resolution": res, "plan": plan, "keys": keys, "policy": pol})
        out = json.loads(out_bytes.decode("utf-8"))

        self.assertEqual(out["meta"]["profile"], ["https://nrces.in/ndhm/fhir/r4/StructureDefinition/DocumentBundle"])
        self.assertEqual(out["meta"]["source"], "[SOURCE]")
        self.assertEqual(fhir.validate_fhir_bundle(out, expected_profiles=raw["meta"]["profile"]), [])
        self.assertTrue(len(locmap.clinical_ranges()) > 0)

        pat = [e["resource"] for e in out["entry"] if e["resource"]["resourceType"] == "Patient"][0]
        self.assertEqual(pat["name"], [{"text": "PSN-ABCDEF12"}])
        self.assertNotIn("birthDate", pat)
        self.assertNotIn("contact", pat)
        self.assertEqual(pat["address"], [{"district": "Hardoi", "state": "Uttar Pradesh", "country": "IN"}])
        self.assertNotIn("9876543210", pat["text"]["div"])
        self.assertNotIn("Asha Devi", pat["text"]["div"])
        self.assertNotIn("Savitri", json.dumps(pat))

        prac = [e["resource"] for e in out["entry"] if e["resource"]["resourceType"] == "Practitioner"][0]
        self.assertEqual(prac["name"][0]["text"], "Dr. Ramesh Srivastava")
        self.assertEqual(prac["identifier"][0]["value"], "[REG_NO]")

        cond = [e["resource"] for e in out["entry"] if e["resource"]["resourceType"] == "Condition"][0]
        self.assertEqual(cond["recordedDate"], "2026-04-03")

        docr = [e["resource"] for e in out["entry"] if e["resource"]["resourceType"] == "DocumentReference"][0]
        dec_att = base64.b64decode(docr["content"][0]["attachment"]["data"]).decode("utf-8")
        self.assertNotIn("Asha Devi", dec_att)
        self.assertNotIn("9876543210", dec_att)
        self.assertIn("2026-04-09", dec_att)

        binr = [e["resource"] for e in out["entry"] if e["resource"]["resourceType"] == "Binary"][0]
        self.assertEqual(base64.b64decode(binr["data"]), b"[BINARY_NEUTRALIZED]")

        # Contract 2-arg projector compatibility check
        out2, _ = pf.projector([], {rk: "PSN-99887766"})
        self.assertIn(b"PSN-99887766", out2)

    def test_validator_rejects_broken_ref(self) -> None:
        raw = _sample_bundle()
        raw["entry"][0]["resource"]["section"][0]["entry"][0]["reference"] = "urn:uuid:missing-999"
        errs = fhir.validate_fhir_bundle(raw)
        self.assertTrue(any("unresolved_ref" in e for e in errs))

    def test_media_drop_text_binary_and_dx_rollup(self) -> None:
        pol = load_policy()
        keys = load_keys({"RAKSHA_SECRET": "test-secret-fhir2"}, pbkdf2_iterations=100)
        raw = _sample_bundle()
        raw["entry"].append(
            {
                "fullUrl": "urn:uuid:bin-txt",
                "resource": {
                    "resourceType": "Binary",
                    "id": "bin-txt",
                    "contentType": "text/plain",
                    "data": base64.b64encode(b"Patient Asha Devi mobile 9876543210").decode("ascii"),
                },
            }
        )
        raw["entry"].append(
            {
                "fullUrl": "urn:uuid:med-1",
                "resource": {
                    "resourceType": "Media",
                    "id": "med-1",
                    "status": "completed",
                    "content": {"contentType": "image/png", "data": base64.b64encode(b"PNGDATA").decode("ascii")},
                },
            }
        )
        txt = json.dumps(raw)
        entry = FileEntry(Path("fhir/b2.json"), "fhir/b2.json", "def", txt.encode("utf-8"), "json", "utf-8", txt)
        pf = fhir.parse(entry, pol, raw_obj=raw)
        rk = ("fhir/b2.json", "doc")
        res = Resolution(patient_of={rk: "PSN-ZZZZ1111"}, cluster_of={rk: "C1"}, family_of={rk: "F1"}, shift_days={"F1": 35})
        plan = PrivacyPlan(
            age_band={"C1": "40-49"},
            location_level={rk: "state"},
            dx_rollup={rk: "Infectious disease"},
            suppress_qi={rk},
        )
        assert pf.projector is not None
        out_bytes, _ = pf.projector({"resolution": res, "plan": plan, "keys": keys, "policy": pol})
        out = json.loads(out_bytes.decode("utf-8"))

        pat = [e["resource"] for e in out["entry"] if e["resource"]["resourceType"] == "Patient"][0]
        self.assertEqual(pat["gender"], "unknown")
        cond = [e["resource"] for e in out["entry"] if e["resource"]["resourceType"] == "Condition"][0]
        self.assertEqual(cond["code"]["text"], "Infectious disease")
        bin_txt = [e["resource"] for e in out["entry"] if e["resource"]["id"] == "bin-txt"][0]
        dec_txt = base64.b64decode(bin_txt["data"]).decode("utf-8")
        self.assertNotIn("Asha Devi", dec_txt)
        self.assertNotIn("9876543210", dec_txt)
        med = [e["resource"] for e in out["entry"] if e["resource"]["resourceType"] == "Media"][0]
        self.assertEqual(med["status"], "entered-in-error")
        self.assertNotIn("content", med)

        raw["entry"][-2]["resource"]["data"] = "not!valid!b64!!!"
        self.assertIn("Binary_invalid_base64", fhir.validate_fhir_bundle(raw))


if __name__ == "__main__":
    unittest.main()
