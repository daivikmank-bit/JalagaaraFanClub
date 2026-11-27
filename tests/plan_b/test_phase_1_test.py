"""Synthetic Plan B phase 1 boundary test."""

import pytest

from kavachmed.contracts import Context, Field, Record
from kavachmed.detect.arbiter import detect
from kavachmed.detect.ids import detect_ids
from kavachmed.shield import protect


def test_clinical_value_next_to_identifier():
    ctx = Context()
    text = "BP 120/80; phone 9876543210"
    rec = Record(key=("synthetic/phase1.txt", "doc"), fields=[Field(0, "doc", text)])
    spans = detect(rec, ctx, protect(rec, ctx))
    phone_start = text.index("9876543210")
    bp_end = len("BP 120/80")
    assert any(s.typ == "PHONE" and s.start <= phone_start and s.end >= len(text) for s in spans)
    assert not any(s.start < bp_end and s.end > 0 for s in spans)


@pytest.mark.parametrize("text", ["Dose 1-0-1", "Lab Seq 98XXXXXX10 mmol/L", "code 12XXXXXX10"])
def test_masked_phone_rule_does_not_flag_unrelated_values(text):
    rec = Record(key=("synthetic/clinical.txt", "doc"), fields=[Field(0, "doc", text)])
    assert not any(s.typ == "PHONE" for s in detect_ids(rec, Context()))


def test_masked_phone_near_clinical_value():
    text = "BP 120/80; Phone: 98XXXXXX10"
    rec = Record(key=("synthetic/masked-phone.txt", "doc"), fields=[Field(0, "doc", text)])
    spans = detect(rec, Context())
    assert any(s.typ == "PHONE" and text[s.start:s.end] == "98XXXXXX10" for s in spans)
    assert not any(s.start < len("BP 120/80") and s.end > 0 for s in spans)


def test_complaint_shield_does_not_hide_following_patient():
    ctx = Context()
    text = "C/O fever for 3 days patient Velora Neris"
    rec = Record(key=("synthetic/complaint.txt", "doc"), fields=[Field(0, "doc", text)])
    shield = protect(rec, ctx)
    name_start = text.index("Velora")
    assert any(text[s.start:s.end] == "C/O fever for 3 days" for s in shield)
    assert not any(s.start < len(text) and s.end > name_start for s in shield)
    assert any(s.start <= name_start and s.end >= len(text) for s in detect(rec, ctx, shield))


def test_labelled_aadhaar_next_to_vital_and_compressed_ipv6():
    text = "Aadhaar: 2345 6789 0124; BP 120/80; IP ::1; IP 2001:db8::; IP ::ffff:192.0.2.1. time 12:34"
    rec = Record(key=("synthetic/ids.txt", "doc"), fields=[Field(0, "doc", text)])
    found = {(text[s.start:s.end], s.typ) for s in detect_ids(rec, Context())}
    assert ("2345 6789 0124", "AADHAAR") in found
    assert ("::1", "IP_ADDRESS") in found
    assert ("2001:db8::", "IP_ADDRESS") in found
    assert ("::ffff:192.0.2.1", "IP_ADDRESS") in found
    assert ("12:34", "IP_ADDRESS") not in found


def test_labelled_local_id_with_single_digit_component():
    text = "UHID: XY/7/123456"
    rec = Record(key=("synthetic/local-id.txt", "doc"), fields=[Field(0, "doc", text)])
    assert any(text[s.start:s.end] == "XY/7/123456" and s.typ == "MRN" for s in detect(rec, Context()))


def test_unlabelled_compact_ids_and_clinical_package_code():
    ctx = Context()
    text = "Ref ZX-9876543; Ref QRS654321"
    rec = Record(key=("synthetic/compact-id.txt", "doc"), fields=[Field(0, "doc", text)])
    spans = detect(rec, ctx)
    for identifier in ("ZX-9876543", "QRS654321"):
        start = text.index(identifier)
        assert any(s.typ == "MRN" and s.start <= start and s.end >= start + len(identifier) for s in spans)
    for clinical in ("package code PKG-987654", "PKG-987654"):
        rec = Record(key=("synthetic/package.txt", "doc"), fields=[Field(0, "doc", clinical)])
        assert not any(s.typ == "MRN" for s in detect_ids(rec, ctx))


@pytest.mark.parametrize("honorific", ["ji", "sahab", "didi", "bhai", "babu", "amma", "babaji"])
def test_name_with_honorific_is_fully_detected(honorific):
    text = f"Zoravar {honorific} ka BP 120/80"
    name = f"Zoravar {honorific}"
    rec = Record(key=("synthetic/name.txt", "doc"), fields=[Field(0, "doc", text)])
    spans = detect(rec, Context())
    assert all(any(s.start <= i < s.end for s in spans) for i in range(len(name)) if name[i].isalnum())
    assert not any(s.start < len(text) and s.end > text.index("BP 120/80") for s in spans)


@pytest.mark.parametrize("lab", [
    "Haemoglobin 12.8 g/dL",
    "Hemoglobin 13 g/dL",
    "Random Blood Sugar 102 mg/dL",
    "Serum Creatinine 0.9 mg/dL",
    "Platelet count 250000",
])
def test_lab_labels_kept_next_to_patient_and_identifier(lab):
    text = f"Patient Name: Zoravar Neris; {lab}; Phone: 9876543210"
    rec = Record(key=("synthetic/lab.txt", "doc"), fields=[Field(0, "doc", text)])
    spans = detect(rec, Context())
    start = text.index(lab)
    assert not any(s.start < start + len(lab) and s.end > start for s in spans)
    assert any("Zoravar Neris" in text[s.start:s.end] for s in spans)
    assert any(s.typ == "PHONE" and text[s.start:s.end] == "9876543210" for s in spans)


@pytest.mark.parametrize("name", ["Serum", "Haemoglobin", "Platelet"])
def test_lab_word_in_patient_name_field_is_not_shielded(name):
    text = f"Patient Name: {name}"
    rec = Record(key=("synthetic/collision.txt", "doc"), fields=[Field(0, "doc", text)])
    spans = detect(rec, Context())
    assert any(text[s.start:s.end] == name and s.typ == "PATIENT_NAME" for s in spans)


def test_lowercase_honorific_name_still_detected():
    text = "zoravar ji ka BP 120/80"
    rec = Record(key=("synthetic/lowercase.txt", "doc"), fields=[Field(0, "doc", text)])
    assert any(text[s.start:s.end] == "zoravar ji" for s in detect(rec, Context()))
