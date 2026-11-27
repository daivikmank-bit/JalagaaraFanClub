"""Failing-first unit tests for Plan B Phase 1 (synthetic data only)."""

from __future__ import annotations

import json
from pathlib import Path

from kavachmed.contracts import Context, Field, Record
from kavachmed.detect.address import segment_address
from kavachmed.detect.context import detect_context
from kavachmed.detect.default_deny import detect_default_deny
from kavachmed.detect.ids import detect_ids
from kavachmed.shield import protect
from kavachmed.typing import type_columns


def _rec(text: str, path: str = "note", meta: dict | None = None) -> Record:
    return Record(key=("synthetic/doc1.txt", "doc"), fields=[Field(idx=0, path=path, text=text)], meta=meta or {})


def _ctx(**kwargs) -> Context:
    return Context(
        policy={
            "detect": {
                "stack": "recall_first",
                "vehicle_plates": True,
                "doctor_reg_remove_digits": True,
                "default_deny": {"score": 0.45},
            }
        },
        **kwargs,
    )


# --- B1: Semantic column/key typing ---

def test_b1_typing_latin_romanized_devanagari_and_chaos():
    records = [
        Record(
            key=("syn/reg.csv", "row:0"),
            fields=[
                Field(idx=0, path="Naam", text="Aarav Mehta"),
                Field(idx=1, path="Umar", text="42"),
                Field(idx=2, path="Ling", text="M"),
                Field(idx=3, path="col_x9", text="9876543210"),
                Field(idx=4, path="col_y2", text="2345 6789 0123"),
                Field(idx=5, path="पता", text="H.No. 14, Ward 3, Sitapur"),
            ],
        ),
        Record(
            key=("syn/reg.csv", "row:1"),
            fields=[
                Field(idx=0, path="Naam", text="Priya Nair"),
                Field(idx=1, path="Umar", text="35"),
                Field(idx=2, path="Ling", text="F"),
                Field(idx=3, path="col_x9", text="+91 91234-56789"),
                Field(idx=4, path="col_y2", text="91-1234-5678-9012"),
                Field(idx=5, path="पता", text="Mohalla Samarth Nagar, PIN 261001"),
            ],
        ),
    ]
    coltypes = type_columns(records)
    assert coltypes["Naam"] == "name"
    assert coltypes["Umar"] == "age"
    assert coltypes["Ling"] == "sex"
    assert coltypes["col_x9"] == "phone"
    assert coltypes["col_y2"] in {"aadhaar", "abha", "id"}
    assert coltypes["पता"] == "address"


# --- B2: Typed identifiers (FR-D1) ---

def test_b2_aadhaar_verhoeff_optional_and_clinical_safe():
    # Invalid Verhoeff checksum Aadhaar must still be caught
    rec = _rec("Aadhaar: 2345 6789 0124, masked XXXX-XXXX-4321, ABHA 91-1234-5678-9012, abha@abdm")
    spans = detect_ids(rec, _ctx())
    typs = {s.typ for s in spans}
    assert "AADHAAR" in typs
    assert "AADHAAR_MASKED" in typs
    assert "ABHA_NUMBER" in typs
    assert "ABHA_ADDRESS" in typs

    # Clinical numbers must NOT be flagged
    clin = _rec("Tab Dolo 650 mg 1-0-1 x 5 days; BP 120/80; ref range 70-140; LabSeq 234567890123 mmol/L")
    clin_spans = detect_ids(clin, _ctx())
    assert not any(s.typ in {"PHONE", "AADHAAR", "MRN"} for s in clin_spans)


def test_b2_phones_pan_email_uhid():
    rec = _rec("Ph: +91 98765 43210, alt 09123456789, deva ९८७६५४३२१०, PAN ABCDE1234F, email test.user@example.org, UHID: SNJ/26/001234")
    spans = detect_ids(rec, _ctx())
    typs = {s.typ for s in spans}
    assert {"PHONE", "PAN", "EMAIL", "MRN"}.issubset(typs)
    phones = [s for s in spans if s.typ == "PHONE"]
    assert len(phones) >= 3


# --- B3: FR-D9 extended shapes ---

def test_b3_extended_shapes_and_negative_context():
    rec = _rec(
        "VID 2345678901234567, EPIC ABC1234567, GSTIN 27ABCDE1234F1Z5, "
        "UPI user@oksbi, URL https://hospital.example.org/pt, IP 192.168.1.45, "
        "Veh KA01AB1234, Dr. Reg No. MCI-45678"
    )
    spans = detect_ids(rec, _ctx())
    matched_texts = {rec.fields[0].text[s.start:s.end] for s in spans}
    assert any("2345678901234567" in t for t in matched_texts)
    assert any("ABC1234567" in t for t in matched_texts)
    assert any("27ABCDE1234F1Z5" in t for t in matched_texts)
    assert any("user@oksbi" in t for t in matched_texts)
    assert any("KA01AB1234" in t for t in matched_texts)
    assert any("45678" in t for t in matched_texts)
    assert not any("Dr." in t for t in matched_texts)


# --- B4: Clinical Shield (FR-D4) ---

def test_b4_clinical_shield_locks_and_co_resolver():
    ctx = _ctx(doctor_registry={"Ramesh Srivastava", "Ananya Deshmukh"})
    rec = _rec(
        "Dr. Ramesh Srivastava advised Tab Dolo 650 mg 1-0-1 x 5 days. "
        "K/C/O DM2 x 9 yrs. H/O TB 9 yrs back. C/O fever for 3 days. "
        "Parkinson's disease. Ward 9 (Medicine). Referred by ASHA worker. "
        "C/O Mahesh Kumar (brother)."
    )
    shield_spans = protect(rec, ctx)
    protected_texts = [rec.fields[0].text[s.start:s.end] for s in shield_spans if s.protected]
    exempt_kinds = {s.exempt_kind for s in shield_spans if s.protected}

    assert any("Ramesh Srivastava" in t for t in protected_texts)
    assert any("1-0-1" in t or "Dolo 650" in t for t in protected_texts)
    assert any("fever" in t.lower() for t in protected_texts)
    assert any("Parkinson" in t for t in protected_texts)
    assert any("Ward 9" in t for t in protected_texts)
    assert any("ASHA worker" in t for t in protected_texts)
    assert "doctor_registry" in exempt_kinds
    assert "dose" in exempt_kinds or "drug" in exempt_kinds
    assert "history" in exempt_kinds
    assert "eponym" in exempt_kinds
    assert "ward" in exempt_kinds
    # C/O Mahesh Kumar must NOT be shielded
    assert not any("Mahesh Kumar" in t for t in protected_texts)


# --- B5: Safe set S \ N ---

def test_b5_safe_set_name_subtraction():
    safe_path = Path(__file__).resolve().parents[2] / "config" / "lexicons" / "safe" / "safe_words.txt"
    assert safe_path.exists(), "safe_words.txt must exist"
    safe_words = {line.strip().lower() for line in safe_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    for forbidden in ("anita", "aurora", "कुमार", "कपूर"):
        assert forbidden not in safe_words
    for expected in ("fever", "hypertension", "mg", "january"):
        assert expected in safe_words


# --- B6: Context grammar ---

def test_b6_context_grammar_multilingual():
    rec = _rec(
        "Pt. Name: Kavita Joshi\n"
        "नाम: सुरेश वर्मा\n"
        "S/o Mohanlal Joshi\n"
        "Kamal Bhaiya: please send report\n"
        "Rohan ka BP 130/80 hai\n"
        "Witnessed by Sunita Devi"
    )
    spans = detect_context(rec, _ctx())
    texts_roles = [(rec.fields[0].text[s.start:s.end], s.role) for s in spans]
    assert any("Kavita Joshi" in t and r == "patient" for t, r in texts_roles)
    assert any("सुरेश वर्मा" in t and r == "patient" for t, r in texts_roles)
    assert any("Mohanlal Joshi" in t and r == "relative" for t, r in texts_roles)
    assert any("Kamal Bhaiya" in t for t, _ in texts_roles)
    assert any("Rohan" in t for t, _ in texts_roles)
    assert any("Sunita Devi" in t and r == "third_party" for t, r in texts_roles)


# --- B8: Default-deny (FR-D11) ---

def test_b8_default_deny_unlabelled_names():
    safe_path = Path(__file__).resolve().parents[2] / "config" / "lexicons" / "safe" / "safe_words.txt"
    safe_words = {line.strip().lower() for line in safe_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    ctx = _ctx(lexicons={"safe_words": safe_words})
    rec = _rec("Zoravar Singhania presented with fever and cough. देवेंद्र राठौड़ को बुखार है।")
    shield_spans = protect(rec, ctx)
    dd_spans = detect_default_deny(rec, ctx, shield_spans)
    dd_texts = [rec.fields[0].text[s.start:s.end] for s in dd_spans]
    assert any("Zoravar Singhania" in t for t in dd_texts)
    assert any("देवेंद्र राठौड़" in t for t in dd_texts)
    assert not any("fever" in t.lower() for t in dd_texts)
    assert all(s.src == "default_deny" and s.score == 0.45 for s in dd_spans)


# --- B9: Address segmenter ---

def test_b9_address_segmenter_keeps_district():
    rec = _rec("C/O Ramesh Verma, H.No. 42, Ward 7, Mohalla Raja Ekansh, near Pani Tanki, Village Durgamau, Tehsil Sadar, District Sitapur, PIN 261001")
    spans = segment_address(rec, _ctx())
    typs = {s.typ for s in spans}
    assert {"RELATIVE_NAME", "HOUSE", "LOCALITY", "LANDMARK", "VILLAGE", "PIN"}.issubset(typs)
    texts = " ".join(rec.fields[0].text[s.start:s.end] for s in spans)
    assert "Sitapur" not in texts
