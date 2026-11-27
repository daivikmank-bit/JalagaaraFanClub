"""Canonical ICD-10 <-> English <-> Devanagari diagnosis concept map and rarity rollup (D13, FR-P6)."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

_CONCEPT_TABLE: list[tuple[str, str, str, str, tuple[str, ...]]] = [
    ("ASTHMA", "J45", "Respiratory disease", "Asthma", ("asthma", "bronchial asthma", "दमा", "j45", "j45.9")),
    ("COPD", "J44", "Respiratory disease", "Chronic obstructive pulmonary disease", ("copd", "chronic bronchitis", "emphysema", "j44", "j44.9")),
    ("PNEUMONIA", "J18", "Respiratory disease", "Pneumonia", ("pneumonia", "निमोनिया", "j18", "j18.9")),
    ("URI", "J06", "Respiratory disease", "Upper respiratory infection", ("uri", "urti", "acute uri", "सर्दी खांसी", "j06", "j06.9")),
    ("DM2", "E11", "Endocrine/metabolic disease", "Type 2 diabetes mellitus", ("dm2", "dm", "t2dm", "diabetes", "diabetes mellitus", "type 2 diabetes", "मधुमेह", "शुगर", "e11", "e11.9")),
    ("HYPOTHYROID", "E03", "Endocrine/metabolic disease", "Hypothyroidism", ("hypothyroidism", "hypothyroid", "e03", "e03.9")),
    ("ADDISON", "E27", "Endocrine/metabolic disease", "Adrenal disorder", ("addison's disease", "addison disease", "e27.1", "e27")),
    ("HTN", "I10", "Circulatory disease", "Hypertension", ("hypertension", "htn", "high bp", "उच्च रक्तचाप", "i10")),
    ("CAD", "I25", "Circulatory disease", "Ischemic heart disease", ("cad", "coronary artery disease", "ihd", "angina", "i25", "i25.9")),
    ("STROKE", "I64", "Circulatory disease", "Cerebrovascular disease", ("stroke", "cva", "cerebrovascular accident", "पक्षघात", "i64")),
    ("TB", "A15", "Infectious disease", "Tuberculosis", ("tuberculosis", "tb", "pulmonary tb", "क्षय रोग", "टीबी", "a15", "a16")),
    ("MALARIA", "B54", "Infectious disease", "Malaria", ("malaria", "मलेरिया", "b50", "b54")),
    ("KALA_AZAR", "B55", "Infectious disease", "Leishmaniasis", ("kala-azar", "kala azar", "visceral leishmaniasis", "काला-अज़ार", "कालाजार", "b55", "b55.0")),
    ("DENGUE", "A90", "Infectious disease", "Dengue fever", ("dengue", "dengue fever", "डेंगू", "a90", "a91")),
    ("TYPHOID", "A01", "Infectious disease", "Typhoid fever", ("typhoid", "enteric fever", "टाइफाइड", "a01", "a01.0")),
    ("GASTRO", "A09", "Infectious disease", "Gastroenteritis", ("gastroenteritis", "acute gastroenteritis", "age", "diarrhea", "दस्त", "a09")),
    ("ANEMIA", "D64", "Hematologic disease", "Anemia", ("anemia", "anaemia", "iron deficiency anemia", "एनीमिया", "रक्ताल्पता", "d50", "d64", "d64.9")),
    ("HODGKIN", "C81", "Neoplasm", "Lymphoma", ("hodgkin's lymphoma", "hodgkin lymphoma", "c81")),
    ("BELL_PALSY", "G51", "Nervous system disease", "Facial nerve disorder", ("bell's palsy", "bells palsy", "g51.0", "g51")),
    ("CKD", "N18", "Genitourinary disease", "Chronic kidney disease", ("ckd", "chronic kidney disease", "n18", "n18.9")),
    ("UTI", "N39", "Genitourinary disease", "Urinary tract infection", ("uti", "urinary tract infection", "n39.0")),
    ("APPENDICITIS", "K35", "Digestive disease", "Appendicitis", ("appendicitis", "acute appendicitis", "k35")),
    ("JAUNDICE", "R17", "Digestive/hepatic condition", "Jaundice", ("jaundice", "hyperbilirubinemia", "पीलिया", "r17")),
]

_ICD_RE = re.compile(r"\b([A-TV-Z]\d{2})(?:\.\d{1,2})?\b")

_SYNONYM_TO_CONCEPT: dict[str, tuple[str, str, str, str]] = {}
for cid, icd3, chap, en, syns in _CONCEPT_TABLE:
    for s in syns:
        _SYNONYM_TO_CONCEPT[unicodedata.normalize("NFKC", s).casefold().strip()] = (cid, icd3, chap, en)

_ICD_CHAPTERS = [
    ("A", "B", "Infectious disease"),
    ("C", "D", "Neoplasm or blood disorder"),
    ("E", "E", "Endocrine/metabolic disease"),
    ("F", "F", "Mental/behavioral disorder"),
    ("G", "G", "Nervous system disease"),
    ("H", "H", "Eye/ear disorder"),
    ("I", "I", "Circulatory disease"),
    ("J", "J", "Respiratory disease"),
    ("K", "K", "Digestive disease"),
    ("L", "L", "Skin disorder"),
    ("M", "M", "Musculoskeletal disorder"),
    ("N", "N", "Genitourinary disease"),
    ("O", "O", "Obstetric condition"),
    ("R", "R", "Clinical symptom/finding"),
    ("S", "T", "Injury/poisoning"),
]


def _chapter_for_icd(icd3: str) -> str:
    ch = icd3[:1].upper()
    for lo, hi, label in _ICD_CHAPTERS:
        if lo <= ch <= hi:
            return label
    return "Clinical condition"


def canonicalize_dx(text: str) -> tuple[str, str, str]:
    """Map diagnosis string or ICD-10 code to (concept_id, icd3, chapter_label)."""
    norm = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    if norm in _SYNONYM_TO_CONCEPT:
        cid, icd3, chap, _en = _SYNONYM_TO_CONCEPT[norm]
        return (cid, icd3, chap)

    for syn, info in sorted(_SYNONYM_TO_CONCEPT.items(), key=lambda kv: -len(kv[0])):
        if len(syn) >= 3 and re.search(rf"(?<!\w){re.escape(syn)}(?!\w)", norm):
            return (info[0], info[1], info[2])

    m = _ICD_RE.search(text.upper())
    if m:
        icd3 = m.group(1)
        return (f"ICD:{icd3}", icd3, _chapter_for_icd(icd3))

    return (f"RAW:{norm}", "UNK", "Clinical condition")


def plan_dx_rollups(dx_by_patient: dict[str, list[str]], k_dx: int = 3) -> dict[str, dict[str, str]]:
    """Count frequency over canonical concepts across patients; return rollup label for rare concepts."""
    concept_patients: Counter[str] = Counter()
    patient_concepts: dict[str, list[tuple[str, str, str, str]]] = {}

    for pid, terms in sorted(dx_by_patient.items()):
        seen_cids: set[str] = set()
        entries: list[tuple[str, str, str, str]] = []
        for t in terms:
            cid, icd3, chap = canonicalize_dx(t)
            entries.append((t, cid, icd3, chap))
            if cid not in seen_cids:
                concept_patients[cid] += 1
                seen_cids.add(cid)
        patient_concepts[pid] = entries

    rollups: dict[str, dict[str, str]] = {}
    for pid, entries in patient_concepts.items():
        mapping: dict[str, str] = {}
        for raw_t, cid, _icd3, chap in entries:
            if concept_patients[cid] < k_dx:
                mapping[raw_t] = chap
        if mapping:
            rollups[pid] = mapping
    return rollups
