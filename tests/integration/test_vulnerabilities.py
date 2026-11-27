"""Adversarial regression suite: evasion, hostile inputs, side channels (synthetic data only)."""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import unicodedata
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _run(inp: Path, out: Path, tmp: Path, secret: str | None = "vuln-regression-secret") -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "RAKSHA_SECRET"}
    env.update({"TMPDIR": str(tmp), "HOME": str(tmp), "PYTHONDONTWRITEBYTECODE": "1"})
    if secret is not None:
        env["RAKSHA_SECRET"] = secret
    return subprocess.run([sys.executable, str(REPO / "main.py"), "--input", str(inp), "--output", str(out)],
                          cwd=REPO, env=env, capture_output=True, text=True, timeout=600)


def _all_text(out: Path) -> str:
    parts = []
    for f in out.rglob("*"):
        parts.append(str(f.relative_to(out)))
        if f.is_file():
            parts.append(f.read_text(encoding="utf-8", errors="ignore"))
    t = unicodedata.normalize("NFKC", " ".join(parts)).replace("​", "")
    return t.translate(str.maketrans("ѕаоіЅАОІ", "saoiSAOI")).casefold()


CASES = [
    ("n1.txt", "Brought in by her daughter Tejaswini Kharbanda after a fall.\n", ["Kharbanda"]),
    ("n2.txt", "मरीज़ को उनके बेटे विभूतिनारायण ठकराल लेकर आए।\n", ["विभूतिनारायण", "ठकराल"]),
    ("n3.txt", "rukmini ko 3 din se bukhar hai, dawai di.\n", ["rukmini"]),
    ("n4.txt", "Patient Name: Ѕаrоjіnі Bhatnagar\n", ["bhatnagar"]),
    ("n5.txt", "Name: Prav​ee​na Lalwani\n", ["lalwani"]),
    ("p1.txt", "Contact on ९८७६५४३२१० for reports\n", ["९८७६५४३२१०", "9876543210"]),
    ("p2.txt", "call 98 76 51 23 40 if fever rises\n", ["98 76 51 23 40"]),
    ("d1.txt", "Accompanied by Dr. Anuradha (daughter of patient), not the treating doctor.\n", ["anuradha"]),
    ("k1.json", json.dumps({"Damayanti Rawat": {"bp": "120/80"}}), ["damayanti"]),
    ("h1.csv", "time,UHID-445566 Gurpreet Sandhu\n08:00,120/80\n", ["gurpreet", "445566"]),
    ("Discharge_Nirmala_Kandpal_2026.txt", "Discharged stable.\n", ["nirmala", "kandpal"]),
    ("r1.csv", "Patient Name,DOB (MM/DD/YYYY)\nRavi Teja,1979-20-04\n", ["1979-20-04"]),  # unseen YYYY-DD-MM
    ("x1.xml", '<?xml version="1.0"?><doc><patient><name>Shubhangi Tiwari</name></patient>'
               '<id extension="UHID-778899" root="2.16.356"/><text>Known case of asthma</text></doc>',
     ["shubhangi", "778899"]),
]


@pytest.fixture(scope="module")
def evasion_run(tmp_path_factory):
    base = tmp_path_factory.mktemp("evasion")
    inp, out, tmp = base / "in", base / "out", base / "tmp"
    inp.mkdir(); tmp.mkdir()
    for fname, content, _ in CASES:
        (inp / fname).write_text(content, encoding="utf-8")
    blob = base64.b64encode(base64.b64encode(b"Visitor: Hemlata Pargaien 9811177722").decode().encode()).decode()
    (inp / "b1.json").write_text(json.dumps({"attachment": blob}))
    outside = base / "outside.txt"
    outside.write_text("Name: Pooja Mehra 9876511111")
    try:
        (inp / "link.txt").symlink_to(outside)
    except OSError:
        pass
    p = _run(inp, out, tmp)
    return p, out, tmp


def test_run_succeeds_and_contract(evasion_run):
    p, out, _ = evasion_run
    assert p.returncode == 0, p.stderr[-500:]
    validator = REPO / "validate_output.py"
    if not validator.is_file():  # submission repo ships without the kit; run from the kit to check the contract
        validator = Path(os.environ.get("KIT_DIR", REPO.parent / "Jalagaara")) / "validate_output.py"
    if not validator.is_file():
        pytest.skip("kit validate_output.py not available")
    v = subprocess.run([sys.executable, str(validator), "--input", str(out.parent / "in"),
                        "--output", str(out)], capture_output=True, text=True)
    assert "CONTRACT OK" in v.stdout


@pytest.mark.parametrize("fname,_content,needles", CASES)
def test_disguised_pii_never_survives(evasion_run, fname, _content, needles):
    _, out, _ = evasion_run
    hay = _all_text(out)
    for n in needles:
        assert n.casefold() not in hay, f"{fname}: survived"


def test_double_base64_payload_scrubbed(evasion_run):
    _, out, _ = evasion_run
    raw = (out / "sanitized").rglob("*.json")
    text = ""
    for f in raw:
        t = f.read_text(encoding="utf-8")
        text += t
        for tok in t.replace('"', " ").split():
            try:
                d = base64.b64decode(tok + "=" * (-len(tok) % 4)).decode("utf-8", "ignore")
                text += d + base64.b64decode(d + "=" * (-len(d) % 4)).decode("utf-8", "ignore")
            except Exception:
                pass
    assert "Pargaien" not in text and "9811177722" not in text


def test_symlink_outside_input_never_read_into_output(evasion_run):
    _, out, _ = evasion_run
    assert "pooja" not in _all_text(out) and "9876511111" not in _all_text(out)


def test_xml_sanitised_and_well_formed(evasion_run):
    _, out, _ = evasion_run
    xmls = list((out / "sanitized").rglob("*.xml"))
    assert xmls, "XML should be sanitised, not quarantined"
    import xml.parsers.expat as expat
    expat.ParserCreate().Parse(xmls[0].read_bytes(), True)
    assert "asthma" in xmls[0].read_text()


def test_no_side_channel_writes(evasion_run):
    p, _, tmp = evasion_run
    assert not any(tmp.rglob("*"))
    assert "vuln-regression-secret" not in p.stdout + p.stderr


def test_missing_secret_fails_closed(tmp_path):
    inp = tmp_path / "in"; inp.mkdir(); (inp / "a.txt").write_text("Name: Ravi Teja")
    p = _run(inp, tmp_path / "out", tmp_path, secret=None)
    assert p.returncode != 0 and not (tmp_path / "out" / "sanitized").exists()
    assert "Traceback" not in p.stderr
