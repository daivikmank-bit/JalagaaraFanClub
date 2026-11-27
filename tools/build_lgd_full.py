"""Rebuild the full LGD geography lexicons (FR-X1) from the India Data Portal LGD-codes dataset.

Source: India Data Portal dataset ``lgd-codes`` (resources "District LGD Codes" and
"Sub District LGD Codes"), extracted from lgdirectory.gov.in (data retrieval 2023-03-01),
licence Open Data Commons Attribution (ODC-BY). Fetched via the public CKAN datastore API at
build time only; the runtime never touches the network.

Existing Devanagari names (hand-entered for a subset) are preserved by matching
(state, district) names. Output codes are the real LGD district/sub-district codes.
Never derived from dev/ data.

Usage (dev machine, network allowed):
    python tools/build_lgd_full.py            # fetch + rebuild + update MANIFEST.json
    python tools/build_lgd_full.py --from-dir DIR   # rebuild from saved datastore JSON
"""
from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
import urllib.request
from pathlib import Path

_API = "https://ckandev.indiadataportal.com/api/3/action/datastore_search?resource_id={rid}&limit=32000"
DISTRICT_RID = "19df978a-675e-4ff5-8015-d0e1de447319"
SUBDISTRICT_RID = "42cd589e-2dc4-4f9c-a245-f201cee357e7"
ROOT = Path(__file__).resolve().parents[1]
PLACES = ROOT / "config" / "lexicons" / "places"
SOURCE = "India Data Portal 'lgd-codes' dataset (extracted from Local Government Directory, lgdirectory.gov.in; retrieved by IDP 2023-03-01)"
URL = "https://ckandev.indiadataportal.com/dataset/lgd-codes"
LICENCE = "Open Data Commons Attribution License (ODC-BY); underlying data GODL-India"


def _norm(s: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", s).casefold().split())


def _fetch(rid: str, from_dir: Path | None) -> list[dict]:
    if from_dir is not None:
        return json.loads((from_dir / f"{rid}.json").read_text(encoding="utf-8"))["result"]["records"]
    req = urllib.request.Request(_API.format(rid=rid), headers={"User-Agent": "kavachmed-lexicon-builder"})
    with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 - fixed https URL, dev-only tool
        return json.loads(r.read().decode("utf-8"))["result"]["records"]


def _read_tsv(path: Path) -> list[list[str]]:
    if not path.is_file():
        return []
    return [ln.split("\t") for ln in path.read_text(encoding="utf-8").splitlines()[1:] if ln.strip()]


def build(from_dir: Path | None = None) -> dict[str, int]:
    old_states = _read_tsv(PLACES / "lgd_states.tsv")
    old_dist = _read_tsv(PLACES / "lgd_districts.tsv")
    old_sub = _read_tsv(PLACES / "lgd_subdistricts.tsv")
    state_hi = {_norm(r[1]): r[2] for r in old_states if len(r) >= 3}
    old_state_name = {r[0]: _norm(r[1]) for r in old_states if len(r) >= 2}
    dist_hi = {(old_state_name.get(r[1], ""), _norm(r[2])): r[3] for r in old_dist if len(r) >= 4 and r[3]}
    sub_hi = {_norm(r[2]): r[3] for r in old_sub if len(r) >= 4 and r[3]}

    dists = _fetch(DISTRICT_RID, from_dir)
    subs = _fetch(SUBDISTRICT_RID, from_dir)

    states: dict[str, str] = {}
    for r in dists:
        states.setdefault(str(r["state_code"]).zfill(2), str(r["state_name"]).strip())
    for code, name in [(r[0], r[1]) for r in old_states if len(r) >= 2]:
        states.setdefault(code, name)  # keep any state present only in the old list

    st_lines = ["state_code\tstate_name\tstate_name_hi"]
    for code in sorted(states):
        st_lines.append(f"{code}\t{states[code]}\t{state_hi.get(_norm(states[code]), '')}")

    d_lines = ["district_code\tstate_code\tdistrict_name\tdistrict_name_hi"]
    for r in sorted(dists, key=lambda r: (str(r["state_code"]).zfill(2), str(r["district_code"]).zfill(3))):
        sc, dc, name = str(r["state_code"]).zfill(2), str(r["district_code"]).zfill(3), str(r["district_name"]).strip()
        hi = dist_hi.get((_norm(states.get(sc, "")), _norm(name)), "")
        d_lines.append(f"{dc}\t{sc}\t{name}\t{hi}")

    s_lines = ["subdistrict_code\tdistrict_code\tsubdistrict_name\tsubdistrict_name_hi"]
    for r in sorted(subs, key=lambda r: (str(r["district_code"]).zfill(3), str(r["sub_district_code"]).zfill(4))):
        dc, sdc, name = str(r["district_code"]).zfill(3), str(r["sub_district_code"]).zfill(4), str(r["sub_district_name"]).strip()
        s_lines.append(f"{sdc}\t{dc}\t{name}\t{sub_hi.get(_norm(name), '')}")

    out = {"lgd_states.tsv": st_lines, "lgd_districts.tsv": d_lines, "lgd_subdistricts.tsv": s_lines}
    for fname, lines in out.items():
        (PLACES / fname).write_text("\n".join(lines) + "\n", encoding="utf-8")

    man_p = ROOT / "config" / "lexicons" / "MANIFEST.json"
    man = json.loads(man_p.read_text(encoding="utf-8"))
    entries = [e for e in man["entries"] if e.get("path") not in {f"places/{f}" for f in out}]
    for fname, lines in out.items():
        p = PLACES / fname
        entries.append({
            "path": f"places/{fname}", "source_name": SOURCE, "source_version": "2023-03-01",
            "url": URL, "licence": LICENCE, "retrieved_date": "2026-11-27",
            "build_script": "tools/build_lgd_full.py", "record_count": len(lines) - 1,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "notes": "Devanagari names kept from the earlier hand-entered subset where names match",
        })
    man["entries"] = sorted(entries, key=lambda e: e["path"])
    man_p.write_text(json.dumps(man, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {k: len(v) - 1 for k, v in out.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-dir", type=Path, default=None)
    a = ap.parse_args()
    print(build(a.from_dir))
