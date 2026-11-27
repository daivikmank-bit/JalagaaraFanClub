"""B5: Build safe_words.txt (S \\ N) from clinical/places lexicons minus names & generic_name_tokens."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
LEX_DIR = ROOT / "config" / "lexicons"
POLICY_PATH = ROOT / "config" / "policy.yaml"


def _load_words(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        line.strip().lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def main() -> None:
    clinical_files = sorted((LEX_DIR / "clinical").glob("*.txt"))
    place_files = sorted((LEX_DIR / "places").glob("*.txt")) if (LEX_DIR / "places").exists() else []
    name_files = sorted((LEX_DIR / "names").glob("*.txt"))

    safe_s: set[str] = set()
    for path in clinical_files + place_files:
        safe_s |= _load_words(path)

    names_n: set[str] = set()
    for path in name_files:
        names_n |= _load_words(path)

    if POLICY_PATH.exists():
        pol = yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8")) or {}
        generic_tokens = {str(t).strip().lower() for t in pol.get("linkage", {}).get("generic_name_tokens", []) if str(t).strip()}
        names_n |= generic_tokens

    safe_minus_n = sorted(safe_s - names_n)

    out_dir = LEX_DIR / "safe"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_path = out_dir / "safe_words.txt"
    safe_path.write_text("\n".join(safe_minus_n) + "\n", encoding="utf-8")

    entries = [
        {
            "path": "clinical/drugs.txt",
            "source_name": "MoHFW NLEM / WHO Essential Medicines (seed v1)",
            "source_version": "NLEM-2022-seed",
            "url": "https://mohfw.gov.in/",
            "licence": "Public Domain / CC0",
            "retrieved_date": "2026-11-26",
            "build_script": "manual_seed",
            "record_count": len(_load_words(LEX_DIR / "clinical" / "drugs.txt")),
        },
        {
            "path": "clinical/symptoms_conditions.txt",
            "source_name": "WHO ICD-10 / Clinical Terminology (seed v1)",
            "source_version": "ICD10-2019-seed",
            "url": "https://icd.who.int/",
            "licence": "CC BY-ND 3.0 IGO / Public Clinical Terms",
            "retrieved_date": "2026-11-26",
            "build_script": "manual_seed",
            "record_count": len(_load_words(LEX_DIR / "clinical" / "symptoms_conditions.txt")),
        },
        {
            "path": "names/generic_names_seed.txt",
            "source_name": "Synthetic Indian Name Seed (Faker-style v1)",
            "source_version": "seed-v1",
            "url": "synthetic",
            "licence": "MIT",
            "retrieved_date": "2026-11-26",
            "build_script": "manual_seed",
            "record_count": len(_load_words(LEX_DIR / "names" / "generic_names_seed.txt")),
        },
        {
            "path": "safe/safe_words.txt",
            "source_name": "Derived Safe Set (S \\ N) with policy generic_name_tokens subtracted",
            "source_version": "v1",
            "url": "local:build_clinical_lexicons",
            "licence": "Derived",
            "retrieved_date": "2026-11-26",
            "build_script": "tools/build_clinical_lexicons.py",
            "record_count": len(safe_minus_n),
        },
    ]

    manifest = {"schema_version": "1.0", "entries": entries}
    (LEX_DIR / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Built safe_words.txt: S={len(safe_s)}, N={len(names_n)}, S\\N={len(safe_minus_n)}")


if __name__ == "__main__":
    main()
