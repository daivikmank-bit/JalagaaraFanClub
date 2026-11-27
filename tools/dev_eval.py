"""Repeatable dev evaluation against Jalagaara/dev/truth/truth.json (dev-only tool, never imported by runtime)."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from kavachmed.contracts import Context, Field, Record
from kavachmed.detect.arbiter import detect
from kavachmed.shield import protect


def _alnum(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z\u0900-\u097F]+", "", s)


def _digits(s: str) -> str:
    return re.sub(r"\D+", "", s)


def evaluate_dev_texts(kit_dir: Path, ctx: Context | None = None) -> dict[str, object]:
    """Evaluate detector on all UTF-8 plain-text files in dev/truth/truth.json."""
    truth_path = kit_dir / "dev" / "truth" / "truth.json"
    input_dir = kit_dir / "dev" / "input"
    truth = json.loads(truth_path.read_text(encoding="utf-8"))

    if ctx is None:
        root = Path(__file__).resolve().parents[1]
        safe = {l.strip().lower() for l in (root / "config" / "lexicons" / "safe" / "safe_words.txt").read_text(encoding="utf-8").splitlines() if l.strip()}
        names = {l.strip().lower() for l in (root / "config" / "lexicons" / "names" / "generic_names_seed.txt").read_text(encoding="utf-8").splitlines() if l.strip()}
        drugs = {l.strip().lower() for l in (root / "config" / "lexicons" / "clinical" / "drugs.txt").read_text(encoding="utf-8").splitlines() if l.strip()}
        ctx = Context(
            policy={"detect": {"stack": "recall_first", "recall_threshold": 0.35, "precision_threshold": 0.55, "vehicle_plates": True, "doctor_reg_remove_digits": True, "default_deny": {"score": 0.45}}},
            lexicons={"safe_words": safe, "names": names, "drugs": drugs, "places": set()},
        )

    pii_stats: dict[str, Counter[str]] = defaultdict(Counter)
    mrn_digits_total = 0
    mrn_digits_covered = 0
    clin_overlap: dict[str, Counter[str]] = defaultdict(Counter)
    files_evaluated = 0

    for rel_path, entry in sorted(truth.get("files", {}).items()):
        if entry.get("expect") != "sanitize" or not rel_path.endswith(".txt"):
            continue
        fp = input_dir / rel_path
        if not fp.exists():
            continue
        text = fp.read_text(encoding="utf-8")
        files_evaluated += 1
        rec = Record(key=(rel_path, "doc"), fields=[Field(0, "doc", text)])
        sh = protect(rec, ctx)
        spans = detect(rec, ctx, sh)

        covered = [False] * len(text)
        for sp in spans:
            for i in range(max(0, sp.start), min(len(text), sp.end)):
                covered[i] = True

        for unit in entry.get("units", {}).values():
            for ent in unit.get("entities", []):
                if ent.get("channel") != "content":
                    continue
                surf = ent.get("surface", "")
                typ = ent.get("type", "UNKNOWN")
                if not surf:
                    continue
                pos = 0
                while True:
                    idx = text.find(surf, pos)
                    if idx == -1:
                        break
                    end = idx + len(surf)
                    alnum_idxs = [i for i in range(idx, end) if text[i].isalnum()]
                    if alnum_idxs:
                        hit_cnt = sum(1 for i in alnum_idxs if covered[i])
                        if hit_cnt == len(alnum_idxs):
                            pii_stats[typ]["full"] += 1
                        elif hit_cnt > 0:
                            pii_stats[typ]["partial"] += 1
                        else:
                            pii_stats[typ]["missed"] += 1
                        if typ == "MRN":
                            dig_idxs = [i for i in range(idx, end) if text[i].isdigit()]
                            mrn_digits_total += len(dig_idxs)
                            mrn_digits_covered += sum(1 for i in dig_idxs if covered[i])
                    pos = end

            for med in unit.get("preserve", []):
                surf = med.get("text", "")
                kind = med.get("kind", "unknown")
                if not surf:
                    continue
                pos = 0
                while True:
                    idx = text.find(surf, pos)
                    if idx == -1:
                        break
                    end = idx + len(surf)
                    clin_overlap[kind]["total"] += 1
                    if any(covered[i] for i in range(idx, end)):
                        clin_overlap[kind]["overlapped"] += 1
                    else:
                        clin_overlap[kind]["untouched"] += 1
                    pos = end

    return {
        "files_evaluated": files_evaluated,
        "pii": {k: dict(v) for k, v in sorted(pii_stats.items())},
        "mrn_digit_coverage": mrn_digits_covered / max(1, mrn_digits_total),
        "mrn_digits": {"covered": mrn_digits_covered, "total": mrn_digits_total},
        "clinical_overlap": {k: dict(v) for k, v in sorted(clin_overlap.items())},
    }


if __name__ == "__main__":
    kit = Path(__file__).resolve().parents[2] / "Jalagaara"
    print(json.dumps(evaluate_dev_texts(kit), indent=2))
