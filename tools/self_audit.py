#!/usr/bin/env python3
"""Static self-audit for KavachMed submission code.

Scans ONLY submission artifacts (kavachmed/, main.py, config/, tests/,
requirements.txt, raksha.toml). Deliberately skips dev/, starter/, plans/,
docs/, dataset-bhargav/, tools/chaos_mutator.py, and tools/oracle/ because
this branch lives inside the Jalagaara workspace repo.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

BANNED_RUNTIME_IMPORTS = frozenset({
    "cryptography", "openai", "anthropic", "google", "transformers",
    "requests", "httpx", "urllib", "socket", "orjson",
})

RESIDUAL_B_FORBIDDEN = frozenset({
    "detect", "shield", "evidence", "linkage", "residual_a", "typing", "propagate",
})

BANNED_REQ_RE = re.compile(
    r"^\s*(openai|anthropic|google-generativeai|google-genai|ollama|llama[-_]cpp|llama-index|"
    r"langchain\S*|vllm|mistralai|cohere|groq|together|litellm|gpt4all|ctransformers|cryptography)\b",
    re.I | re.M,
)


def _iter_py_files(root: Path, dirs: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    if (root / "main.py").is_file():
        files.append(root / "main.py")
    for d in dirs:
        base = root / d
        if base.is_dir():
            files.extend(sorted(base.rglob("*.py")))
    return files


def _check_imports(py_files: list[Path], root: Path) -> list[str]:
    errors: list[str] = []
    for path in py_files:
        rel = path.relative_to(root).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        for node in ast.walk(tree):
            mods: list[str] = []
            if isinstance(node, ast.Import):
                mods = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            for mod in mods:
                top = mod.split(".")[0]
                if top in BANNED_RUNTIME_IMPORTS:
                    errors.append(f"banned import '{mod}' in {rel}")
                if rel == "kavachmed/residual_b.py":
                    sub = mod.split(".")[1] if mod.startswith("kavachmed.") and len(mod.split(".")) > 1 else top
                    if sub in RESIDUAL_B_FORBIDDEN:
                        errors.append(f"Layer B independence violation: '{mod}' imported in {rel}")
    return errors


def _check_oracle_refs(py_files: list[Path], root: Path) -> list[str]:
    errors: list[str] = []
    for path in py_files:
        rel = path.relative_to(root).as_posix()
        txt = path.read_text(encoding="utf-8", errors="ignore")
        for needle in ("KM_ORACLE", "truth.json"):
            if needle in txt:
                errors.append(f"forbidden reference '{needle}' in {rel}")
    return errors


def _check_dev_strings(root: Path, kit_dir: Path) -> list[str]:
    truth_path = kit_dir / "dev" / "truth" / "truth.json"
    if not truth_path.is_file():
        return []
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    canaries: set[str] = set()
    for c in truth.get("canaries", []):
        if isinstance(c, str) and len(c) >= 6:
            canaries.add(c)
        elif isinstance(c, dict):
            for v in c.values():
                if isinstance(v, str) and len(v) >= 6:
                    canaries.add(v)

    if not canaries:
        return []

    errors: list[str] = []
    scan_targets = _iter_py_files(root, ("kavachmed", "tests"))
    for path in scan_targets:
        rel = path.relative_to(root).as_posix()
        txt = path.read_text(encoding="utf-8", errors="ignore")
        hits = sum(1 for c in canaries if c in txt)
        if hits:
            errors.append(f"dev_canary_leak: {hits} match(es) in {rel}")
    return errors


def _check_lexicon_manifest(root: Path) -> list[str]:
    lex_dir = root / "config" / "lexicons"
    if not lex_dir.is_dir():
        return []
    manifest_path = lex_dir / "MANIFEST.json"
    lex_files = [
        p.relative_to(lex_dir).as_posix()
        for p in sorted(lex_dir.rglob("*"))
        if p.is_file() and p.name != "MANIFEST.json"
    ]
    if not lex_files:
        return []
    if not manifest_path.is_file():
        return ["config/lexicons/MANIFEST.json missing while lexicon files exist"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(manifest, dict) and isinstance(manifest.get("entries"), list):
        entries = {
            e["path"]: e
            for e in manifest["entries"]
            if isinstance(e, dict) and isinstance(e.get("path"), str)
        }
    else:
        entries = manifest.get("files", manifest) if isinstance(manifest, dict) else {}
    errors: list[str] = []
    for lf in lex_files:
        meta = entries.get(lf) if isinstance(entries, dict) else None
        if not isinstance(meta, dict) or not meta.get("licence"):
            errors.append(f"lexicon '{lf}' missing MANIFEST.json entry or licence")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description="KavachMed static self-audit")
    ap.add_argument("--root", default=".", help="Repo root")
    ap.add_argument("--kit", default=".", help="Kit directory containing dev/")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    kit_dir = Path(args.kit).resolve()

    errors: list[str] = []
    req = root / "requirements.txt"
    if req.is_file():
        bad = BANNED_REQ_RE.findall(req.read_text(encoding="utf-8", errors="ignore"))
        if bad:
            errors.append(f"banned packages in requirements.txt: {', '.join(sorted(set(bad)))}")

    py_runtime = _iter_py_files(root, ("kavachmed",))
    errors.extend(_check_imports(py_runtime, root))
    errors.extend(_check_oracle_refs(py_runtime, root))
    errors.extend(_check_dev_strings(root, kit_dir))
    errors.extend(_check_lexicon_manifest(root))

    if errors:
        for e in errors:
            print(f"AUDIT FAIL: {e}", file=sys.stderr)
        return 1
    print("SELF AUDIT OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

