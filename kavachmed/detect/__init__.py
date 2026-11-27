"""PII detection entry point and lexicon Context builder (Owner B)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from kavachmed.contracts import Context, Record, Span
from kavachmed.detect.arbiter import detect as _arbiter_detect


def _load_word_set(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    words: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        w = line.strip()
        if w and not w.startswith("#"):
            words.add(w.lower())
    return words


def build_context(policy: dict[str, Any], keys: Any = None, config_dir: Path | None = None) -> Context:
    """Load licensed lexicons from config/lexicons/ and return a populated Context."""
    if config_dir is None:
        config_dir = Path(__file__).resolve().parent.parent.parent / "config"
    lex_root = config_dir / "lexicons"
    lexicons: dict[str, set[str]] = {
        "drugs": _load_word_set(lex_root / "clinical" / "drugs.txt"),
        "symptoms": _load_word_set(lex_root / "clinical" / "symptoms_conditions.txt"),
        "names": _load_word_set(lex_root / "names" / "generic_names_seed.txt"),
        "safe_words": _load_word_set(lex_root / "safe" / "safe_words.txt"),
    }
    lexicons["places"] = _load_places(lex_root / "places")
    # Safe set S\N (FR-D11): clinical vocabulary + drug/condition lexicon tokens, minus every
    # name-gazetteer token (Latin and Devanagari) so common-word names are still caught.
    names_all = (set(lexicons["names"]) | _load_gz_words(lex_root / "names" / "names_seed.txt.gz")
                 | _load_word_set(lex_root / "names" / "indianpersona_names.txt"))
    clinical = _load_vocab(lex_root / "clinical" / "clinical_vocab.txt")
    for key in ("drugs", "symptoms"):
        for entry in lexicons[key]:
            clinical.update(w for w in entry.replace("-", " ").split() if len(w) > 1)
    lexicons["safe_words"] = (set(lexicons["safe_words"]) | clinical) - names_all
    lexicons["names"] = names_all
    return Context(policy=policy, lexicons=lexicons, keys=keys)


def _load_vocab(path: Path) -> set[str]:
    """Whitespace-separated vocabulary file; '#' lines are comments."""
    if not path.is_file():
        return set()
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("#"):
            continue
        out.update(w.lower() for w in line.split())
    return out


def _load_gz_words(path: Path) -> set[str]:
    import gzip
    if not path.is_file():
        return set()
    return {w.strip().lower() for w in gzip.decompress(path.read_bytes()).decode("utf-8").splitlines() if w.strip() and not w.startswith("#")}


def _load_places(places_dir: Path) -> set[str]:
    """LGD state/district/sub-district names (Latin + Devanagari) from Owner D's lexicon.

    Without it default-deny redacts every district, which privacy.plan keeps by policy.
    """
    out: set[str] = set()
    # States and districts only: 7k sub-district names include many person-like words, and a
    # place hit clears a token from default-deny. Sub-districts are used by D's location code.
    for fname in ("lgd_states.tsv", "lgd_districts.tsv"):
        f = places_dir / fname
        if not f.is_file():
            continue
        for line in f.read_text(encoding="utf-8").splitlines()[1:]:
            for col in line.split("\t")[1:]:
                col = col.strip()
                if col and not col.isdigit():
                    out.add(col.lower())
                    out.update(w.lower() for w in col.split() if len(w) > 2)
    return out


def _ensure_context(ctx: Context | dict[str, Any]) -> Context:
    if isinstance(ctx, Context):
        return ctx
    return Context(policy=ctx.get("policy", {}), keys=ctx.get("keys"))


def detect(rec: Record, ctx: Context | dict[str, Any], shield_spans: list[Span] | None = None) -> list[Span]:
    """Run Owner B's arbitrated detector stack."""
    return _arbiter_detect(rec, _ensure_context(ctx), shield_spans=shield_spans)
