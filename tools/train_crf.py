"""B15 (P1): Synthetic CRF trainer (<= 5 MB model, synthetic data only)."""

from __future__ import annotations

from pathlib import Path

from kavachmed.contracts import Context
from kavachmed.detect.crf import HAS_CRFSUITE, token_features

MAX_MODEL_BYTES = 5 * 1024 * 1024


def train_synthetic_crf(out_path: Path, ctx: Context) -> bool:
    """Train a compact CRF on synthetic templates if python-crfsuite is installed."""
    if not HAS_CRFSUITE:
        print("B15 CRF cut: python-crfsuite not installed; skipping model training.")
        return False

    import pycrfsuite  # type: ignore[import-not-found]

    samples = [
        (["Pt", "Name", "Kavita", "Joshi", "fever"], ["O", "O", "B-PER", "I-PER", "O"]),
        (["Rohan", "ka", "BP", "normal", "hai"], ["B-PER", "O", "O", "O", "O"]),
        (["नाम", "सुरेश", "वर्मा", "बुखार"], ["O", "B-PER", "I-PER", "O"]),
    ]
    trainer = pycrfsuite.Trainer(verbose=False)
    for toks, tags in samples * 10:
        feats = [token_features(toks, i, ctx) for i in range(len(toks))]
        trainer.append(feats, tags)
    trainer.set_params({"c1": 1.0, "c2": 1e-3, "max_iterations": 50})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.train(str(out_path))
    if out_path.stat().st_size > MAX_MODEL_BYTES:
        out_path.unlink()
        raise ValueError("Trained CRF model exceeds 5 MB limit")
    return True


if __name__ == "__main__":
    train_synthetic_crf(Path(__file__).resolve().parents[1] / "models" / "ner.crfsuite", Context())
