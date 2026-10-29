"""Sentence-transformer embeddings computed locally and cached per snapshot.

The model is frozen and pretrained (never fine-tuned on this data), so embedding test questions
does not leak label information. No text is sent to any external service.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_BODY_CHARS = 1500


def embedding_text(title: str, body: str) -> str:
    return f"{title}. {body[:MAX_BODY_CHARS]}"


def _device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def encode(texts: list[str], model_name: str = DEFAULT_MODEL, batch_size: int = 128) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=_device())
    emb = model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    return np.asarray(emb, dtype=np.float32)


def cache_paths(artifacts_dir: Path, snapshot_id: str, model_name: str) -> tuple[Path, Path]:
    slug = model_name.split("/")[-1]
    base = artifacts_dir / "embeddings" / snapshot_id
    return base / f"{slug}.npy", base / f"{slug}.ids.parquet"


def load_or_compute(
    questions: pd.DataFrame, artifacts_dir: Path, snapshot_id: str, model_name: str = DEFAULT_MODEL
) -> tuple[np.ndarray, np.ndarray]:
    """Return (embeddings [n x d] L2-normalised, question_ids) aligned row-by-row."""
    emb_path, ids_path = cache_paths(artifacts_dir, snapshot_id, model_name)
    if emb_path.exists() and ids_path.exists():
        ids = pd.read_parquet(ids_path)["question_id"].to_numpy()
        return np.load(emb_path), ids
    texts = [embedding_text(t, b) for t, b in zip(questions["title"], questions["body_text"], strict=True)]
    log.info("encoding %d questions with %s", len(texts), model_name)
    emb = encode(texts, model_name)
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, emb)
    pd.DataFrame({"question_id": questions["question_id"].to_numpy()}).to_parquet(ids_path, index=False)
    (emb_path.parent / "meta.json").write_text(
        json.dumps(
            {"model": model_name, "n": int(len(emb)), "dim": int(emb.shape[1]), "max_body_chars": MAX_BODY_CHARS}
        )
    )
    return emb, questions["question_id"].to_numpy()
