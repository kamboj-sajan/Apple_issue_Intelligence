"""Semantic clusters of questions, used as issue groups for anomaly detection.

K-means is fit on *training-split* research-window embeddings only, then every question is
assigned to its nearest centroid, so later questions never influence cluster definitions.
Each cluster is described by its most distinctive title terms (class-based TF-IDF), which keeps
cluster-level signals interpretable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer

from ml.data.raw_store import atomic_write_text


def fit_clusters(emb: np.ndarray, fit_mask: np.ndarray, k: int = 40, seed: int = 0) -> tuple[np.ndarray, KMeans]:
    km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(emb[fit_mask])
    return km.predict(emb), km


def describe_clusters(titles: pd.Series, labels: np.ndarray, top: int = 8) -> dict[int, list[str]]:
    """Class-based TF-IDF: term frequency within cluster x log(1 + avg terms per cluster / term freq overall)."""
    vec = CountVectorizer(stop_words="english", min_df=3, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9.+-]{2,}\b")
    X = vec.fit_transform(titles.fillna(""))
    vocab = np.array(vec.get_feature_names_out())
    ks = np.unique(labels)
    tf = np.vstack([np.asarray(X[labels == c].sum(axis=0)).ravel() for c in ks]).astype(float)
    tf_norm = tf / np.maximum(tf.sum(axis=1, keepdims=True), 1)
    idf = np.log(1 + tf.sum(axis=1).mean() / np.maximum(tf.sum(axis=0), 1))
    score = tf_norm * idf
    return {int(c): vocab[np.argsort(-score[i])[:top]].tolist() for i, c in enumerate(ks)}


def run_clusters(processed_dir: Path, emb: np.ndarray, emb_ids: np.ndarray, k: int = 40, seed: int = 0) -> dict:
    q = pd.read_parquet(processed_dir / "questions.parquet", columns=["question_id", "title"])
    splits = pd.read_parquet(processed_dir / "splits.parquet").set_index("question_id")
    order = pd.Series(np.arange(len(emb_ids)), index=emb_ids)
    q = q[q["question_id"].isin(order.index)].reset_index(drop=True)
    E = emb[order.loc[q["question_id"]].to_numpy()]
    fit_mask = (
        q["question_id"].map(splits["split"]).eq("train")
        & q["question_id"].map(splits["in_research_window"]).astype(bool)
    ).to_numpy()
    labels, km = fit_clusters(E, fit_mask, k, seed)
    terms = describe_clusters(q["title"], labels)
    pd.DataFrame({"question_id": q["question_id"], "cluster": labels}).to_parquet(
        processed_dir / "clusters.parquet", index=False
    )
    meta = {
        "k": k,
        "seed": seed,
        "fit_on": "train split, research window",
        "n_fit": int(fit_mask.sum()),
        "inertia": float(km.inertia_),
        "sizes": pd.Series(labels).value_counts().sort_index().astype(int).to_dict(),
        "top_terms": terms,
    }
    np.save(processed_dir / "cluster_centroids.npy", km.cluster_centers_.astype(np.float32))
    atomic_write_text(processed_dir / "clusters_meta.json", json.dumps(meta, indent=2))
    return meta
