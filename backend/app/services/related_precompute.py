"""Precompute top-k *earlier* related questions per question for BM25, embedding and GraphSAGE.

Serving differs from evaluation in two documented ways:
* BM25 is fit on the whole snapshot (there is no evaluation cutoff when serving).
* GraphSAGE re-ranks the 100 most similar earlier questions by embedding, using the graph as of the
  snapshot. Scoring every earlier question with the MLP decoder for all 45k questions is too slow for
  a batch job. The evaluated model ranked all candidates; see docs/link_prediction.md.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ml.retrieval.bm25 import BM25, question_text, tokenize

log = logging.getLogger(__name__)

GRAPHSAGE_RERANK_POOL = 100


def _top_earlier(scores: np.ndarray, row_pos: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Mask candidates at or after each row's position, then take the top-k columns per row."""
    cols = np.arange(scores.shape[1])
    scores = np.where(cols[None, :] < row_pos[:, None], scores, -np.inf)
    kk = min(k, scores.shape[1])
    top = np.argpartition(-scores, kk - 1, axis=1)[:, :kk]
    vals = np.take_along_axis(scores, top, axis=1)
    order = np.argsort(-vals, axis=1)
    return np.take_along_axis(top, order, axis=1), np.take_along_axis(vals, order, axis=1)


def _rows(method: str, ids: np.ndarray, top: np.ndarray, vals: np.ndarray, start: int) -> list[tuple]:
    out = []
    for r in range(top.shape[0]):
        rank = 0
        for c, v in zip(top[r], vals[r], strict=True):
            if np.isfinite(v):
                rank += 1
                out.append((int(ids[start + r]), method, rank, int(ids[c]), float(v)))
    return out


def precompute(
    questions: pd.DataFrame,
    emb: np.ndarray | None,
    k: int = 10,
    block: int = 1024,
    graphsage: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (related rows, term idf). ``questions`` must be sorted by created_at; ``emb`` aligned to it."""
    ids = questions["question_id"].to_numpy()
    n = len(ids)
    rows: list[tuple] = []

    tokens = [tokenize(question_text(t, b)) for t, b in zip(questions["title"], questions["body_text"], strict=True)]
    bm = BM25().fit(tokens).index(tokens)
    for s in range(1, n, block):
        e = min(s + block, n)
        Q = np.vstack([bm.query_vector(tokens[i]) for i in range(s, e)])  # [b, V] query term counts
        sc = np.asarray(bm.matrix[:e] @ Q.T).T  # [b, e]: BM25 of each query against docs created up to e
        top, vals = _top_earlier(sc, np.arange(s, e), k)
        rows += _rows("bm25", ids, top, vals, s)
    log.info("bm25 related done")

    if emb is not None:
        for s in range(1, n, block):
            e = min(s + block, n)
            top, vals = _top_earlier(emb[s:e] @ emb[:e].T, np.arange(s, e), k)
            rows += _rows("embedding", ids, top, vals, s)
        log.info("embedding related done")
        if graphsage is not None:
            rows += _graphsage_rerank(ids, emb, graphsage, k, block)
            log.info("graphsage related done")

    related = pd.DataFrame(rows, columns=["question_id", "method", "rank", "related_id", "score"])
    inv = {i: t for t, i in bm.vocab.items()}
    idf = pd.DataFrame({"term": [inv[i] for i in range(len(inv))], "idf": bm.idf})
    return related, idf


def _graphsage_rerank(ids: np.ndarray, emb: np.ndarray, gs: dict, k: int, block: int) -> list[tuple]:
    import torch

    model, z, x, node_of = gs["model"], gs["z"], gs["x"], gs["node_of"]
    nodes = torch.from_numpy(node_of)
    rows = []
    with torch.no_grad():
        for s in range(1, len(ids), block):
            e = min(s + block, len(ids))
            pool, _ = _top_earlier(emb[s:e] @ emb[:e].T, np.arange(s, e), GRAPHSAGE_RERANK_POOL)
            for r in range(e - s):
                i = s + r
                cands = pool[r][pool[r] < i]
                if len(cands) == 0:
                    continue
                qn, cn = nodes[i], nodes[torch.from_numpy(cands)]
                sc = model.score(z[qn], z[cn], x[qn], x[cn]).numpy()
                order = np.argsort(-sc)[:k]
                rows += [
                    (int(ids[i]), "graphsage", rank + 1, int(ids[cands[j]]), float(sc[j]))
                    for rank, j in enumerate(order)
                ]
    return rows


def load_graphsage(
    artifacts: Path, snapshot: str, questions: pd.DataFrame, emb: np.ndarray, emb_ids: np.ndarray, model_name: str
) -> dict | None:
    """Load the seed-0 checkpoint and embed every node with the full snapshot graph (serving)."""
    ckpt = artifacts / "models" / snapshot / "graphsage_seed0.pt"
    if not ckpt.exists():
        return None
    import torch

    from ml.experiments.link_prediction import _entity_embeddings
    from ml.models.graphsage_link import GraphSAGELink, TrainConfig
    from ml.models.link_data import build_link_graph

    gdir = artifacts / "graph" / snapshot
    nodes, edges = pd.read_parquet(gdir / "nodes.parquet"), pd.read_parquet(gdir / "edges.parquet")
    graph = build_link_graph(
        nodes, edges, emb, emb_ids, _entity_embeddings(nodes, artifacts / "embeddings" / snapshot, model_name)
    )
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = TrainConfig(**state["config"])
    model = GraphSAGELink(graph.x.shape[1], cfg)
    model.load_state_dict(state["state_dict"])
    model.eval()
    with torch.no_grad():
        z = model.encode(graph.x, graph.node_type, graph.edge_index_as_of(None))
    node_of = graph.question_node.loc[questions["question_id"]].to_numpy()
    return {"model": model, "z": z, "x": graph.x, "node_of": node_of, "config": state["config"]}
