"""Related-question retrieval (BM25, embedding cosine), explanations, and leakage-safe evaluation.

Evaluation protocol (docs/research_design.md, section 5):

* Positives: duplicate-closure and body-link pairs (undirected) with both endpoints in the corpus.
* Queries: test-split, research-window questions with >=1 positive created *before* the query.
* Candidates: every question created before the query (realistic "find earlier related
  questions"); all methods rank the identical candidate set.
* BM25 vocabulary/IDF are fit only on questions created before the test boundary.
* Link labels are never used as features by these baselines.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.data.raw_store import atomic_write_text
from ml.evaluation.ranking import query_metrics, ranks_of_relevant, summarize
from ml.retrieval.bm25 import BM25, question_text, tokenize

EXPLAIN_ENTITY_TYPES = ("device", "os", "error", "tech", "category")


@dataclass
class EvalSet:
    queries: pd.DataFrame  # query_id, query_pos, relevant_pos(list), relevant_types(list)
    order: np.ndarray  # question_ids sorted by created_at (row i of the index)


class RetrievalIndex:
    def __init__(
        self,
        questions: pd.DataFrame,
        entities: pd.DataFrame | None = None,
        embeddings: np.ndarray | None = None,
        embedding_ids: np.ndarray | None = None,
        bm25_fit_before: pd.Timestamp | None = None,
    ):
        self.q = questions.sort_values(["created_at", "question_id"]).reset_index(drop=True)
        self.pos = pd.Series(np.arange(len(self.q)), index=self.q["question_id"].to_numpy())
        self.tokens = [tokenize(question_text(t, b)) for t, b in zip(self.q["title"], self.q["body_text"], strict=True)]
        fit_mask = (self.q["created_at"] < bm25_fit_before).to_numpy() if bm25_fit_before is not None else None
        fit_docs = (
            [t for t, keep in zip(self.tokens, fit_mask, strict=True) if keep] if fit_mask is not None else self.tokens
        )
        self.bm25 = BM25().fit(fit_docs).index(self.tokens)
        self.bm25_fit_before = bm25_fit_before
        self.emb = None
        if embeddings is not None:
            idx = pd.Series(np.arange(len(embedding_ids)), index=embedding_ids)
            self.emb = embeddings[idx.loc[self.q["question_id"]].to_numpy()]
        self.entities = (
            entities[entities["entity_type"].isin(EXPLAIN_ENTITY_TYPES)]
            .groupby("question_id")[["entity_type", "value"]]
            .apply(lambda g: set(zip(g["entity_type"], g["value"], strict=True)))
            if entities is not None and len(entities)
            else pd.Series(dtype=object)
        )

    def scores(self, method: str, i: int) -> np.ndarray:
        if method == "bm25":
            return self.bm25.scores(self.tokens[i])
        if method == "embedding":
            if self.emb is None:
                raise ValueError("embeddings not loaded")
            return self.emb @ self.emb[i]
        raise ValueError(f"unknown method {method}")

    def explain(self, i: int, j: int) -> dict[str, Any]:
        a, b = self.q.iloc[i], self.q.iloc[j]
        ea = self.entities.get(a["question_id"], set())
        eb = self.entities.get(b["question_id"], set())
        shared = sorted(ea & eb)
        return {
            "shared_tags": sorted(set(a["tags"]) & set(b["tags"])),
            "shared_entities": {
                t: sorted(v for tt, v in shared if tt == t)
                for t in EXPLAIN_ENTITY_TYPES
                if any(tt == t for tt, _ in shared)
            },
            "bm25_top_terms": [
                {"term": t, "weight": round(w, 3)} for t, w in self.bm25.term_contributions(self.tokens[i], j)
            ],
            "embedding_cosine": float(self.emb[i] @ self.emb[j]) if self.emb is not None else None,
        }

    def related(self, question_id: int, method: str, k: int = 10, only_earlier: bool = False) -> list[dict[str, Any]]:
        i = int(self.pos[question_id])
        s = self.scores(method, i).astype(float)
        s[i] = -np.inf
        if only_earlier:
            s[i:] = -np.inf
        top = np.argsort(-s)[:k]
        return [
            {
                "question_id": int(self.q.at[j, "question_id"]),
                "title": self.q.at[j, "title"],
                "link": self.q.at[j, "link"],
                "score": float(s[j]),
                "explanation": self.explain(i, j),
            }
            for j in top
            if np.isfinite(s[j])
        ]


def build_eval_set(index: RetrievalIndex, links: pd.DataFrame, splits: pd.DataFrame, split: str = "test") -> EvalSet:
    q = index.q
    in_corpus = links[links["source_id"].isin(index.pos.index) & links["target_id"].isin(index.pos.index)]
    pairs = pd.concat(
        [
            in_corpus[["source_id", "target_id", "link_type"]].rename(columns={"source_id": "a", "target_id": "b"}),
            in_corpus[["target_id", "source_id", "link_type"]].rename(columns={"target_id": "a", "source_id": "b"}),
        ]
    )
    split_of = splits.set_index("question_id")["split"]
    window = splits.set_index("question_id")["in_research_window"]
    pairs = pairs[pairs["a"].map(split_of).eq(split) & pairs["a"].map(window).fillna(False).astype(bool)]
    pairs = pairs.assign(a_pos=index.pos.loc[pairs["a"]].to_numpy(), b_pos=index.pos.loc[pairs["b"]].to_numpy())
    pairs = pairs[pairs["b_pos"] < pairs["a_pos"]]  # counterpart must be created earlier
    grouped = pairs.groupby("a").agg(
        query_pos=("a_pos", "first"),
        relevant_pos=("b_pos", lambda s: sorted(set(s))),
        relevant_types=("link_type", lambda s: sorted(set(s))),
    )
    queries = grouped.reset_index().rename(columns={"a": "query_id"})
    return EvalSet(queries=queries, order=q["question_id"].to_numpy())


def evaluate(
    index: RetrievalIndex, eval_set: EvalSet, methods: tuple[str, ...] = ("bm25", "embedding")
) -> pd.DataFrame:
    rows = []
    for method in methods:
        for r in eval_set.queries.itertuples(index=False):
            s = index.scores(method, r.query_pos)[: r.query_pos]  # candidates: strictly earlier questions
            ranks = ranks_of_relevant(s, r.relevant_pos)
            rows.append(
                {
                    "method": method,
                    "query_id": r.query_id,
                    "n_candidates": r.query_pos,
                    "has_duplicate_label": "duplicate" in r.relevant_types,
                    **query_metrics(ranks),
                }
            )
    return pd.DataFrame(rows)


def run_retrieval_eval(
    index: RetrievalIndex, links: pd.DataFrame, splits: pd.DataFrame, out_dir: Path, meta: dict
) -> dict:
    es = build_eval_set(index, links, splits)
    methods = ("bm25", "embedding") if index.emb is not None else ("bm25",)
    per_query = evaluate(index, es, methods)
    summary = summarize(per_query)
    dup = summarize(per_query[per_query["has_duplicate_label"]]).assign(stratum="duplicate_label_queries")
    out_dir.mkdir(parents=True, exist_ok=True)
    per_query.to_parquet(out_dir / "per_query.parquet", index=False)
    table = pd.concat([summary.assign(stratum="all_queries"), dup], ignore_index=True)
    table.to_csv(out_dir / "summary.csv", index=False)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "n_queries": int(len(es.queries)),
        "n_queries_with_duplicate_label": int(
            per_query.loc[per_query["method"] == methods[0], "has_duplicate_label"].sum()
        ),
        "methods": list(methods),
        "bm25_fit_before": str(index.bm25_fit_before),
        "candidate_rule": "all questions created before the query",
        **meta,
    }
    atomic_write_text(out_dir / "manifest.json", json.dumps(manifest, indent=2, default=str))
    return {"manifest": manifest, "table": table}
