"""Graph-heuristic baseline: Jaccard similarity of question neighbourhoods in the entity graph.

A question's neighbourhood is its set of tag, device, OS-version, error, technical-entity and
category nodes. These edges exist from the moment the question is posted, so the heuristic uses no
future information. Question-question similarity edges are deliberately excluded: they are the
embedding kNN and would duplicate the embedding baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

ENTITY_EDGE_TYPES = (
    "QUESTION_HAS_TAG",
    "QUESTION_MENTIONS_DEVICE",
    "QUESTION_MENTIONS_VERSION",
    "QUESTION_CONTAINS_ERROR",
    "QUESTION_MENTIONS_TECH",
    "QUESTION_IN_CATEGORY",
)


class JaccardHeuristic:
    def __init__(self, edges: pd.DataFrame, question_order: np.ndarray):
        e = edges[edges["edge_type"].isin(ENTITY_EDGE_TYPES)]
        qpos = pd.Series(np.arange(len(question_order)), index=[f"question:{q}" for q in question_order])
        e = e[e["src"].isin(qpos.index)]
        ents = pd.Index(e["dst"].unique())
        self.M = sparse.csr_matrix(
            (np.ones(len(e)), (qpos.loc[e["src"]].to_numpy(), ents.get_indexer(e["dst"]))),
            shape=(len(question_order), len(ents)),
        )
        self.M.data[:] = 1.0  # binary
        self.deg = np.asarray(self.M.sum(axis=1)).ravel()

    def scores(self, query_pos: int) -> np.ndarray:
        """Jaccard(N(q), N(c)) for candidates c in [0, query_pos)."""
        inter = np.asarray((self.M[:query_pos] @ self.M[query_pos].T).todense()).ravel()
        union = self.deg[:query_pos] + self.deg[query_pos] - inter
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(union > 0, inter / union, 0.0)
