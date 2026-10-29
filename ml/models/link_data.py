"""Tensors for GraphSAGE link prediction, built from the Milestone 4 graph tables.

* **Message-passing graph:** feature edges only (tags, devices, OS versions, errors, technical
  entities, categories, semantic and temporal question-question edges), in both directions.
  Label edges (duplicate / body link) are never used for message passing, so a positive pair
  can't be "read off" the graph. Week nodes and spike edges are left out (time hubs).
* **Node features:** questions use their sentence-transformer embedding; every other node uses the
  embedding of a short text such as ``"tag: macos-sonoma"`` or ``"os: macos:14"``. Every node
  therefore has content features, and unseen entities can be embedded at inference (inductive).
* **Time:** each edge keeps ``known_at``. ``edge_index_as_of(t)`` returns only edges known at or
  before ``t``, so a query is embedded with the graph as it existed when it was posted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

FEATURE_EDGE_TYPES = (
    "QUESTION_HAS_TAG",
    "QUESTION_MENTIONS_DEVICE",
    "QUESTION_MENTIONS_VERSION",
    "QUESTION_CONTAINS_ERROR",
    "QUESTION_MENTIONS_TECH",
    "QUESTION_IN_CATEGORY",
    "QUESTION_SEMANTICALLY_SIMILAR",
    "QUESTION_TEMPORALLY_NEAR_QUESTION",
)
NODE_TYPES = ("question", "tag", "device", "os", "error", "tech", "category")


def to_ns(values) -> np.ndarray:
    """UTC timestamps -> int64 nanoseconds. Always use this: pandas may store datetimes in
    microseconds, and a bare ``astype("int64")`` then silently returns microseconds, which makes
    every comparison with ``Timestamp.value`` (nanoseconds) wrong by a factor of 1000."""
    return np.asarray(pd.DatetimeIndex(pd.to_datetime(values, utc=True)).as_unit("ns").asi8)


def entity_text(node_id: str) -> str:
    t, v = node_id.split(":", 1)
    return f"{t}: {v.replace('-', ' ').replace(':', ' ')}"


@dataclass
class LinkGraph:
    node_ids: np.ndarray
    node_type: torch.Tensor  # [N] int
    x: torch.Tensor  # [N, d]
    edge_src: np.ndarray  # directed, both directions included
    edge_dst: np.ndarray
    edge_time: np.ndarray  # int64 ns
    question_node: pd.Series  # question_id -> node index

    def edge_index_as_of(self, t_ns: int | None = None) -> torch.Tensor:
        m = slice(None) if t_ns is None else self.edge_time <= t_ns
        return torch.from_numpy(np.vstack([self.edge_src[m], self.edge_dst[m]]))


def build_link_graph(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    question_emb: np.ndarray,
    question_emb_ids: np.ndarray,
    entity_emb: dict[str, np.ndarray],
) -> LinkGraph:
    keep_nodes = nodes[nodes["node_type"].isin(NODE_TYPES)].reset_index(drop=True)
    idx = pd.Series(np.arange(len(keep_nodes)), index=keep_nodes["node_id"].to_numpy())
    e = edges[edges["edge_type"].isin(FEATURE_EDGE_TYPES) & edges["src"].isin(idx.index) & edges["dst"].isin(idx.index)]
    s, d = idx.loc[e["src"]].to_numpy(), idx.loc[e["dst"]].to_numpy()
    t = to_ns(e["known_at"])

    dim = question_emb.shape[1]
    x = np.zeros((len(keep_nodes), dim), dtype=np.float32)
    qmask = (keep_nodes["node_type"] == "question").to_numpy()
    qids = keep_nodes.loc[qmask, "node_id"].str.split(":").str[1].astype(np.int64).to_numpy()
    epos = pd.Series(np.arange(len(question_emb_ids)), index=question_emb_ids)
    x[qmask] = question_emb[epos.loc[qids].to_numpy()]
    for i in np.nonzero(~qmask)[0]:
        x[i] = entity_emb[keep_nodes.at[i, "node_id"]]

    type_index = {t: i for i, t in enumerate(NODE_TYPES)}
    return LinkGraph(
        node_ids=keep_nodes["node_id"].to_numpy(),
        node_type=torch.tensor(keep_nodes["node_type"].map(type_index).to_numpy(), dtype=torch.long),
        x=torch.from_numpy(x),
        edge_src=np.concatenate([s, d]),
        edge_dst=np.concatenate([d, s]),
        edge_time=np.concatenate([t, t]),
        question_node=pd.Series(np.nonzero(qmask)[0], index=qids),
    )


def label_pairs(links: pd.DataFrame, questions: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Undirected label pairs oriented (query = later question, target = earlier question)."""
    created = questions.set_index("question_id")["created_at"]
    lk = links[links["source_id"].isin(created.index) & links["target_id"].isin(created.index)]
    a, b = lk["source_id"].to_numpy(), lk["target_id"].to_numpy()
    later = created.loc[a].to_numpy() >= created.loc[b].to_numpy()
    out = pd.DataFrame(
        {
            "query_id": np.where(later, a, b),
            "target_id": np.where(later, b, a),
            "link_type": lk["link_type"].to_numpy(),
            "known_at": pd.to_datetime(lk["known_at"], utc=True).to_numpy(),
        }
    )
    out = out[out["query_id"] != out["target_id"]]
    sp = splits.set_index("question_id")
    out["split"] = out["query_id"].map(sp["split"]).to_numpy()
    out["query_in_window"] = out["query_id"].map(sp["in_research_window"]).fillna(False).astype(bool).to_numpy()
    agg = out.groupby(["query_id", "target_id"], as_index=False).agg(
        link_type=("link_type", lambda s: "duplicate" if "duplicate" in set(s) else "body_link"),
        known_at=("known_at", "min"),
        split=("split", "first"),
        query_in_window=("query_in_window", "first"),
    )
    return agg
