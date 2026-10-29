"""Build node and edge tables for the heterogeneous question graph.

Construction rules (also in docs/graph_schema.md):

* Entity edges come from the rule-based entity table; the weight reflects where the evidence was
  found (title/tags > body > code). Category edges use the normalised rule score.
* Semantic edges link each question to at most ``knn_k`` *earlier* questions whose embedding cosine is
  at least ``knn_threshold``, so a question never connects to the future.
* Temporal-proximity edges link a question to at most ``temporal_max_per_question`` earlier
  questions posted within ``temporal_window_days`` that share at least ``temporal_min_shared``
  *specific* entities (devices, OS versions, errors, technical entities, tags), ignoring
  generic ones that occur in more than ``generic_max_df`` of questions.
* Duplicate and body-link edges are label edges (link-prediction targets), flagged in the schema.
* Anomaly signals become ISSUE_SPIKED_DURING_WINDOW edges, known at the end of the signal week.
"""

from __future__ import annotations

import json
from collections import Counter, deque
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

from ml.graph.schema import EDGE_COLUMNS, EDGE_TYPES, FIELD_WEIGHT, ISSUE_NODE_TYPES, NODE_COLUMNS, node_id

ENTITY_EDGE = {
    "device": "QUESTION_MENTIONS_DEVICE",
    "os": "QUESTION_MENTIONS_VERSION",
    "error": "QUESTION_CONTAINS_ERROR",
    "tech": "QUESTION_MENTIONS_TECH",
}
LINK_EDGE = {"duplicate": "QUESTION_DUPLICATE_OF", "body_link": "QUESTION_LINKS_TO"}


@dataclass
class GraphConfig:
    knn_k: int = 10
    knn_threshold: float = 0.6
    knn_block: int = 2048
    temporal_window_days: float = 7.0
    temporal_min_shared: int = 2
    temporal_max_per_question: int = 5
    temporal_tau_days: float = 3.0
    generic_max_df: float = 0.05


def _qid(ids) -> list[str]:
    return [node_id("question", int(i)) for i in ids]


def question_nodes(questions: pd.DataFrame, splits: pd.DataFrame | None) -> pd.DataFrame:
    q = questions
    nodes = pd.DataFrame(
        {
            "node_id": _qid(q["question_id"]),
            "node_type": "question",
            "label": q["title"].to_numpy(),
            "known_at": q["created_at"].to_numpy(),
            "question_id": q["question_id"].to_numpy(),
            "link": q["link"].to_numpy(),
            "in_research_window": q.get("in_research_window", pd.Series(True, index=q.index)).to_numpy(),
            "primary_category": q.get("primary_category", pd.Series(None, index=q.index)).to_numpy(),
        }
    )
    if splits is not None:
        nodes["split"] = nodes["question_id"].map(splits.set_index("question_id")["split"]).to_numpy()
    return nodes


def entity_and_tag_edges(questions: pd.DataFrame, entities: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    created = questions.set_index("question_id")["created_at"]
    ent = entities[entities["question_id"].isin(created.index)].copy()
    ent["created_at"] = ent["question_id"].map(created)
    ent["src"] = _qid(ent["question_id"])

    parts = []
    e = ent[ent["entity_type"].isin(ENTITY_EDGE)].copy()
    e["dst"] = [node_id(t, v) for t, v in zip(e["entity_type"], e["value"], strict=True)]
    e["w"] = e["field"].map(FIELD_WEIGHT).fillna(0.5)
    e = e.sort_values("w", ascending=False).drop_duplicates(["src", "dst"])
    parts.append(
        pd.DataFrame(
            {
                "src": e["src"],
                "dst": e["dst"],
                "edge_type": e["entity_type"].map(ENTITY_EDGE),
                "weight": e["w"],
                "known_at": e["created_at"],
                "evidence": e["field"] + ": " + e["surface"].astype(str).str.slice(0, 80),
            }
        )
    )

    c = ent[ent["entity_type"] == "category"].copy()
    c["score"] = c["rule"].str.extract(r"score=(\d+)")[0].astype(float)
    c["w"] = c["score"] / c.groupby("question_id")["score"].transform("max")
    parts.append(
        pd.DataFrame(
            {
                "src": c["src"],
                "dst": [node_id("category", v) for v in c["value"]],
                "edge_type": "QUESTION_IN_CATEGORY",
                "weight": c["w"],
                "known_at": c["created_at"],
                "evidence": c["surface"].astype(str).str.slice(0, 120),
            }
        )
    )

    t = questions[["question_id", "tags", "created_at"]].explode("tags").dropna(subset=["tags"])
    parts.append(
        pd.DataFrame(
            {
                "src": _qid(t["question_id"]),
                "dst": [node_id("tag", v) for v in t["tags"]],
                "edge_type": "QUESTION_HAS_TAG",
                "weight": 1.0,
                "known_at": t["created_at"],
                "evidence": "tag",
            }
        )
    )
    parts.append(
        pd.DataFrame(
            {
                "src": _qid(questions["question_id"]),
                "dst": [node_id("week", w.date()) for w in questions["week_start"]],
                "edge_type": "QUESTION_POSTED_IN_WINDOW",
                "weight": 1.0,
                "known_at": questions["created_at"].to_numpy(),
                "evidence": "created_at",
            }
        )
    )
    edges = pd.concat(parts, ignore_index=True).drop_duplicates(["src", "dst", "edge_type"])

    # Entity, tag, category and week nodes: known from their first appearance.
    first = edges.groupby("dst")["known_at"].min()
    ent_nodes = pd.DataFrame({"node_id": first.index, "known_at": first.to_numpy()})
    ent_nodes["node_type"] = ent_nodes["node_id"].str.split(":", n=1).str[0]
    ent_nodes["label"] = ent_nodes["node_id"].str.split(":", n=1).str[1]
    weeks = ent_nodes["node_type"] == "week"
    ent_nodes.loc[weeks, "known_at"] = pd.to_datetime(ent_nodes.loc[weeks, "label"]).dt.tz_localize("UTC").to_numpy()
    return ent_nodes[NODE_COLUMNS], edges[EDGE_COLUMNS]


def semantic_knn_edges(
    emb: np.ndarray, question_ids: np.ndarray, created_at: pd.Series, cfg: GraphConfig
) -> pd.DataFrame:
    """Top-k most similar *earlier* questions (by creation order) above the cosine threshold."""
    order = np.argsort(created_at.to_numpy(), kind="stable")
    E, ids, times = emb[order], np.asarray(question_ids)[order], created_at.to_numpy()[order]
    n, k = len(E), cfg.knn_k
    src, dst, w = [], [], []
    for s in range(1, n, cfg.knn_block):
        e = min(s + cfg.knn_block, n)
        sims = E[s:e] @ E[:e].T  # only columns < e can be earlier
        rows = np.arange(s, e)
        sims[np.arange(e)[None, :] >= rows[:, None]] = -np.inf  # mask self and later questions
        kk = min(k, e - 1)
        top = np.argpartition(-sims, kk - 1, axis=1)[:, :kk] if kk > 0 else np.empty((e - s, 0), int)
        for r, cols in enumerate(top):
            vals = sims[r, cols]
            keep = vals >= cfg.knn_threshold
            src.extend([ids[s + r]] * int(keep.sum()))
            dst.extend(ids[cols[keep]])
            w.extend(vals[keep])
    pos = pd.Series(np.arange(n), index=ids)
    out = pd.DataFrame({"src": _qid(src), "dst": _qid(dst), "weight": np.asarray(w, dtype=float)})
    out["known_at"] = times[pos.loc[src].to_numpy()] if len(src) else pd.Series(dtype="datetime64[ns, UTC]")
    out["edge_type"] = "QUESTION_SEMANTICALLY_SIMILAR"
    out["evidence"] = "cosine=" + out["weight"].round(3).astype(str)
    return out[EDGE_COLUMNS]


def temporal_near_edges(questions: pd.DataFrame, entities: pd.DataFrame, cfg: GraphConfig) -> pd.DataFrame:
    q = questions.sort_values(["created_at", "question_id"])
    specific = entities[entities["entity_type"].isin(["device", "os", "error", "tech"])]
    keys = pd.concat(
        [
            specific[["question_id"]].assign(key=specific["entity_type"] + ":" + specific["value"]),
            q[["question_id", "tags"]]
            .explode("tags")
            .dropna()
            .rename(columns={"tags": "key"})
            .assign(key=lambda d: "tag:" + d["key"]),
        ]
    ).drop_duplicates()
    df = keys["key"].value_counts() / len(q)
    keys = keys[keys["key"].map(df) <= cfg.generic_max_df]
    sets = keys.groupby("question_id")["key"].apply(frozenset)

    window = pd.Timedelta(days=cfg.temporal_window_days)
    postings: dict[str, deque] = {}
    other_time: dict[int, pd.Timestamp] = {}
    rows = []
    for qid, t in zip(q["question_id"], q["created_at"], strict=True):
        mine = sets.get(qid, frozenset())
        shared: Counter = Counter()
        for key in mine:
            dq = postings.setdefault(key, deque())
            while dq and dq[0][0] < t - window:
                dq.popleft()
            for _, other in dq:
                shared[other] += 1
        cands = []
        for other, cnt in shared.items():
            if cnt >= cfg.temporal_min_shared:
                ot = other_time[other]
                days = (t - ot).total_seconds() / 86400
                cands.append((cnt * np.exp(-days / cfg.temporal_tau_days), other, cnt))
        for weight, other, cnt in sorted(cands, reverse=True)[: cfg.temporal_max_per_question]:
            evid = ", ".join(sorted(mine & sets.get(other, frozenset()))[:6])
            rows.append((qid, other, weight, t, f"{cnt} shared: {evid}"))
        for key in mine:
            postings[key].append((t, qid))
        other_time[qid] = t
    out = pd.DataFrame(rows, columns=["s", "d", "weight", "known_at", "evidence"])
    out["src"], out["dst"] = _qid(out["s"]), _qid(out["d"])
    out["edge_type"] = "QUESTION_TEMPORALLY_NEAR_QUESTION"
    return out[EDGE_COLUMNS]


def label_edges(links: pd.DataFrame, questions: pd.DataFrame) -> pd.DataFrame:
    created = questions.set_index("question_id")["created_at"]
    lk = links[links["source_id"].isin(created.index) & links["target_id"].isin(created.index)].copy()
    # A link cannot be known before both endpoints exist.
    known = pd.concat([lk["known_at"], lk["source_id"].map(created), lk["target_id"].map(created)], axis=1).max(axis=1)
    return pd.DataFrame(
        {
            "src": _qid(lk["source_id"]),
            "dst": _qid(lk["target_id"]),
            "edge_type": lk["link_type"].map(LINK_EDGE),
            "weight": 1.0,
            "known_at": known,
            "evidence": lk["link_type"],
        }
    )[EDGE_COLUMNS]


def spike_edges(signals: pd.DataFrame, node_ids: set[str]) -> pd.DataFrame:
    s = signals[signals["group_type"].isin(ISSUE_NODE_TYPES)].copy()
    s["src"] = [node_id(t, v) for t, v in zip(s["group_type"], s["group_value"], strict=True)]
    s["dst"] = [node_id("week", w.date()) for w in s["week_start"]]
    s = s[s["src"].isin(node_ids) & s["dst"].isin(node_ids)]
    return pd.DataFrame(
        {
            "src": s["src"],
            "dst": s["dst"],
            "edge_type": "ISSUE_SPIKED_DURING_WINDOW",
            "weight": (-np.log10(s["q_value"].clip(1e-300, 1))).clip(lower=1e-3),
            "known_at": s["week_start"] + pd.Timedelta(days=7),
            "evidence": [
                json.dumps({"anomaly_id": a, "observed": int(o), "expected": round(float(x), 2), "signal_type": st})
                for a, o, x, st in zip(s["anomaly_id"], s["observed"], s["expected"], s["signal_type"], strict=True)
            ],
        }
    )[EDGE_COLUMNS]


def build_graph_tables(
    questions: pd.DataFrame,
    entities: pd.DataFrame,
    links: pd.DataFrame,
    emb: np.ndarray | None = None,
    emb_ids: np.ndarray | None = None,
    signals: pd.DataFrame | None = None,
    splits: pd.DataFrame | None = None,
    cfg: GraphConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = cfg or GraphConfig()
    qnodes = question_nodes(questions, splits)
    enodes, eedges = entity_and_tag_edges(questions, entities)
    parts = [eedges, temporal_near_edges(questions, entities, cfg), label_edges(links, questions)]
    if emb is not None:
        pos = pd.Series(np.arange(len(emb_ids)), index=emb_ids)
        q = questions[questions["question_id"].isin(pos.index)]
        E = emb[pos.loc[q["question_id"]].to_numpy()]
        parts.append(semantic_knn_edges(E, q["question_id"].to_numpy(), q["created_at"].reset_index(drop=True), cfg))
    nodes = pd.concat([qnodes, enodes], ignore_index=True)
    if signals is not None and len(signals):
        parts.append(spike_edges(signals, set(nodes["node_id"])))
    edges = pd.concat(parts, ignore_index=True).drop_duplicates(["src", "dst", "edge_type"]).reset_index(drop=True)
    edges["known_at"] = pd.to_datetime(edges["known_at"], utc=True)
    nodes["known_at"] = pd.to_datetime(nodes["known_at"], utc=True)
    return nodes, edges


def filter_as_of(nodes: pd.DataFrame, edges: pd.DataFrame, as_of: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The graph exactly as it could have been built at ``as_of`` (strictly earlier facts only)."""
    return nodes[nodes["known_at"] < as_of], edges[edges["known_at"] < as_of]


def to_networkx(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    edge_types: set[str] | None = None,
    include_label_edges: bool = True,
) -> nx.MultiGraph:
    """Undirected multigraph; parallel edges are keyed by edge type; ``src``/``dst`` keep direction."""
    G = nx.MultiGraph()
    attrs = nodes.drop(columns=["node_id"]).to_dict("records")
    G.add_nodes_from(zip(nodes["node_id"], attrs, strict=True))
    e = edges
    if edge_types is not None:
        e = e[e["edge_type"].isin(edge_types)]
    if not include_label_edges:
        e = e[~e["edge_type"].map(lambda t: EDGE_TYPES[t].label_edge)]
    G.add_edges_from(
        (s, d, t, {"edge_type": t, "weight": w, "known_at": k, "evidence": ev, "src": s, "dst": d})
        for s, d, t, w, k, ev in zip(
            e["src"], e["dst"], e["edge_type"], e["weight"], e["known_at"], e["evidence"], strict=True
        )
    )
    return G
