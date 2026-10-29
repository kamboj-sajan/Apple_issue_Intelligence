"""Graph statistics, components, communities, neighbourhood exploration, bridge detection and
time-stamped snapshots."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from ml.graph.build import filter_as_of
from ml.graph.schema import EDGE_TYPES, node_type_of

QQ_FEATURE_EDGES = {"QUESTION_SEMANTICALLY_SIMILAR", "QUESTION_TEMPORALLY_NEAR_QUESTION"}


def graph_statistics(nodes: pd.DataFrame, edges: pd.DataFrame) -> dict[str, Any]:
    deg = pd.concat([edges["src"], edges["dst"]]).value_counts()
    deg = deg.reindex(nodes["node_id"], fill_value=0)
    by_type = nodes.set_index("node_id")["node_type"]
    degree_summary = {
        t: {
            "n": int(len(d)),
            "mean": float(d.mean()),
            "median": float(d.median()),
            "p90": float(d.quantile(0.9)),
            "max": int(d.max()),
            "isolated": int((d == 0).sum()),
        }
        for t, d in deg.groupby(by_type)
    }
    n = len(nodes)
    return {
        "n_nodes": int(n),
        "n_edges": int(len(edges)),
        "density": float(2 * len(edges) / (n * (n - 1))) if n > 1 else 0.0,
        "nodes_by_type": nodes["node_type"].value_counts().to_dict(),
        "edges_by_type": edges["edge_type"].value_counts().to_dict(),
        "degree_by_node_type": degree_summary,
    }


def degree_distribution(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Histogram of degree per node type (log-binned) for plotting."""
    deg = pd.concat([edges["src"], edges["dst"]]).value_counts().reindex(nodes["node_id"], fill_value=0)
    df = pd.DataFrame({"node_type": nodes["node_type"].to_numpy(), "degree": deg.to_numpy()})
    bins = [0, 1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 5000, 10**9]
    df["bin"] = pd.cut(df["degree"], bins, right=False).astype(str)
    return df.groupby(["node_type", "bin"], observed=True).size().rename("count").reset_index()


def component_summary(G: nx.Graph) -> dict[str, Any]:
    sizes = sorted((len(c) for c in nx.connected_components(G)), reverse=True)
    n = G.number_of_nodes()
    return {
        "n_components": len(sizes),
        "giant_fraction": sizes[0] / n if sizes else 0.0,
        "largest_sizes": sizes[:5],
        "singletons": int(sum(1 for s in sizes if s == 1)),
    }


def question_projection(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.Graph:
    """Weighted simple graph over questions using only feature (non-label) question-question edges."""
    e = edges[edges["edge_type"].isin(QQ_FEATURE_EDGES)]
    agg = e.groupby(["src", "dst"])["weight"].sum().reset_index()
    G = nx.Graph()
    G.add_nodes_from(nodes.loc[nodes["node_type"] == "question", "node_id"])
    G.add_weighted_edges_from(agg.itertuples(index=False, name=None))
    return G


def communities(G: nx.Graph, seed: int = 0, resolution: float = 1.0, min_size: int = 10) -> pd.Series:
    """Louvain communities (seeded); nodes in communities smaller than ``min_size`` get -1."""
    comms = nx.community.louvain_communities(G, weight="weight", resolution=resolution, seed=seed)
    comms = sorted(comms, key=len, reverse=True)
    labels = {}
    for i, c in enumerate(comms):
        for nid in c:
            labels[nid] = i if len(c) >= min_size else -1
    return pd.Series(labels, name="community")


def describe_communities(labels: pd.Series, nodes: pd.DataFrame, edges: pd.DataFrame, top: int = 5) -> pd.DataFrame:
    """Top tags / OS versions / devices per community and its time span."""
    lab = labels[labels >= 0]
    attach = edges[
        edges["src"].isin(lab.index)
        & edges["edge_type"].isin(["QUESTION_HAS_TAG", "QUESTION_MENTIONS_VERSION", "QUESTION_MENTIONS_DEVICE"])
    ].assign(community=lambda d: d["src"].map(lab))
    created = nodes.set_index("node_id")["known_at"]
    rows = []
    for c, members in lab.groupby(lab):
        a = attach[attach["community"] == c]
        tops = {
            t: a.loc[a["edge_type"] == et, "dst"].str.split(":", n=1).str[1].value_counts().head(top).index.tolist()
            for t, et in (
                ("tags", "QUESTION_HAS_TAG"),
                ("os", "QUESTION_MENTIONS_VERSION"),
                ("devices", "QUESTION_MENTIONS_DEVICE"),
            )
        }
        times = created.loc[members.index]
        rows.append(
            {
                "community": int(c),
                "size": int(len(members)),
                **tops,
                "first": times.min(),
                "median_time": times.sort_values().iloc[len(times) // 2],
                "last": times.max(),
            }
        )
    return pd.DataFrame(rows).sort_values("size", ascending=False).reset_index(drop=True)


def bridging_communities(
    labels: pd.Series, edges: pd.DataFrame, node_prefix: str = "tag", min_share: float = 0.15, max_overlap: float = 0.1
) -> pd.DataFrame:
    """Communities that connect values (tags, devices, OS versions) which rarely co-occur on the same question.

    For each community, every pair of values that each cover >= ``min_share`` of its questions is
    scored by ``min(share_a, share_b) * (1 - overlap_ab)``, where ``overlap_ab = |A & B| / min(|A|, |B|)``
    is the global question-level overlap coefficient of the two values. (Jaccard was used first but is
    misleading when one value is very common, e.g. the ``macos`` tag on ~40 % of questions, because it
    is small even for pairs that co-occur constantly.) High scores mean the community links questions
    that use different vocabularies or concern different products/versions.
    """
    etype = {"tag": "QUESTION_HAS_TAG", "os": "QUESTION_MENTIONS_VERSION", "device": "QUESTION_MENTIONS_DEVICE"}[
        node_prefix
    ]
    att = edges.loc[edges["edge_type"] == etype, ["src", "dst"]]
    sets = att.groupby("dst")["src"].apply(set)
    lab = labels[labels >= 0]
    att = att[att["src"].isin(lab.index)].assign(community=lambda d: d["src"].map(lab))
    rows = []
    for c, a in att.groupby("community"):
        size = int((lab == c).sum())
        shares = a.groupby("dst")["src"].nunique() / size
        frequent = shares[shares >= min_share].index
        for x, y in combinations(sorted(frequent), 2):
            sx, sy = sets[x], sets[y]
            overlap = len(sx & sy) / min(len(sx), len(sy))
            if overlap <= max_overlap:
                rows.append(
                    {
                        "community": int(c),
                        "community_size": size,
                        "value_a": x,
                        "value_b": y,
                        "share_a": float(shares[x]),
                        "share_b": float(shares[y]),
                        "global_overlap": overlap,
                        "bridge_score": float(min(shares[x], shares[y]) * (1 - overlap)),
                    }
                )
    return (
        pd.DataFrame(rows).sort_values("bridge_score", ascending=False).reset_index(drop=True)
        if rows
        else pd.DataFrame()
    )


def neighbors(
    G: nx.MultiGraph,
    node: str,
    edge_types: set[str] | None = None,
    node_types: set[str] | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Typed neighbours of ``node`` with edge evidence, strongest first."""
    out = []
    for nbr, keyed in G[node].items():
        if node_types and node_type_of(nbr) not in node_types:
            continue
        for et, attrs in keyed.items():
            if edge_types and et not in edge_types:
                continue
            out.append(
                {
                    "node_id": nbr,
                    "node_type": node_type_of(nbr),
                    "label": G.nodes[nbr].get("label"),
                    "edge_type": et,
                    "weight": float(attrs["weight"]),
                    "direction": "out" if attrs["src"] == node else "in",
                    "evidence": attrs.get("evidence"),
                    "is_label_edge": EDGE_TYPES[et].label_edge,
                }
            )
    return sorted(out, key=lambda r: -r["weight"])[:limit]


def ego_subgraph(G: nx.MultiGraph, node: str, depth: int = 1, max_nodes: int = 200, **filters) -> dict[str, list]:
    """Bounded neighbourhood for visualisation: breadth-first, strongest edges first, capped at ``max_nodes``."""
    seen, frontier, edges_out = {node}, [node], []
    for _ in range(depth):
        nxt = []
        for u in frontier:
            for r in neighbors(G, u, limit=max_nodes, **filters):
                v = r["node_id"]
                if v not in seen and len(seen) >= max_nodes:
                    continue
                edges_out.append({"source": u, "target": v, **{k: r[k] for k in ("edge_type", "weight", "evidence")}})
                if v not in seen:
                    seen.add(v)
                    nxt.append(v)
        frontier = nxt
    nodes_out = [{"node_id": n, "node_type": node_type_of(n), "label": G.nodes[n].get("label")} for n in seen]
    return {"nodes": nodes_out, "edges": edges_out, "truncated": len(seen) >= max_nodes}


def snapshot_series(nodes: pd.DataFrame, edges: pd.DataFrame, as_of_dates: list[pd.Timestamp]) -> pd.DataFrame:
    """Structure of the graph as it existed at each date (built only from facts known before it)."""
    rows = []
    for t in as_of_dates:
        n, e = filter_as_of(nodes, edges, t)
        if n.empty:
            continue
        qq = question_projection(n, e)
        comp = component_summary(qq)
        stats = graph_statistics(n, e)
        rows.append(
            {
                "as_of": t,
                "n_nodes": stats["n_nodes"],
                "n_edges": stats["n_edges"],
                **{f"nodes_{k}": v for k, v in stats["nodes_by_type"].items()},
                **{f"edges_{k}": v for k, v in stats["edges_by_type"].items()},
                "qq_components": comp["n_components"],
                "qq_giant_fraction": comp["giant_fraction"],
                "qq_singletons": comp["singletons"],
                "mean_question_degree": stats["degree_by_node_type"].get("question", {}).get("mean", np.nan),
            }
        )
    return pd.DataFrame(rows)
