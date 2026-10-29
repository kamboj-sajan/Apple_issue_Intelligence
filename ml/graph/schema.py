"""Heterogeneous graph schema: node types, edge types, identifiers and validation.

Node ids are ``"<node_type>:<value>"``, e.g. ``question:488267``, ``tag:wifi``, ``os:macos:14``,
``error:error:-36``, ``week:2023-09-04``. They are stable across rebuilds of the same snapshot
because they derive only from source identifiers and canonical entity values.

Every node and edge carries ``known_at``: the earliest time the fact was observable. Filtering on
``known_at < t`` reconstructs the graph as it could have been built at time ``t``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

NODE_TYPES = ("question", "tag", "device", "os", "error", "tech", "category", "week")


@dataclass(frozen=True)
class EdgeType:
    name: str
    src: str
    dst: str
    weight: str  # how the weight is derived (documentation)
    label_edge: bool = False  # True for link-prediction targets; excluded from prediction graphs


EDGE_TYPES = {
    e.name: e
    for e in (
        EdgeType("QUESTION_HAS_TAG", "question", "tag", "1.0 (explicit tag)"),
        EdgeType("QUESTION_MENTIONS_DEVICE", "question", "device", "field weight: title/tags 1.0, body 0.6, code 0.4"),
        EdgeType("QUESTION_MENTIONS_VERSION", "question", "os", "field weight: title/tags 1.0, body 0.6, code 0.4"),
        EdgeType("QUESTION_CONTAINS_ERROR", "question", "error", "field weight: title/tags 1.0, body 0.6, code 0.4"),
        EdgeType("QUESTION_MENTIONS_TECH", "question", "tech", "field weight: title/tags 1.0, body 0.6, code 0.4"),
        EdgeType("QUESTION_IN_CATEGORY", "question", "category", "rule score / max rule score of the question"),
        EdgeType("QUESTION_POSTED_IN_WINDOW", "question", "week", "1.0"),
        EdgeType(
            "QUESTION_SEMANTICALLY_SIMILAR", "question", "question", "cosine similarity (top-k earlier, >= threshold)"
        ),
        EdgeType(
            "QUESTION_TEMPORALLY_NEAR_QUESTION", "question", "question", "shared specific entities x exp(-days/tau)"
        ),
        EdgeType("ISSUE_SPIKED_DURING_WINDOW", "*issue", "week", "-log10(q) of the anomaly signal"),
        EdgeType("QUESTION_DUPLICATE_OF", "question", "question", "1.0 (community duplicate closure)", label_edge=True),
        EdgeType("QUESTION_LINKS_TO", "question", "question", "1.0 (hyperlink in question body)", label_edge=True),
    )
}
ISSUE_NODE_TYPES = ("tag", "device", "os", "error", "tech", "category")
FIELD_WEIGHT = {"title": 1.0, "tags": 1.0, "body": 0.6, "code": 0.4}

NODE_COLUMNS = ["node_id", "node_type", "label", "known_at"]
EDGE_COLUMNS = ["src", "dst", "edge_type", "weight", "known_at", "evidence"]


def node_id(node_type: str, value: object) -> str:
    return f"{node_type}:{value}"


def node_type_of(nid: str) -> str:
    return nid.split(":", 1)[0]


class SchemaError(ValueError):
    pass


def validate(nodes: pd.DataFrame, edges: pd.DataFrame) -> dict[str, int]:
    """Raise SchemaError on violations; return counts of checks performed."""
    problems = []
    if not nodes["node_id"].is_unique:
        problems.append("duplicate node ids")
    bad_types = set(nodes["node_type"]) - set(NODE_TYPES)
    if bad_types:
        problems.append(f"unknown node types {bad_types}")
    prefix = nodes["node_id"].str.split(":", n=1).str[0]
    if (prefix != nodes["node_type"]).any():
        problems.append("node id prefix does not match node_type")
    unknown_edges = set(edges["edge_type"]) - set(EDGE_TYPES)
    if unknown_edges:
        problems.append(f"unknown edge types {unknown_edges}")
    known = set(nodes["node_id"])
    dangling = ~edges["src"].isin(known) | ~edges["dst"].isin(known)
    if dangling.any():
        problems.append(f"{int(dangling.sum())} edges reference missing nodes")
    if (edges["src"] == edges["dst"]).any():
        problems.append("self loops")
    if edges.duplicated(["src", "dst", "edge_type"]).any():
        problems.append("duplicate (src, dst, edge_type)")
    w = edges["weight"].to_numpy(dtype=float)
    if not np.isfinite(w).all() or (w <= 0).any():
        problems.append("weights must be finite and > 0")
    if edges["known_at"].isna().any() or nodes["known_at"].isna().any():
        problems.append("missing known_at")
    src_t = edges["src"].str.split(":", n=1).str[0]
    dst_t = edges["dst"].str.split(":", n=1).str[0]
    for et, spec in EDGE_TYPES.items():
        m = edges["edge_type"] == et
        if not m.any():
            continue
        ok_src = src_t[m].isin(ISSUE_NODE_TYPES) if spec.src == "*issue" else src_t[m].eq(spec.src)
        if not (ok_src.all() and dst_t[m].eq(spec.dst).all()):
            problems.append(f"{et} endpoints violate schema")
    # An edge cannot be known before both of its endpoints exist.
    node_time = nodes.set_index("node_id")["known_at"]
    early = (edges["known_at"] < edges["src"].map(node_time)) | (edges["known_at"] < edges["dst"].map(node_time))
    if early.any():
        problems.append(f"{int(early.sum())} edges known before an endpoint")
    if problems:
        raise SchemaError("; ".join(problems))
    return {"nodes": len(nodes), "edges": len(edges), "edge_types": int(edges["edge_type"].nunique())}
