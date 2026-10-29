"""Knowledge-graph tests on small SYNTHETIC inputs."""

import numpy as np
import pandas as pd
import pytest

from ml.graph import analysis as A
from ml.graph.build import (
    GraphConfig,
    build_graph_tables,
    filter_as_of,
    semantic_knn_edges,
    temporal_near_edges,
    to_networkx,
)
from ml.graph.schema import SchemaError, validate

T0 = pd.Timestamp("2024-01-01", tz="UTC")


def corpus():
    n = 8
    created = [T0 + pd.Timedelta(days=d) for d in (0, 1, 2, 3, 20, 21, 22, 40)]
    q = pd.DataFrame(
        {
            "question_id": np.arange(100, 100 + n),
            "title": [f"q{i}" for i in range(n)],
            "link": [f"https://apple.stackexchange.com/questions/{100 + i}" for i in range(n)],
            "created_at": created,
            "tags": [
                ["wifi", "macos"],
                ["wifi", "macos"],
                ["battery"],
                ["wifi", "macos"],
                ["bootcamp", "windows"],
                ["bootcamp", "windows"],
                ["display"],
                ["wifi"],
            ],
            "in_research_window": True,
            "primary_category": "network-connectivity",
        }
    )
    q["week_start"] = q["created_at"].dt.tz_localize(None).dt.to_period("W-SUN").dt.start_time.dt.tz_localize("UTC")
    ent = pd.DataFrame(
        [
            (100, "device", "macbook-pro", "MacBook Pro", "title", "r"),
            (100, "os", "macos:14", "Sonoma", "body", "r"),
            (101, "device", "macbook-pro", "MBP", "body", "r"),
            (101, "os", "macos:14", "macOS 14", "code", "r"),
            (103, "device", "macbook-pro", "MacBook Pro", "title", "r"),
            (103, "os", "macos:14", "Sonoma", "title", "r"),
            (104, "error", "error:-36", "error -36", "code", "r"),
            (100, "category", "network-connectivity", "tag:wifi", "rules", "score=3"),
            (102, "category", "battery-power", "battery", "rules", "score=1"),
            (102, "category", "update-install", "update", "rules", "score=2"),
        ],
        columns=["question_id", "entity_type", "value", "surface", "field", "rule"],
    )
    links = pd.DataFrame(
        {
            "source_id": [103, 105],
            "target_id": [100, 104],
            "link_type": ["duplicate", "body_link"],
            "known_at": [T0 + pd.Timedelta(days=10), T0 + pd.Timedelta(days=21)],
            "target_in_corpus": True,
        }
    )
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(n, 16)).astype(np.float32)
    emb[3] = emb[0] + 0.05 * rng.normal(size=16)  # q103 ~ q100
    emb[5] = emb[4] + 0.05 * rng.normal(size=16)  # q105 ~ q104
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    splits = pd.DataFrame({"question_id": q["question_id"], "split": ["train"] * 4 + ["val"] * 2 + ["test"] * 2})
    signals = pd.DataFrame(
        {
            "group_type": ["tag"],
            "group_value": ["wifi"],
            "week_start": [q["week_start"][0]],
            "q_value": [0.001],
            "anomaly_id": ["abc"],
            "observed": [3],
            "expected": [1.0],
            "signal_type": ["spike"],
        }
    )
    return q, ent, links, emb, splits, signals


CFG = GraphConfig(knn_k=2, knn_threshold=0.5, temporal_min_shared=2, generic_max_df=1.0)


def build(cfg=CFG):
    q, ent, links, emb, splits, signals = corpus()
    return build_graph_tables(q, ent, links, emb, q["question_id"].to_numpy(), signals, splits, cfg)


def test_build_validates_and_has_all_edge_types():
    nodes, edges = build()
    assert validate(nodes, edges)["nodes"] == len(nodes)
    assert {
        "QUESTION_HAS_TAG",
        "QUESTION_MENTIONS_DEVICE",
        "QUESTION_MENTIONS_VERSION",
        "QUESTION_CONTAINS_ERROR",
        "QUESTION_IN_CATEGORY",
        "QUESTION_POSTED_IN_WINDOW",
        "QUESTION_SEMANTICALLY_SIMILAR",
        "QUESTION_TEMPORALLY_NEAR_QUESTION",
        "ISSUE_SPIKED_DURING_WINDOW",
        "QUESTION_DUPLICATE_OF",
        "QUESTION_LINKS_TO",
    } <= set(edges["edge_type"])
    assert {"question:100", "tag:wifi", "os:macos:14", "error:error:-36", "week:2024-01-01"} <= set(nodes["node_id"])


def test_entity_edge_weights_reflect_evidence_field_and_rule_score():
    _, edges = build()
    e = edges.set_index(["src", "dst"])["weight"]
    assert e[("question:100", "device:macbook-pro")] == 1.0  # title
    assert e[("question:101", "os:macos:14")] == 0.4  # code only
    assert e[("question:102", "category:update-install")] == 1.0
    assert e[("question:102", "category:battery-power")] == 0.5


def test_semantic_edges_only_point_to_earlier_questions_above_threshold():
    q, _, _, emb, _, _ = corpus()
    e = semantic_knn_edges(emb, q["question_id"].to_numpy(), q["created_at"], GraphConfig(knn_k=3, knn_threshold=0.9))
    t = q.set_index("question_id")["created_at"]
    src = e["src"].str.split(":").str[1].astype(int)
    dst = e["dst"].str.split(":").str[1].astype(int)
    assert (src.map(t) > dst.map(t)).all()
    assert (e["weight"] >= 0.9).all() and e.groupby("src").size().max() <= 3
    assert {("question:103", "question:100"), ("question:105", "question:104")} <= set(
        zip(e["src"], e["dst"], strict=True)
    )


def test_temporal_edges_need_shared_specific_entities_within_window():
    q, ent, *_ = corpus()
    e = temporal_near_edges(q, ent, GraphConfig(temporal_min_shared=2, generic_max_df=1.0))
    pairs = set(zip(e["src"], e["dst"], strict=True))
    assert ("question:101", "question:100") in pairs  # 1 day apart, share wifi+macos+device+os
    assert ("question:107", "question:103") not in pairs  # shares only 'wifi' and is 37 days later
    assert ("question:105", "question:104") in pairs
    assert all(int(s.split(":")[1]) > int(d.split(":")[1]) for s, d in pairs)
    # Generic filter: if everything is "too common", no edges.
    assert temporal_near_edges(q, ent, GraphConfig(generic_max_df=0.0)).empty


def test_label_edges_flagged_and_excluded_from_prediction_graph():
    nodes, edges = build()
    dup = edges[edges["edge_type"] == "QUESTION_DUPLICATE_OF"].iloc[0]
    assert dup["known_at"] == T0 + pd.Timedelta(days=10)  # closure time, not creation time
    G = to_networkx(nodes, edges, include_label_edges=False)
    assert not any(k in ("QUESTION_DUPLICATE_OF", "QUESTION_LINKS_TO") for _, _, k in G.edges(keys=True))


def test_as_of_snapshot_contains_no_future_facts():
    nodes, edges = build()
    t = T0 + pd.Timedelta(days=5)
    n, e = filter_as_of(nodes, edges, t)
    assert (e["known_at"] < t).all() and (n["known_at"] < t).all()
    assert "question:104" not in set(n["node_id"])
    assert "QUESTION_DUPLICATE_OF" not in set(e["edge_type"])  # closed on day 10
    validate(n, e)  # a snapshot is itself a valid graph


def test_validation_catches_schema_violations():
    nodes, edges = build()
    bad = pd.concat([edges, edges.iloc[[0]].assign(dst="tag:missing")])
    with pytest.raises(SchemaError, match="missing nodes"):
        validate(nodes, bad)
    swapped = edges.copy()
    m = swapped["edge_type"] == "QUESTION_HAS_TAG"
    swapped.loc[m, ["src", "dst"]] = swapped.loc[m, ["dst", "src"]].to_numpy()
    with pytest.raises(SchemaError, match="QUESTION_HAS_TAG endpoints"):
        validate(nodes, swapped)
    early = edges.copy()
    early.loc[0, "known_at"] = T0 - pd.Timedelta(days=30)
    with pytest.raises(SchemaError, match="known before an endpoint"):
        validate(nodes, early)


def test_neighbors_ego_subgraph_and_statistics():
    nodes, edges = build()
    G = to_networkx(nodes, edges)
    nb = A.neighbors(G, "question:103", node_types={"question"})
    assert {r["node_id"] for r in nb} >= {"question:100"}
    dup = [r for r in nb if r["edge_type"] == "QUESTION_DUPLICATE_OF"][0]
    assert dup["direction"] == "out" and dup["is_label_edge"]
    ego = A.ego_subgraph(G, "tag:wifi", depth=2, max_nodes=5)
    assert len(ego["nodes"]) <= 5 and ego["truncated"]
    stats = A.graph_statistics(nodes, edges)
    assert stats["nodes_by_type"]["question"] == 8 and stats["n_edges"] == len(edges)


def test_communities_and_bridges_on_constructed_graph():
    # Two dense question cliques; clique 0 mixes tags 'x' and 'y' that never co-occur on a question.
    qs = [f"question:{i}" for i in range(20)]
    nodes = pd.DataFrame(
        {
            "node_id": qs + ["tag:x", "tag:y", "tag:z"],
            "node_type": ["question"] * 20 + ["tag"] * 3,
            "label": "",
            "known_at": T0,
        }
    )
    rows = []
    for block in (range(0, 10), range(10, 20)):
        for a in block:
            for b in block:
                if a > b:
                    rows.append((f"question:{a}", f"question:{b}", "QUESTION_SEMANTICALLY_SIMILAR", 0.9))
    for i in range(10):
        rows.append((f"question:{i}", "tag:x" if i < 5 else "tag:y", "QUESTION_HAS_TAG", 1.0))
    for i in range(10, 20):
        rows.append((f"question:{i}", "tag:z", "QUESTION_HAS_TAG", 1.0))
    edges = pd.DataFrame(rows, columns=["src", "dst", "edge_type", "weight"]).assign(known_at=T0, evidence="")
    labels = A.communities(A.question_projection(nodes, edges), seed=0, min_size=5)
    assert labels.nunique() == 2
    br = A.bridging_communities(labels, edges, "tag", min_share=0.3)
    assert len(br) == 1 and {br.loc[0, "value_a"], br.loc[0, "value_b"]} == {"tag:x", "tag:y"}
    assert A.component_summary(A.question_projection(nodes, edges))["n_components"] == 2


def test_spike_edge_weight_is_minus_log10_q():
    _, edges = build()
    w = edges.loc[edges["edge_type"] == "ISSUE_SPIKED_DURING_WINDOW", "weight"].iloc[0]
    assert w == pytest.approx(3.0)
