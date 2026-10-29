"""Link-prediction components on small SYNTHETIC graphs (no downloads)."""

import pytest

pytest.importorskip("torch_geometric")  # skipped where the ML extras are not installed

import numpy as np
import pandas as pd
import torch

from ml.evaluation.link_eval import classification_metrics, evaluate_methods, paired_mrr_difference
from ml.graph.build import GraphConfig, build_graph_tables
from ml.models.graphsage_link import (
    GraphSAGELink,
    Scorer,
    TrainConfig,
    build_training_pairs,
    sample_negatives,
    train_model,
)
from ml.models.link_data import build_link_graph, entity_text, label_pairs
from ml.retrieval.graph_heuristic import JaccardHeuristic
from ml.retrieval.related import RetrievalIndex, build_eval_set
from tests.ml.test_graph import T0, corpus


def setup():
    q, ent, links, emb, splits, signals = corpus()
    q = q.assign(
        body_text=[f"wifi macbook problem {i}" if i % 2 else f"battery display issue {i}" for i in range(len(q))]
    )
    splits = splits.assign(in_research_window=True)
    nodes, edges = build_graph_tables(
        q,
        ent,
        links,
        emb,
        q["question_id"].to_numpy(),
        signals,
        splits,
        GraphConfig(knn_k=2, knn_threshold=0.5, generic_max_df=1.0),
    )
    ent_ids = nodes.loc[~nodes["node_type"].isin(["question", "week"]), "node_id"]
    rng = np.random.default_rng(0)
    ent_emb = {n: rng.normal(size=emb.shape[1]).astype(np.float32) for n in ent_ids}
    graph = build_link_graph(nodes, edges, emb, q["question_id"].to_numpy(), ent_emb)
    index = RetrievalIndex(q, ent, emb, q["question_id"].to_numpy())
    return q, ent, links, emb, splits, nodes, edges, graph, index


def test_entity_text():
    assert entity_text("os:macos:14") == "os: macos 14"
    assert entity_text("tag:macos-sonoma") == "tag: macos sonoma"


def test_link_graph_excludes_labels_weeks_and_respects_time():
    q, *_, nodes, edges, graph, _ = setup()
    ids = set(graph.node_ids)
    assert not any(n.startswith("week:") for n in ids)
    pairs = set(zip(graph.node_ids[graph.edge_src], graph.node_ids[graph.edge_dst], strict=True))
    assert ("question:103", "question:100") in pairs and (
        "question:100",
        "question:103",
    ) in pairs  # semantic, both dirs
    dup_edges = edges[edges["edge_type"] == "QUESTION_DUPLICATE_OF"]
    for s, d in zip(dup_edges["src"], dup_edges["dst"], strict=True):
        feature = edges[(edges["src"] == s) & (edges["dst"] == d) & (edges["edge_type"] != "QUESTION_DUPLICATE_OF")]
        if feature.empty:
            assert (s, d) not in pairs
    t = (T0 + pd.Timedelta(days=2, hours=1)).value
    ei = graph.edge_index_as_of(t)
    assert graph.edge_time[graph.edge_time <= t].size == ei.shape[1]
    late = graph.question_node[104]
    assert not ((ei[0] == late) | (ei[1] == late)).any()  # q104 is posted on day 20


def test_label_pairs_orientation_and_duplicate_precedence():
    q, _, links, _, splits, *_ = setup()
    extra = pd.DataFrame(
        {
            "source_id": [100],
            "target_id": [103],
            "link_type": ["body_link"],
            "known_at": [T0 + pd.Timedelta(days=4)],
            "target_in_corpus": [True],
        }
    )
    p = label_pairs(pd.concat([links, extra]), q, splits)
    row = p[(p["query_id"] == 103) & (p["target_id"] == 100)]
    assert len(row) == 1 and row["link_type"].iloc[0] == "duplicate"  # merged, duplicate wins
    assert (p["query_id"] > p["target_id"]).all()  # later question is the query here


def test_negative_sampling_excludes_related_and_prefers_other_categories():
    q, *_, index = setup()
    cat = np.array(["a", "b", "a", "b", "a", "b", "a", "b"])
    rng = np.random.default_rng(0)
    rand, hard = sample_negatives(index, 7, {3, 7}, cat, rng, n_random=3, n_hard=3, max_cos=1.01)
    assert 3 not in rand + hard and all(c < 7 for c in rand + hard)
    assert all(cat[c] != cat[7] for c in hard[: sum(cat[:7] != cat[7])])


def test_jaccard_heuristic_values():
    edges = pd.DataFrame(
        {
            "src": ["question:1", "question:1", "question:2", "question:2", "question:3"],
            "dst": ["tag:a", "tag:b", "tag:a", "tag:c", "tag:d"],
            "edge_type": "QUESTION_HAS_TAG",
        }
    )
    j = JaccardHeuristic(edges, np.array([1, 2, 3]))
    assert j.scores(1) == pytest.approx([1 / 3])
    assert j.scores(2) == pytest.approx([0.0, 0.0])


def test_training_and_strict_scoring_smoke():
    q, ent, links, emb, splits, nodes, edges, graph, index = setup()
    pairs = label_pairs(links, q, splits)
    cfg = TrainConfig(hidden=8, layers=1, epochs=2, eval_every=1, n_random=2, n_hard=1, seed=0)
    rows = build_training_pairs(index, pairs, pairs, cfg, seed=0)
    assert (rows.groupby("query_pos")["label"].max() == 1).all()
    assert set(rows["kind"]) >= {"positive", "random"}
    val = build_eval_set(index, links, splits.assign(split="val"), "val")
    model, hist = train_model(graph, index, rows, int((T0 + pd.Timedelta(days=60)).value), val, cfg)
    assert len(hist) == 2 and all(np.isfinite(h["loss"]) for h in hist)
    scorer = Scorer(model, graph, index)
    s = scorer.scores(5)
    assert s.shape == (5,) and np.isfinite(s).all()
    # Strict as-of: adding an edge from the future must not change embeddings at an earlier time.
    t3 = int(q.loc[3, "created_at"].value)
    z_before = scorer.embeddings(t3).clone()
    z_late_before = scorer.embeddings(t3 + 10**15).clone()
    a, b = graph.question_node[103], graph.question_node[101]
    graph.edge_src = np.append(graph.edge_src, [a, b])
    graph.edge_dst = np.append(graph.edge_dst, [b, a])
    graph.edge_time = np.append(graph.edge_time, [t3 + 10**12, t3 + 10**12])  # ~17 minutes after q103
    fresh = Scorer(model, graph, index)
    assert torch.allclose(z_before, fresh.embeddings(t3))
    assert not torch.allclose(z_late_before, fresh.embeddings(t3 + 10**15))


def test_decoders_are_symmetric_in_shape():
    for dec in ("mlp", "dot"):
        m = GraphSAGELink(16, TrainConfig(hidden=8, decoder=dec))
        z = torch.randn(5, 16)
        assert m.score(z[0], z).shape == (5,)


def test_unified_evaluation_uses_identical_pairs():
    q, ent, links, emb, splits, nodes, edges, graph, index = setup()
    es = build_eval_set(index, links, splits.assign(split="test"), "test")
    scorers = {"emb": lambda p: index.scores("embedding", p)[:p], "zero": lambda p: np.zeros(p)}
    negatives = {r.query_pos: [0, 1] for r in es.queries.itertuples()}
    per_query, pairs = evaluate_methods(es, scorers, negatives)
    a = pairs[pairs["method"] == "emb"][["query_id", "cand_pos", "label"]].reset_index(drop=True)
    b = pairs[pairs["method"] == "zero"][["query_id", "cand_pos", "label"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    d = paired_mrr_difference(per_query, "emb", "zero")
    assert d["mean_diff"] >= 0 and d["n_queries"] == len(es.queries)
    cls = classification_metrics(pairs, n_boot=20)
    assert set(cls["metric"]) == {"roc_auc", "average_precision"}
    assert cls.loc[(cls["method"] == "zero") & (cls["metric"] == "roc_auc"), "mean"].iloc[0] == pytest.approx(0.5)


def test_to_ns_is_nanoseconds_regardless_of_storage_unit():
    from ml.models.link_data import to_ns

    s_us = pd.Series(pd.to_datetime(["2024-01-01T00:00:00Z"])).astype("datetime64[us, UTC]")
    assert to_ns(s_us)[0] == pd.Timestamp("2024-01-01T00:00:00Z").value


def test_listwise_loss_matches_manual_softmax():
    from ml.models.graphsage_link import _listwise_loss

    logits = torch.tensor([2.0, 0.0, 1.0, 0.5, -1.0])
    y = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0])
    block = torch.tensor([0, 0, 0, 1, 1])
    manual = (-torch.log_softmax(logits[:3], 0)[0] - torch.log_softmax(logits[3:], 0)[0]) / 2
    assert torch.allclose(_listwise_loss(logits, y, block), manual)


def test_text_residual_and_zero_layer_ablation_run():
    m = GraphSAGELink(16, TrainConfig(hidden=8, layers=0, text_residual=True))
    x = torch.randn(4, 16)
    z = m.encode(x, torch.zeros(4, dtype=torch.long), torch.empty((2, 0), dtype=torch.long))
    assert m.score(z[0], z, x[0], x).shape == (4,)
