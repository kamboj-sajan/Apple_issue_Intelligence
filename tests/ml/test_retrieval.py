"""Retrieval and ranking-metric tests on tiny SYNTHETIC corpora."""

import numpy as np
import pandas as pd
import pytest
from rank_bm25 import BM25Okapi

from ml.evaluation.ranking import bootstrap_ci, query_metrics, ranks_of_relevant
from ml.retrieval.bm25 import BM25, tokenize
from ml.retrieval.related import RetrievalIndex, build_eval_set, evaluate

DOCS = [
    "wifi keeps disconnecting on macbook pro after sonoma update",
    "battery drains fast on iphone 15",
    "time machine backup fails with error -36",
    "macbook pro wifi drops every few minutes",
    "external display flickers on m1 mac",
    "iphone battery health shows service",
]


def test_bm25_matches_rank_bm25_reference():
    toks = [tokenize(d) for d in DOCS]
    ref = BM25Okapi(toks)
    ours = BM25().fit(toks).index(toks)
    for query in (["wifi", "macbook"], ["battery", "iphone", "iphone"], ["nothing-here"]):
        assert np.allclose(ours.scores(query), ref.get_scores(query))


def test_bm25_fit_on_subset_scores_unseen_docs_without_refitting():
    toks = [tokenize(d) for d in DOCS]
    bm = BM25().fit(toks[:3]).index(toks)
    assert bm.scores(["wifi"])[3] > 0  # doc 3 is not in the fit corpus but is indexed
    assert "minutes" not in bm.vocab  # vocabulary only from the fit corpus


def test_term_contributions_explain_score():
    toks = [tokenize(d) for d in DOCS]
    bm = BM25().fit(toks).index(toks)
    contrib = bm.term_contributions(toks[0], 3)
    assert {t for t, _ in contrib} >= {"wifi", "macbook"}
    assert sum(w for _, w in contrib) == pytest.approx(bm.scores(toks[0])[3])


def test_ranks_and_metrics_handle_ties_fairly():
    assert list(ranks_of_relevant(np.array([0.9, 0.5, 0.5, 0.1]), [1])) == [2.5]
    assert list(ranks_of_relevant(np.zeros(5), [0])) == [3.0]  # constant scores get the middle rank
    m = query_metrics(np.array([2.0, 7.0]), ks=(1, 5, 10))
    assert m["rr"] == 0.5 and m["hits@1"] == 0 and m["hits@5"] == 1
    assert m["recall@5"] == 0.5 and m["precision@5"] == pytest.approx(0.2) and m["recall@10"] == 1


def test_bootstrap_ci_contains_mean_and_is_deterministic():
    v = np.random.default_rng(0).random(200)
    lo, hi = bootstrap_ci(v, seed=1)
    assert lo < v.mean() < hi
    assert bootstrap_ci(v, seed=1) == (lo, hi)


def _corpus():
    t0 = pd.Timestamp("2024-01-01", tz="UTC")
    q = pd.DataFrame(
        {
            "question_id": [10, 11, 12, 13, 14, 15],
            "title": [d.split(" on ")[0] for d in DOCS],
            "body_text": DOCS,
            "tags": [["wifi"], ["battery"], ["time-machine"], ["wifi", "macbook-pro"], ["display"], ["battery"]],
            "created_at": [t0 + pd.Timedelta(days=i) for i in range(6)],
            "link": [f"https://apple.stackexchange.com/questions/{i}" for i in range(10, 16)],
        }
    )
    splits = pd.DataFrame(
        {"question_id": q["question_id"], "split": ["train"] * 3 + ["test"] * 3, "in_research_window": True}
    )
    links = pd.DataFrame(
        {"source_id": [13, 15, 11], "target_id": [10, 11, 15], "link_type": ["duplicate", "body_link", "body_link"]}
    )
    emb = np.random.default_rng(0).normal(size=(6, 8)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    ent = pd.DataFrame(
        {"question_id": [10, 13], "entity_type": ["device", "device"], "value": ["macbook-pro", "macbook-pro"]}
    )
    return q, splits, links, emb, ent


def test_eval_set_uses_only_earlier_counterparts_and_test_queries():
    q, splits, links, emb, ent = _corpus()
    idx = RetrievalIndex(q, ent, emb, q["question_id"].to_numpy(), bm25_fit_before=q["created_at"][3])
    es = build_eval_set(idx, links, splits)
    # 13->10 (earlier) and 15->11 (earlier) are valid; 11->15 is a train query and is excluded.
    assert sorted(es.queries["query_id"]) == [13, 15]
    row = es.queries.set_index("query_id").loc[13]
    assert row["relevant_pos"] == [0] and row["relevant_types"] == ["duplicate"]


def test_evaluate_and_explain():
    q, splits, links, emb, ent = _corpus()
    idx = RetrievalIndex(q, ent, emb, q["question_id"].to_numpy(), bm25_fit_before=q["created_at"][3])
    per_query = evaluate(idx, build_eval_set(idx, links, splits))
    bm = per_query[(per_query["method"] == "bm25")].set_index("query_id")
    assert bm.loc[13, "rr"] == 1.0  # wifi/macbook question finds its duplicate first
    assert (per_query["n_candidates"] == per_query["query_id"].map({13: 3, 15: 5})).all()
    rel = idx.related(13, "bm25", k=2, only_earlier=True)
    assert rel[0]["question_id"] == 10
    ex = rel[0]["explanation"]
    assert ex["shared_tags"] == ["wifi"] and ex["shared_entities"] == {"device": ["macbook-pro"]}
    assert ex["bm25_top_terms"][0]["weight"] > 0 and ex["embedding_cosine"] is not None


def test_clusters_fit_only_on_mask_and_describe_with_distinct_terms():
    from ml.preprocessing.clusters import describe_clusters, fit_clusters

    rng = np.random.default_rng(0)
    a = rng.normal([5, 0], 0.1, (30, 2))
    b = rng.normal([-5, 0], 0.1, (30, 2))
    c = rng.normal([0, 5], 0.1, (30, 2))  # only appears outside the fit mask
    emb = np.vstack([a, b, c])
    mask = np.r_[np.ones(60, bool), np.zeros(30, bool)]
    labels, km = fit_clusters(emb, mask, k=2, seed=0)
    assert len(set(labels[:30])) == 1 and len(set(labels[30:60])) == 1 and labels[0] != labels[30]
    assert km.cluster_centers_[:, 1].max() < 1  # unseen group 'c' did not shape the centroids
    titles = pd.Series(["wifi drops network"] * 30 + ["battery drain charging"] * 30)
    terms = describe_clusters(titles, labels[:60])
    assert "wifi" in terms[labels[0]] and "battery" in terms[labels[30]]
