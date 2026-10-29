# Research Design (Milestone 1)

This document fixes the research questions, data scope, evaluation protocol and leakage rules
**before** any model is trained, so that later results cannot quietly reshape the questions.

> All outputs of this project are *signals derived from publicly posted questions*. They are not
> evidence of product defects, of causal relationships, or of the number of affected users.

## 1. Hypotheses

| ID | Hypothesis | Primary evidence | Falsified if |
|----|------------|------------------|--------------|
| H1 | Weekly question volume for some issue groups (OS version, device family, tag, error code, category) shows statistically unusual increases beyond seasonality and the overall site trend. | FDR-controlled anomaly flags from ≥3 detectors (rolling robust z, count model, change-point). | After Benjamini–Hochberg correction, no group exceeds the expected-count model more than the null false-discovery rate. |
| H2 | The detectors can be told apart on *spike* (short, transient) vs *sustained shift* (persistent level change). | Synthetic injection study with known spike/shift shapes; precision/recall per shape. | Change-point detector cannot distinguish a 1-week spike from a ≥4-week level shift better than chance on injected data. |
| H3 | A GraphSAGE link predictor over a question–entity graph ranks truly related questions higher than BM25, embedding cosine and a graph heuristic. | MRR, Hits@K, P@K, R@K, ROC-AUC/AP on a **temporal** test split with identical candidate sets. | GraphSAGE's bootstrap 95 % CI of MRR overlaps or falls below the best baseline. (A negative result will be reported, not hidden.) |
| H4 | The semantic content of questions drifts over time independently of volume. | Embedding-centroid distance, MMD, tag/category Jensen–Shannon divergence with permutation-calibrated thresholds. | Drift statistics between adjacent windows are indistinguishable from within-window permutation nulls. |
| H5 | Volume anomalies and semantic drift are *not* the same signal. | Rank correlation between per-window anomaly counts and drift scores. | Near-perfect correlation (ρ > 0.9), which would suggest one is a proxy for the other. |

## 2. Dataset scope

* **Source:** Ask Different (`site=apple`) via the official Stack Exchange API v2.3, `/questions`
  endpoint, with a custom filter (see `ml/data/stackexchange_client.py`).
* **Observed volume (API `filter=total`, queried 2026-10-29):** ~131.9k questions all-time;
  9,353 created in 2019; 4,938 in 2023; 1,456 in 2025. **Site activity is declining steeply**,
  which is a first-order confounder: every detector must model a non-stationary baseline, and
  raw counts are never compared across years without normalisation.
* **Default research window:** questions created from 2019-01-01 (UTC) to the ingestion date.
  Configurable in `configs/pipeline.yaml`.
* **Fields collected:** id, link, title, HTML body, tags, creation / last-activity / last-edit /
  closed timestamps, closed reason and *duplicate-of* targets, score, view count, answer count,
  accepted-answer id. Author **display name and profile link only**, kept solely for CC BY-SA
  attribution and never used as a model feature. No user ids, reputation, avatars or emails.
* **Raw vs normalised:** every API page is stored verbatim (gzip JSON + request metadata) under
  `data/raw/`. Normalised Parquet tables are always rebuilt from raw pages, never edited in place.

## 3. Definitions

* **Time window:** ISO week, Monday 00:00 UTC (configurable to day/month).
* **Issue group:** a key `(group_type, value)`, e.g. `("os", "macos:14")`, `("device", "macbook-pro")`,
  `("tag", "wifi")`, `("error", "error:-36")`, `("category", "battery-power")`. Extracted by
  transparent, dictionary-driven rules (`configs/entities.yaml`) so every assignment is auditable.
* **Related questions (ground truth for link prediction):**
  1. **Duplicate closures**: question A closed as a duplicate of B by the community. Strongest label.
  2. **Explicit body links**: A's body hyperlinks to question B on Ask Different. Weaker label
     (a link can mean "not like this"); reported as a separate stratum.
  Unlinked pairs are **not** treated as ground-truth negatives (see §5).

## 4. Temporal splits and leakage prevention

Splits are by question **creation time**, with boundaries stored in a split manifest:

```
train:  created_at <  T_val
val:    T_val <= created_at < T_test
test:   T_test <= created_at
```

Default boundaries are quantiles (70 % / 15 % / 15 % of questions) rounded down to a week
boundary, so no ISO week straddles two splits.

Leakage rules, each enforced by a test:

1. **Graph edges used for prediction** at evaluation time *t* may only use information available
   at *t*: node features from text, entity edges, and *only those* duplicate/link edges whose
   creating event (closure date or the linking question's creation) precedes the split boundary.
2. **Label edges** (positives) for val/test are removed from the message-passing graph.
3. Duplicate closures happen *after* creation; closure timestamps (not creation) decide
   which split a label edge belongs to for message-passing purposes.
4. **Anomaly baselines** use strictly past windows (lookback excludes the scored window).
5. **Drift thresholds** are calibrated on windows that precede the evaluated window.
6. Embedding model is a frozen, pretrained sentence-transformer: it is never fine-tuned on test text.
7. Vocabulary/IDF statistics for BM25 are fit on the candidate corpus that would be available at
   query time (train ∪ earlier questions), never on future questions.

## 5. Link-prediction evaluation protocol

* **Query set:** each test-split question with ≥1 positive whose counterpart was created earlier.
* **Candidate set:** all questions created before the query (realistic "find earlier related
  questions" task). Every method ranks the identical candidate set.
* **Ranking metrics:** MRR, Hits@{1,5,10}, Precision@{5,10}, Recall@{5,10}.
* **Classification metrics (ROC-AUC, AP):** on positive pairs vs sampled negatives, where negatives
  are (a) random earlier questions and (b) *hard* negatives: high BM25 overlap but no shared
  duplicate/link evidence and different primary category. Because some "negatives" are unlabelled
  positives, these metrics are a lower bound; this caveat is shown next to every number.
* **Uncertainty:** bootstrap 95 % CI over queries; GraphSAGE also over ≥3 random seeds.
* **Methods:** BM25 (`rank-bm25`), sentence-transformer cosine, Jaccard / common-neighbours on the
  entity graph, GraphSAGE (PyG) with an MLP or dot-product decoder.
* **New nodes:** GraphSAGE is inductive. A test question is attached to the graph through its
  entity/tag edges and kNN-semantic edges to *earlier* questions, then embedded with the trained
  aggregator without retraining.

## 6. Anomaly-detection evaluation

* **Injection study:** take real weekly series, inject known spikes (1 week, ×2–×5) and shifts
  (≥4 weeks, +50–200 %) at random positions; measure detection precision/recall, delay and
  false-positive rate per detector. Seeds are recorded.
* **Historical spot checks:** a small manually reviewed list of known events (e.g. major OS
  releases) documented in `docs/labeled_events.md`, with the review procedure. Releases are used
  as *plausibility checks*, not as ground truth for defects.
* **Multiple testing:** Benjamini–Hochberg FDR (q = 0.05 default) across all groups in a window.
* **Minimum volume:** groups with < `min_history_weeks` of history or expected count below a floor
  are reported as "insufficient data", never scored.

## 7. Drift evaluation

* Metrics: cosine distance between window centroids; RBF-kernel MMD; Jensen–Shannon divergence of
  tag, category and entity distributions; cluster birth/death via matched clustering.
* Thresholds: permutation null (shuffle window labels between the two windows), calibrated on
  past windows, with minimum sample size per window.
* Controlled tests: windows built from the same period (no drift expected) vs windows with a
  synthetic topic injection (drift expected).

## 8. Benchmark reproduction plan (Milestone 5)

* **Paper:** W. L. Hamilton, R. Ying, J. Leskovec. *Inductive Representation Learning on Large
  Graphs.* NeurIPS 2017. arXiv:1706.02216. Official code: <http://snap.stanford.edu/graphsage/>
  (TensorFlow). Reputable reimplementation: PyTorch Geometric `SAGEConv` and its examples.
* **Target experiment:** PPI, inductive multi-label node classification. 24 graphs: 20 train,
  2 val, 2 test; 50 features; 121 labels; K = 2 layers; neighbourhood samples S₁ = 25, S₂ = 10;
  ReLU; Adam.
* **Reported numbers (Table 1, supervised micro-F1, PPI):** GraphSAGE-GCN 0.500, GraphSAGE-mean
  0.598, GraphSAGE-LSTM 0.612, GraphSAGE-pool 0.600. We reproduce **GraphSAGE-mean** (directly
  supported by PyG) and, time permitting, pool.
* **Known deviation risks:** the paper notes (footnote 6) that later work achieves much higher PPI
  scores with tuned training; PyG with full-neighbourhood training or more epochs may therefore
  *exceed* the paper. We will report both a paper-faithful configuration (sampled neighbours,
  limited epochs) and note any setting that differs.
* The benchmark is reported in its own document and is **not** evidence of performance on Ask
  Different.

## 9. Acceptance criteria per milestone

| Milestone | Accepted when |
|-----------|---------------|
| M2 Data | Real ingestion run completes with checkpoints; re-running is idempotent (no duplicate ids); raw and normalised data are separated; split manifest written; tests pass offline. |
| M3 Baselines | Weekly aggregation + ≥3 detectors produce stored anomaly tables from real data; BM25 and embedding retrieval return ranked results with scores. |
| M4 Graph | Heterogeneous graph validates against schema; statistics, components, communities and snapshots saved. |
| M5 GraphSAGE | PPI reproduction documented; link predictor trains with early stopping; results table with CIs vs all baselines on identical test pairs. |
| M6 Drift | Drift metrics with calibrated thresholds; controlled tests pass. |
| M7 API/UI | Endpoints serve persisted results from PostgreSQL; dashboard labels demo vs real data. |
| M8 Validation | Clean-checkout reproduction following README; CI green or failures documented. |
