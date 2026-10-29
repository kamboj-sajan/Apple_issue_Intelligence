# GraphSAGE Link Prediction on Ask Different

Code: `ml/models/` (`link_data.py`, `graphsage_link.py`), `ml/experiments/link_prediction.py`,
`ml/evaluation/link_eval.py`, `ml/retrieval/graph_heuristic.py`. Command: `aii link-predict`.
Snapshot `9e8dce73e94a`. All numbers below come from the run written to
`data/artifacts/link_prediction/9e8dce73e94a/` (report, per-query metrics, pair scores, plot).

**Bottom line:** on ranking (MRR), GraphSAGE is **not** significantly better than plain embedding
cosine. On pairwise classification (AP, ROC-AUC) it is significantly better, but most of that gain
comes from supervised training of the scoring function, not from message passing over the graph.

## Task and labels

* **Question:** given a new question, which *earlier* questions are related to it?
* **Positive pairs:** community duplicate closures (strongest label) and hyperlinks from a question
  body to another question. Oriented query = later question, target = earlier question.
* **Unlinked pairs are not treated as true negatives.** Negatives are *sampled* (below), and
  classification metrics are lower bounds because some sampled negatives may be unlabelled
  duplicates.
* **Label text leak fixed before any experiment:** the anchor text of question links (usually the
  linked question's title) is replaced with `[linked question]` (pipeline 0.2.0).

## Splits and leakage controls

| | Rule | Count |
|-|------|-------|
| Train positives | query in train split (created before 2022-06-27) and label known before then | 3,257 pairs (1,477 duplicates) |
| Validation queries | val split (2022-06-27 to 2023-08-13), ≥ 1 earlier positive | 714 |
| Test queries | test split (from 2023-08-14), ≥ 1 earlier positive | 603 (296 with a duplicate label), 748 positive pairs |

* **Message-passing graph:** feature edges only (tags, devices, OS versions, errors, technical
  entities, categories, semantic kNN to earlier questions, temporal proximity). 46,487 nodes and
  1,025,134 directed edges; the training graph has the 701,396 edges known before the validation boundary.
  **Label edges are never used for message passing**, and week nodes and spike edges are excluded.
* **Strict as-of test inference:** each test query is embedded with the graph **exactly as of its own
  creation time**, so neither the query nor any candidate sees later edges. A unit test adds a
  future edge and checks that earlier embeddings don't change.
* **Bug found during development:** pandas stored timestamps in microseconds, and `astype("int64")`
  returned microseconds where nanoseconds were expected. Every as-of filter therefore let **all**
  edges through, including future ones. This was caught by a unit test before any reported run, fixed
  with an explicit `to_ns()`, and covered by a regression test.
* **Candidates:** every question created before the query (36k–45k). All methods rank the
  identical candidate set and score identical pairs.
* BM25 IDF is fit on questions before the boundary that applies (validation or test).

## Model

* **Features:** every node has a 384-d MiniLM embedding: question title + body for questions, and a
  short text such as `"os: macos 14"` for entity nodes. Entities seen for the first time can therefore
  be embedded too (inductive).
* **Encoder:** `h0 = Linear(x) + type_embedding`, then K mean-aggregation `SAGEConv` layers (PyG
  2.8.1) with ReLU and dropout 0.2. The representation is `[h0 ‖ h_K]`.
* **Decoder:** `MLP([z_q ⊙ z_c, |z_q − z_c|])` or `a·cos(z_q, z_c) + b`. With the *text residual*,
  `+ λ·cos(x_q, x_c)` on the raw embeddings, with λ learned.
* **Training:** full-batch on the training graph; Adam (lr 3e-3, weight decay 1e-5); up to 40 epochs;
  early stopping on validation MRR (patience 3 evaluations, every 2 epochs).
* **Negatives per positive:** 4 random earlier questions + 4 *hard* ones (BM25 top-50 for the query,
  a different primary category, embedding cosine < 0.8 to avoid unlabelled duplicates, never a
  labelled pair). The listwise variants use 32 random negatives.
* **New questions at inference:** a question added at time *t* gets its text embedding and the edges
  that exist at *t* (its entities, and similarity/temporal edges to earlier questions). It is embedded
  with the trained weights, without retraining.

## Model selection (validation, seed 0)

Validation embeds each query with the graph as of the end of its week. That is faster than strict
per-query time and is used only for selection.

| Configuration | Message passing | Val MRR |
|---------------|-----------------|---------|
| `sage2_mlp_bce` (original design) | 2 layers | 0.306 |
| `text_only_mlp_bce` (ablation) | none | 0.315 |
| `sage2_mlp_listwise` | 2 layers | 0.313 |
| **`sage2_mlp_bce_residual`** (selected) | 2 layers | **0.435** |
| `sage2_dot_listwise_residual` | 2 layers | 0.417 |
| `sage1_dot_listwise_residual` | 1 layer | 0.419 |
| `text_only_dot_listwise_residual` (ablation) | none | 0.419 |
| *Embedding cosine (reference)* | – | *0.421* |
| *BM25 (reference)* | – | *0.342* |

The original design was far *below* the embedding baseline. Training loss fell to about 0.003 while
validation MRR plateaued around 0.31: the learned 128-d projection overfits 3,257 positives and throws
away fine-grained text similarity. A listwise loss didn't fix this. Adding the raw-text residual did.
Ablation controls are never eligible for selection.

## Test results (603 queries; GraphSAGE = selected config, 3 seeds)

Ranking (bootstrap 95 % CIs over queries):

| Stratum | Method | MRR | Hits@1 | Hits@10 | Recall@10 |
|---------|--------|-----|--------|---------|-----------|
| All queries | BM25 | 0.341 [0.310, 0.372] | 0.247 | 0.536 | 0.495 |
| | Jaccard (entity neighbourhoods) | 0.084 [0.067, 0.104] | 0.045 | 0.148 | 0.135 |
| | Embedding cosine | 0.435 [0.403, 0.469] | 0.323 | 0.663 | 0.626 |
| | No message passing (ablation) | 0.436 [0.405, 0.468] | 0.320 | 0.668 | 0.633 |
| | **GraphSAGE (mean of 3 seeds)** | **0.441 [0.411, 0.475]** | 0.323 | 0.664 | 0.628 |
| Duplicate-labelled | BM25 | 0.306 [0.264, 0.347] | 0.226 | 0.476 | 0.433 |
| | Jaccard | 0.087 [0.062, 0.113] | 0.047 | 0.155 | 0.136 |
| | Embedding cosine | 0.408 [0.364, 0.448] | 0.284 | 0.666 | 0.618 |
| | No message passing (ablation) | 0.405 [0.362, 0.449] | 0.277 | 0.666 | 0.624 |
| | **GraphSAGE (mean of 3 seeds)** | **0.412 [0.370, 0.454]** | 0.277 | 0.659 | 0.618 |

GraphSAGE across seeds: MRR 0.444 / 0.441 / 0.439 (mean 0.441, sd 0.003).

Paired differences in MRR (same queries, 2,000 bootstrap resamples):

| Comparison | All queries | Duplicate-labelled |
|------------|-------------|--------------------|
| GraphSAGE (seed mean) − embedding | +0.006 [−0.005, +0.017] | +0.004 [−0.012, +0.021] |
| GraphSAGE (seed 0) − no message passing | +0.008 [−0.004, +0.020] | +0.009 [−0.007, +0.024] |
| GraphSAGE (seed mean) − BM25 | +0.100 [+0.070, +0.131] | +0.106 [+0.066, +0.150] |

Pairwise classification (each query's positives vs. the same 5 random + 5 hard sampled negatives; lower bounds):

| Method | ROC-AUC (all pairs) | Average precision (all pairs) |
|--------|---------------------|-------------------------------|
| BM25 | 0.714 [0.699, 0.728] | 0.201 [0.184, 0.227] |
| Jaccard | 0.822 [0.807, 0.838] | 0.363 [0.334, 0.394] |
| Embedding cosine | 0.883 [0.873, 0.893] | 0.494 [0.458, 0.527] |
| No message passing (ablation) | 0.896 [0.886, 0.906] | 0.547 [0.515, 0.581] |
| GraphSAGE (seeds 0/1/2) | 0.899 / 0.899 / 0.901 | 0.558 / 0.560 / 0.563 |

Paired AP differences (seed 0): **vs embedding +0.064 [0.050, 0.078]**; **vs no message passing
+0.012 [0.005, 0.017]**.

![MRR comparison](../data/artifacts/link_prediction/9e8dce73e94a/mrr_comparison.png)
*(The plot is generated locally by `aii link-predict`; `data/` isn't committed.)*

## Interpretation

1. **Ranking:** GraphSAGE's MRR is within noise of embedding cosine. The 95 % CI of the paired
   difference includes 0, in both strata. Hypothesis H3 ("GraphSAGE ranks related questions better
   than the baselines") is **not supported** for ranking.
2. **Pair classification:** GraphSAGE separates true pairs from hard and random negatives significantly
   better than cosine (+0.064 AP). Supervision does most of the work: the no-message-passing model,
   trained the same way, already gains +0.053. Message passing adds a small but significant +0.012 AP.
3. **Lexical vs semantic:** every embedding-based method beats BM25 by about 0.10 MRR; duplicates are
   often phrased differently.
4. **The graph heuristic is weak** (MRR 0.08). Shared tags and entities are too generic: thousands of
   questions share `macos` + `macbook-pro`.

### Why the graph adds so little here

* **The graph is largely made of the same information.** Semantic edges are the embedding kNN, so
  aggregating them re-averages what the text embedding already says.
* **Entity and tag edges are generic hubs.** Mean aggregation over nodes with thousands of neighbours
  (e.g. `tag:macos` with ~18k questions, 13 categories) blurs question identity. That is the
  over-smoothing seen in the variants without the text residual.
* **Few labels:** 3,257 training pairs for a 46k-node graph; the model overfits within ~12 epochs.
* **Sparse new questions:** later questions have few graph neighbours at their creation time (the
  site's volume fell about 6×), so the graph has little extra to add for test queries.

### What would be worth trying next

* Relation-aware aggregation (R-GCN / heterogeneous SAGE) with degree-normalised or attention
  weights, so hub edges don't dominate.
* Edges that aren't derived from the same embedding, e.g. accepted-answer similarity or
  co-answering users (would need more API data and a privacy review).
* More labels: closure history and "linked questions" from answers, not just question bodies.
* A cross-encoder re-ranker on the embedding top-k, which is a stronger text baseline than the graph.

## Reproducibility

* Seeds: 0, 1, 2 (final models); 0 (selection, ablation); 12345 (test negatives).
* Environment: Python 3.12.15, torch 2.15.0, torch-geometric 2.8.1, sentence-transformers 6.2.0,
  macOS arm64 CPU. Recorded in `report.json`.
* Checkpoints with their configs: `data/artifacts/models/9e8dce73e94a/graphsage_seed{0,1,2}.pt`.
* MLflow (SQLite backend `data/mlflow.db`, experiment `ask-different-link-prediction`;
  `mlflow ui --backend-store-uri sqlite:///data/mlflow.db`). **Note:** during the reported run, MLflow
  3.17 refused the old `./mlruns` file store and the code fell back to file outputs, so nothing was
  logged live. The backend has since been switched to SQLite. The reported run's selection and final
  runs were then logged from `report.json` with `backfill_mlflow()` and are tagged
  `source=backfill_from_report_json`. New runs log live (params, per-epoch loss, validation MRR).
* Full run: about 68 minutes on a laptop CPU (7 selection configs, 3 seeds, ablation, strict test evaluation).

```bash
aii embed && aii graph          # prerequisites for the snapshot
aii link-predict                # selection + 3 seeds + ablation + evaluation
aii link-predict --grid-size 1 --seeds 1   # quick single-config run
```
