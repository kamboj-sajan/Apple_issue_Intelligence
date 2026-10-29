# Milestone Log

Numbers below come from actual runs on snapshot `3d642dc498c8` (collected 2026-10-29). Re-running the
pipeline later will give a new snapshot and slightly different numbers.

## Milestone 1: Research design (done)

* [research_design.md](research_design.md): hypotheses H1–H5, scope, split and leakage rules,
  evaluation protocols, GraphSAGE PPI reproduction plan (Table 1 numbers checked against the paper).
* [data_policy.md](data_policy.md): licensing, attribution, API etiquette, privacy.

## Milestone 2: Data pipeline (done)

| Component | File |
|-----------|------|
| API client (spacing, `backoff`, retries, throttle/quota) | `ml/data/stackexchange_client.py` |
| Checkpointed creation-chunk, activity-update and by-id ingestion | `ml/data/ingest.py` |
| Verbatim raw page store | `ml/data/raw_store.py` |
| HTML cleanup / normalisation / link extraction | `ml/preprocessing/text.py` |
| Rule-based entities and categories | `ml/preprocessing/entities.py`, `configs/entities.yaml` |
| Dedup, UTC timestamps, quality report, links table | `ml/preprocessing/normalize.py` |
| Week-aligned temporal splits | `ml/preprocessing/splits.py` |

### Collection

Collected over three runs: two anonymous runs that stopped at the 300/day quota and resumed from
the checkpoint, then one run with an app key. No retries or API errors occurred.

| Item | Value |
|------|-------|
| Raw pages | 505 |
| Questions in research window | 42,555 (2019-01-01 → 2026-10-27) |
| Context questions (linked targets fetched by id) | 2,541 |
| Closed as duplicate | 2,140 questions → 2,452 duplicate links |
| Body links to other questions | 3,014 |
| Link targets present in corpus | 5,287 of 5,466 (97 %) |
| Entity matches | category 69,430 · tech 59,023 · device 54,676 · os 26,950 · error 970 |
| Uncategorised (no category rule fired) | 9,418 (20.9 %) |
| Split (research window) | train 29,741 · val 6,415 (from 2022-06-27) · test 6,399 (from 2023-08-14) |

### Fixes made after inspecting real data

* A bare 8-hex-digit "error code" rule matched memory addresses and register dumps in crash logs.
  It was removed, and uninformative codes (`0`, `1`, `-1`, all-zero hex) are ignored. The most
  frequent codes are now real ones (diskutil `-69743`, `-69877`, `NSPOSIXErrorDomain`).
* Initially only 25 % of duplicate links pointed to a collected question. `aii fetch-linked` now fetches
  targets by id. They are flagged `in_research_window = False`, so they act as link-prediction context
  without inflating volumes or shifting split boundaries.

### Known limitations

* The activity update ran right after collection and found 0 changed questions. Later edits and
  duplicate closures need periodic `aii update` runs.
* Body-link timing is conservative (last edit or creation), because the API doesn't say when a link was added.
* Entity rules favour recall over precision (e.g. `terminal` matches the plain English word).
* `content_license` is missing for 3,097 questions (the API omitted it).
* Deleted questions vanish from pagination, so a resumed chunk can skip an item if page boundaries shift.

## Milestone 3: Anomaly detection and retrieval baselines (done)

| Component | File |
|-----------|------|
| Issue groups, weekly counts, site totals, partial weeks | `ml/anomaly_detection/aggregate.py` |
| Rolling robust z, negative binomial, PELT change points, BH | `ml/anomaly_detection/detectors.py` |
| Screening, quality flags, spike/sustained, evidence, ranking | `ml/anomaly_detection/pipeline.py` |
| Injection study and null calibration | `ml/anomaly_detection/evaluation.py` |
| Semantic clusters (k-means on train embeddings, c-TF-IDF terms) | `ml/preprocessing/clusters.py` |
| BM25 (sparse; equal to `rank_bm25`, fit on pre-test questions) | `ml/retrieval/bm25.py` |
| Sentence-transformer embeddings (local, cached) | `ml/retrieval/embeddings.py` |
| Related-question index, explanations, evaluation | `ml/retrieval/related.py` |
| Ranking metrics with exact ranks and bootstrap CIs | `ml/evaluation/ranking.py` |

### Anomaly detection on real data

1,485 issue groups (tags, tag pairs, OS versions, devices, error codes, technical entities,
categories and 40 semantic clusters) × 409 weeks. Every detector models a group's **share** of
site-wide weekly volume, because Ask Different's total volume falls about 6× between 2019 and 2025.

* **Negative binomial, FDR q ≤ 0.05:** 25 signals (21 spikes, 4 sustained). The rolling detector
  alone flags 230 cells (uncorrected); PELT finds 160 change points.
* The strongest signals line up with documented OS release weeks: macOS Catalina (week of 2019-10-07:
  124 questions vs 6.8 expected), macOS Big Sur (2020-11-09), and the September iOS releases in 2019,
  2020 and 2023. Other signals include Boot Camp / Windows in mid-March 2020 and a Homebrew/Python
  cluster spike in August 2022. These are plausibility checks, not ground truth.
* Annual release cycles are **not** modelled per group, so release weeks show up as signals. That is
  intended (they are real surges in discussion), but it means release-week signals aren't surprising.

### Detector evaluation (400 injection trials into real series; seed 13)

Recall of injected anomalies:

| Injection | Negative binomial (FDR) | Rolling robust z | Change point |
|-----------|-------------------|------------------|--------------|
| Spike ×2 | 0.15 | 0.31 | 0.03 |
| Spike ×3 | 0.29 | 0.43 | 0.02 |
| Spike ×5 | 0.64 | 0.66 | 0.12 |
| Shift ×1.5 (8 wk) | 0.04 | 0.14 | 0.07 |
| Shift ×2 | 0.19 | 0.40 | 0.22 |
| Shift ×3 | 0.45 | 0.60 | 0.65 |

* **Group size dominates.** Below 5 questions/week the negative binomial catches 10–14 % of injected
  spikes; at 5–15/week it catches 60 %, and above 15/week 70 %. With FDR across ~1,500 groups, a few
  extra questions in a small group can't be told apart from noise.
* **Spike vs sustained classification** of detected injections: 81–95 % correct for spikes, 58–89 % for shifts.
* **Null calibration** (Poisson-simulated series, nominal p < 0.05): the negative binomial's
  false-positive rate is **3.1 %**, so it is conservative (the discrete tail test). The rolling z's is **6.7 %**,
  so its normal approximation flags too often. That is why the negative binomial with FDR drives
  the ranked signals and the rolling detector is only supporting evidence.
* The change-point detector uses the whole series, so it only characterises signals and never raises alerts.

### Retrieval baselines (test split)

603 test queries (296 with a duplicate-closure label). Candidates are every question created before
the query (~36k–45k). BM25 IDF is fit only on questions before 2023-08-14. Values are means with
bootstrap 95 % CIs.

| Stratum | Metric | BM25 | Embedding (MiniLM-L6-v2) |
|---------|--------|------|-----------|
| All queries | MRR | 0.438 [0.406, 0.472] | 0.494 [0.461, 0.529] |
| | Hits@10 | 0.622 [0.584, 0.660] | 0.703 [0.668, 0.740] |
| | Recall@10 | 0.584 [0.548, 0.623] | 0.669 [0.634, 0.707] |
| Duplicate-labelled | MRR | 0.324 [0.278, 0.365] | 0.425 [0.379, 0.469] |
| | Hits@10 | 0.497 [0.439, 0.547] | 0.669 [0.615, 0.716] |
| | Recall@10 | 0.457 [0.402, 0.509] | 0.626 [0.572, 0.675] |

Embeddings beat BM25 on duplicates, with non-overlapping CIs, because duplicates are often phrased
differently. On all queries the CIs overlap slightly; a paired comparison comes in Milestone 5.
Precision@10 is low by construction, since most queries have only one or two labelled positives.

### Commands

```bash
aii embed && aii clusters --k 40
aii anomalies                 # signals, detections, change points, series -> data/artifacts/anomalies/<snapshot>/
aii anomaly-eval --trials 400 # injection study + null calibration
aii retrieval-eval            # BM25 vs embedding on the test split
```

### Tests

52 passed. Anomaly tests use synthetic series with known structure: share model vs site decline,
past-only baselines, sparse-group handling, PELT shift vs spike, offset invariance, BH vs statsmodels,
end-to-end ranking, and deterministic evaluation. Retrieval tests check BM25 against `rank_bm25`,
tie-aware ranks, earlier-only candidates, and explanations.

## Milestone 4: Knowledge graph (done)

| Component | File |
|-----------|------|
| Node/edge schema, ids, validation | `ml/graph/schema.py` |
| Construction rules, `as_of` snapshots, NetworkX view | `ml/graph/build.py` |
| Statistics, components, Louvain communities, bridges, neighbours, ego graphs | `ml/graph/analysis.py` |
| End-to-end run | `ml/graph/pipeline.py` (`aii graph`, `aii graph-neighbors`) |

Schema and the reasoning behind each threshold: [graph_schema.md](graph_schema.md).

### Results on snapshot `3d642dc498c8`

* **47,333 nodes**: 45,096 questions, 916 tags, 846 weeks, 308 error codes, 71 OS versions,
  49 devices, 34 technical entities, 13 categories.
* **564,342 edges**, all 12 types, schema validation passed: semantic 167,477 · tag 132,501 ·
  category 69,430 · temporal 54,751 · week 45,096 · device 34,464 · tech 33,540 · OS 20,926 ·
  body-link labels 2,867 · duplicate labels 2,420 · error 856 · spike 14 (cluster and tag-pair signals
  have no node type, so they're not represented).
* **Question degree:** median 15, p90 32, max 145, so the graph is sparse by construction. No isolated nodes.
* **Question-question projection** (semantic + temporal edges): 90.2 % of questions in one giant
  component; 3,755 questions have no semantic or temporal neighbour.
* **39 Louvain communities** (≥ 10 questions), which are coherent: partitions/Boot Camp, startup/SSD,
  browsers/keychain, audio/Bluetooth, Photos/iCloud, displays/Thunderbolt, Wi-Fi, Homebrew/terminal,
  battery/charging, Time Machine, Mail, …
* **Bridges:** an AirTag community linking iPhone and macOS questions; Wi-Fi and audio/Bluetooth
  communities spanning iPhone and MacBook Pro; a Monterey printing community linking macOS 10.15, 11
  and 12.
* **Snapshots:** from 13,430 nodes and 121,662 edges (1 Jan 2020) to 47,333 and 564,342. The giant-component
  share grows from 77 % to 90 %, and mean question degree from 13.2 to 17.6.
* **Example:** a December 2025 macOS Tahoe 26.2 question about the Safari sidebar in full screen is
  linked to its duplicate target, question 209114 (a much older question). It also has a
  semantic edge (cosine 0.82) and temporal edges to other Tahoe 26.2 questions from the same week.

### Issues found and fixed

* The spike-edge weight was computed as `-log10(q).clip(...)`, which clips *before* negating, so q = 0.001
  became −0.001. Schema validation (weights must be > 0) caught it; a regression test now covers it.
* Bridge scoring first used Jaccard, which flagged `keyboard + macos` as a bridge only because `macos`
  is very common. It now uses the overlap coefficient.

### Tests

62 passed in total (10 graph tests). The graph tests cover schema validation (dangling edges, wrong endpoints,
edges known before an endpoint), evidence-based weights, earlier-only semantic and temporal edges,
the generic-entity filter, label-edge flagging and exclusion, `as_of` snapshots with no future facts,
neighbours and bounded ego graphs, and community/bridge detection on a constructed graph.

### Known limitations

* Question text is the latest revision, so post-cutoff edits can leak into text features (affects every
  text method equally; noted for Milestone 5).
* Louvain results depend on the seed (seed 0 is recorded). Communities are descriptive, not ground truth.
* 846 week nodes include pre-2019 weeks that come only from context questions.

### Prerequisites for Milestone 5

* PyTorch Geometric is installed (2.8.1). The PPI benchmark dataset downloads on first use.
* Link-prediction graph: `to_networkx(..., include_label_edges=False)` plus training label edges
  with `known_at` before the validation boundary.
