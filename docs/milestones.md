# Milestone Log

Numbers below come from actual runs on snapshot **`9e8dce73e94a`** (pipeline 0.2.0, data collected
2026-10-29). Pipeline 0.1.0 (snapshot `3d642dc498c8`) kept the anchor text of links to other questions;
that leaked link-prediction labels into the text and was fixed in Milestone 5 (see "Retrieval baselines").
Re-running the pipeline later will give a new snapshot and slightly different numbers.

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

1,484 issue groups (tags, tag pairs, OS versions, devices, error codes, technical entities,
categories and 40 semantic clusters) × 409 weeks. Every detector models a group's **share** of
site-wide weekly volume, because Ask Different's total volume falls about 6× between 2019 and 2025.

* **Negative binomial, FDR q ≤ 0.05:** 21 signals (17 spikes, 4 sustained). The rolling detector
  alone flags 304 cells (uncorrected); PELT finds 173 change points.
* The strongest signals line up with documented OS release weeks: macOS Catalina (week of 2019-10-07:
  124 questions vs 6.8 expected), macOS Big Sur (2020-11-09), and the September iOS releases in 2019,
  2020 and 2023. Other signals include Boot Camp / Windows in mid-March 2020 (also as a semantic
  cluster) and a backup-tag spike in August 2022. These are plausibility checks, not ground truth.
* Annual release cycles are **not** modelled per group, so release weeks show up as signals. That is
  intended (they are real surges in discussion), but it means release-week signals aren't surprising.

### Detector evaluation (400 injection trials into real series; seed 13)

Recall of injected anomalies:

| Injection | Negative binomial (FDR) | Rolling robust z | Change point |
|-----------|-------------------|------------------|--------------|
| Spike ×2 | 0.16 | 0.27 | 0.01 |
| Spike ×3 | 0.41 | 0.63 | 0.03 |
| Spike ×5 | 0.68 | 0.68 | 0.08 |
| Shift ×1.5 (8 wk) | 0.06 | 0.17 | 0.07 |
| Shift ×2 | 0.21 | 0.43 | 0.22 |
| Shift ×3 | 0.62 | 0.71 | 0.74 |

* **Group size dominates.** Below 5 questions/week the negative binomial catches 18 % of injected
  spikes; at 5–15/week 64 %, and above 15/week 86 %. With FDR across ~1,500 groups, a few
  extra questions in a small group can't be told apart from noise.
* **Spike vs sustained classification** of detected injections: 84–96 % correct for spikes, 46–95 % for
  shifts (weak shifts are often called spikes).
* **Null calibration** (Poisson-simulated series, nominal p < 0.05): the negative binomial's
  false-positive rate is **3.1 %**, so it is conservative (the discrete tail test). The rolling z's is **6.8 %**,
  so its normal approximation flags too often. That is why the negative binomial with FDR drives
  the ranked signals and the rolling detector is only supporting evidence.
* The change-point detector uses the whole series, so it only characterises signals and never raises alerts.

### Retrieval baselines (test split)

603 test queries (296 with a duplicate-closure label). Candidates are every question created before
the query (~36k–45k). BM25 IDF is fit only on questions before 2023-08-14. Values are means with
bootstrap 95 % CIs.

| Stratum | Metric | BM25 | Embedding (MiniLM-L6-v2) |
|---------|--------|------|-----------|
| All queries | MRR | 0.341 [0.310, 0.372] | 0.435 [0.403, 0.469] |
| | Hits@10 | 0.536 [0.498, 0.572] | 0.663 [0.627, 0.701] |
| | Recall@10 | 0.495 [0.459, 0.533] | 0.626 [0.589, 0.663] |
| Duplicate-labelled | MRR | 0.306 [0.264, 0.347] | 0.408 [0.364, 0.448] |
| | Hits@10 | 0.476 [0.419, 0.527] | 0.666 [0.611, 0.713] |
| | Recall@10 | 0.433 [0.379, 0.484] | 0.618 [0.564, 0.668] |

Embeddings beat BM25 in both strata with non-overlapping CIs. Precision@10 is low by construction,
since most queries have only one or two labelled positives.

**Correction (found in Milestone 5).** The first version of this table (pipeline 0.1.0) reported
MRR 0.438 (BM25) and 0.494 (embedding) on all queries. Those numbers were inflated: a question that
links to another question usually uses the linked question's **title** as the link text, and those
links are the evaluation labels. Pipeline 0.2.0 replaces such links with `[linked question]`. MRR
fell by 0.06–0.10, mostly on body-link queries, which confirms the leak. Duplicate-labelled queries
changed less (0.324 → 0.306 for BM25, 0.425 → 0.408 for embeddings).

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

### Results on snapshot `9e8dce73e94a`

* **47,333 nodes**: 45,096 questions, 916 tags, 846 weeks, 308 error codes, 71 OS versions,
  49 devices, 34 technical entities, 13 categories.
* **562,964 edges**, all 12 types, schema validation passed: semantic 166,644 · tag 132,501 ·
  category 69,322 · temporal 54,619 · week 45,096 · device 34,362 · tech 33,476 · OS 20,791 ·
  body-link labels 2,867 · duplicate labels 2,420 · error 852 · spike 14 (cluster and tag-pair signals
  have no node type, so they're not represented).
* **Question degree:** median 15, p90 32, max 149, so the graph is sparse by construction. No isolated nodes.
* **Question-question projection** (semantic + temporal edges): 90.2 % of questions in one giant
  component; 3,787 questions have no semantic or temporal neighbour.
* **36 Louvain communities** (≥ 10 questions), which are coherent: partitions/Boot Camp, startup/SSD,
  Homebrew/terminal, browsers/keychain, displays/Thunderbolt, Photos/iCloud, keyboard/trackpad,
  Wi-Fi, audio/Bluetooth, battery/charging, Time Machine, …
* **Bridges:** Wi-Fi and audio/Bluetooth communities spanning iPhone and MacBook Pro; Photos/iCloud
  linking iPhone and macOS questions; AppleScript automation linking Finder questions; communities
  connecting macOS 10.15 and 13, and 12 and 13. (An AirTag bridge seen on the 0.1.0 snapshot didn't
  survive the rebuild. Louvain communities shift between runs, so individual bridges are descriptive.)
* **Snapshots:** from 13,430 nodes and 121,436 edges (1 Jan 2020) to 47,333 and 562,964. The giant-component
  share grows from 77 % to 90 %, and mean question degree from 13.2 to 17.5.
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

## Milestone 5: GraphSAGE benchmark reproduction and link prediction (done)

| Component | File |
|-----------|------|
| PPI reproduction (official sampler, concat mean aggregator) | `ml/experiments/ppi_reproduction.py` |
| Link-prediction tensors, time filtering, label pairs | `ml/models/link_data.py` |
| GraphSAGE link model, hard negatives, training, strict as-of scoring | `ml/models/graphsage_link.py` |
| Selection grid, ablation, unified evaluation, MLflow, checkpoints | `ml/experiments/link_prediction.py` |
| Jaccard graph heuristic | `ml/retrieval/graph_heuristic.py` |
| Ranking/classification metrics, paired bootstrap | `ml/evaluation/link_eval.py` |

### Benchmark reproduction ([benchmark_reproduction.md](benchmark_reproduction.md))

PPI supervised micro-F1, GraphSAGE-mean: paper **0.598**. Ours, faithful to the official code:
**0.580 ± 0.018** (5 seeds, 10 epochs). 50 epochs: 0.631 ± 0.012. PyG `SAGEConv` full-batch: 0.657 ± 0.023.

### Ask Different link prediction ([link_prediction.md](link_prediction.md))

603 test queries, all candidates earlier than the query, strict per-query as-of graph.

| Method | MRR (all) | MRR (duplicates) | AP (sampled pairs) |
|--------|-----------|------------------|--------------------|
| BM25 | 0.341 | 0.306 | 0.201 |
| Jaccard (entities) | 0.084 | 0.087 | 0.363 |
| Embedding cosine | 0.435 | 0.408 | 0.494 |
| GraphSAGE without message passing (ablation) | 0.436 | 0.405 | 0.547 |
| **GraphSAGE** (3 seeds) | **0.441 ± 0.003** | **0.412** | **0.558–0.563** |

* **Ranking:** GraphSAGE − embedding = +0.006 MRR [−0.005, +0.017]. **Not significant**, so H3 is not
  supported for ranking.
* **Pair classification:** GraphSAGE − embedding = +0.064 AP [0.050, 0.078]. Significant, but supervision
  explains most of it: message passing on its own adds +0.012 AP [0.005, 0.017].
* The original design (no text residual) reached only 0.306 validation MRR, against 0.421 for embeddings.
  An ablation grid on validation showed the learned projection overfits and loses text similarity.
  Adding the raw-text residual brought it level with embeddings.

### Issues found and fixed

* **Label leak in text:** links to other questions kept their anchor text (usually the linked
  question's title), and those links are the labels. Fixed in pipeline 0.2.0. Baseline MRR fell by
  0.06–0.10, and every number in this log now uses the fixed text.
* **Time-unit bug:** pandas microsecond timestamps were compared with nanosecond cutoffs, so as-of
  filtering let future edges through. Caught by a unit test before any reported run; fixed with `to_ns()`.
* **CI:** the link-prediction tests need torch and PyG. CI installs the CPU-only torch wheel, and the
  tests skip cleanly where the ML extras are missing.

### Tests

81 passed in total. Link-prediction tests cover: label edges and week nodes excluded from message
passing; as-of filtering; a future edge not changing earlier embeddings; label-pair orientation;
negatives excluding related pairs; Jaccard values; listwise loss vs manual softmax; identical pairs
across methods; paired bootstrap; and the nanosecond conversion.

### Known limitations

* Question text is the latest revision (no revision history), so post-cutoff edits can leak into text
  features for every method equally.
* Classification metrics use sampled negatives that may include unlabelled duplicates, so they are
  lower bounds.
* Validation used week-level as-of graphs (faster); test used strict per-query time.
* One ablation seed; the GraphSAGE − no-message-passing comparison would be stronger with 3 ablation seeds.

## Milestone 6: Drift monitoring (done)

See [drift_monitoring.md](drift_monitoring.md). In short: monthly windows, equal-size samples,
permutation z-scores and thresholds calibrated on 2019–2022. Alerts line up with release months
(Catalina 2019-10, WWDC 2023-06, "tahoe" 2025-09). Controlled tests: 0–5 % false alarms; 80 % detection
of a 20 % topic surge; 100 % at 40 %. Semantic drift vs volume anomalies: ρ = 0.19 (mostly distinct signals).

## Milestone 7: API and dashboard (done)

| Component | Location |
|-----------|----------|
| Schema (SQLAlchemy 2.0, pgvector `vector(384)` + HNSW cosine index, GIN array indexes) | `backend/app/db/models.py` |
| Migrations (Alembic; creates the `vector` extension) | `migrations/versions/0001_initial_schema.py` |
| Loader: artifacts → PostgreSQL via `COPY` in one transaction; related-question precompute | `backend/app/services/loader.py`, `related_precompute.py` |
| Synthetic demo dataset (labelled `demo` in every response) | `backend/app/services/demo.py` |
| Routes, pagination, validation, structured errors | `backend/app/api/routes.py`, `backend/app/core/errors.py` |
| Background jobs (subprocess per step, worker thread, status in `jobs`) | `backend/app/services/jobs.py` |
| Dashboard (Next.js 15, JavaScript, Tailwind 4, Recharts, React Flow) | `frontend/` |
| Containers | `docker-compose.yml`, `backend/Dockerfile`, `frontend/Dockerfile` |

### Real data in the database (snapshot `9e8dce73e94a`)

45,096 questions (with embeddings), 210,540 entity rows, 21 anomalies, 217,997 weekly series points,
47,333 graph nodes, 562,964 edges, 1,352,715 precomputed related-question rows, 73,196 IDF terms,
700 drift metric rows and 3,330 cluster movements. Loading takes about 3.5 minutes on a laptop, most of
it spent on related-question precomputation. Typical API latency on this data is 5–90 ms.

### Verified

* All endpoints return 200 on real and demo data; invalid input returns structured 422s; unknown ids
  return 404s.
* Every dashboard page was rendered with headless Chrome in dark and light mode and inspected. That
  pass found and fixed node overlap in the graph layout, light-themed graph controls in dark mode,
  bar labels colliding with CI whiskers, and wording errors.
* `docker compose up` brings up db + API + dashboard: the API runs migrations at start, serves the
  real snapshot, and ran a background job (`split`) as a subprocess while `/health` answered in 13 ms.

### Issues found and fixed

* Drift `n` columns were floats with NaN, so `COPY` into integer columns failed. They are now nullable `Int64`.
* Raw SQL with `:param IS NULL` failed in PostgreSQL (untyped parameter). Rewritten with SQLAlchemy expressions.
* `/overview` returned 500 on real data: metric payloads mixed ISO-string and datetime timestamps.
  "Latest analysis" now parses each artifact's own timestamp; regression test added.

### Tests

100 passed locally (81 pipeline/ML + 19 API). The API tests run against a dedicated `<db>_test`
database: Alembic upgrade, downgrade to base and upgrade again; pagination and filters;
validation errors; 404s; anomaly explanations; bounded graph neighbourhoods; related-question
evidence (earlier questions only); drift filters; and the full job lifecycle with a real
subprocess. CI now runs them against a `pgvector/pgvector:pg16` service container, plus a separate
frontend lint and build job.

### Known limitations

* Related questions are precomputed (top 10 earlier questions per method). GraphSAGE is served as a
  re-ranker of the embedding top 100, an approximation of the evaluated full ranking (documented in the
  API response and in `docs/api.md`).
* Job runner: one job per type, in-process, single API worker; a restart marks running jobs `interrupted`.
* The API image is 2.8 GB because it includes the CPU ML stack, so jobs can rerun the pipeline.
* The dashboard has no authentication. Ports are bound to 127.0.0.1 in Compose; don't expose it publicly
  as is.

