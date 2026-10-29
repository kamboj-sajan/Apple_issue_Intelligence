# Apple Issue Intelligence

Research-oriented pipeline that analyses public questions from
[Ask Different](https://apple.stackexchange.com/) to surface **emerging discussion patterns**:
unusual increases in question volume, related questions across different terminology, and
semantic drift over time.

> Outputs are signals from publicly posted questions, **not** evidence of product defects,
> causal relationships, or the number of affected users.

## Status

| Milestone | State |
|-----------|-------|
| 1. Research design | Done: [docs/research_design.md](docs/research_design.md) |
| 2. Data pipeline | Done: 42,555 questions (2019-01 → 2026-10) + 2,541 linked context questions |
| 3. Anomaly detection, BM25 and embedding baselines | Done: 21 FDR-controlled signals; embedding MRR 0.435 vs BM25 0.341 |
| 4. Knowledge graph | Done: 47k nodes, 563k typed edges, 36 communities, time snapshots |
| 5. GraphSAGE (benchmark reproduction and link prediction) | Done: PPI 0.580 ± 0.018 (paper 0.598); link MRR 0.441 vs embedding 0.435 (not significant) |
| 6. Drift monitoring | Done: calibrated monthly alerts; 0 % false alarms, 80 % power at a 20 % topic surge |
| 7. API and dashboard | Done: FastAPI + PostgreSQL/pgvector serving real results; Next.js dashboard (6 pages) |
| 8. Final validation | Not started |

Detailed progress and limitations: [docs/milestones.md](docs/milestones.md).

## Quick start

Requires Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate     # or: uv venv --python 3.12 .venv
pip install -e ".[dev,test]"
cp .env.example .env                                   # optional: add STACKEXCHANGE_KEY
pytest -q                                              # tests; API tests need PostgreSQL (skipped otherwise)
```

## Run the dashboard

**With Docker** (PostgreSQL + API + dashboard):

```bash
docker compose up -d --build        # db on 127.0.0.1:5434, API on :8000, dashboard on :3000
docker compose exec api python -m backend.app.cli load-demo   # synthetic demo data, clearly labelled
# or, after running the pipeline (data/ is mounted into the API container):
docker compose exec api python -m backend.app.cli load
```

**Locally** (Python 3.11+, Node 22):

```bash
pip install -e ".[api,ml,dev,test]"
docker compose up -d db                     # or any PostgreSQL 16 with pgvector; set DATABASE_URL
aii-db load                                 # migrate + load the current snapshot (or: aii-db load-demo)
uvicorn backend.app.main:app --port 8000    # API docs at http://localhost:8000/docs
cd frontend && npm ci && npm run dev        # dashboard at http://localhost:3000
```

The dashboard shows a banner on every page saying whether it is displaying **real** results (with
the snapshot id) or the **synthetic demo** dataset. Endpoints, explanations and background-job
limitations: [docs/api.md](docs/api.md).

| Page | What it shows |
|------|---------------|
| Overview | totals, signal count, latest analysis, data freshness, most active categories |
| Emerging issues | weekly observed vs expected volume per issue group; ranked signals with filters; explanation panel with evidence links |
| Knowledge graph | interactive, bounded neighbourhoods (React Flow) with node/edge-type filters and links to questions |
| Related questions | BM25, embedding and GraphSAGE recommendations side by side, each with its evidence |
| Drift monitoring | monthly drift z-scores vs calibrated thresholds, alerts, cluster share changes, new terms, novelty |
| Model evaluation | baselines vs GraphSAGE with CIs, paired tests, selection grid, PPI reproduction, detector evaluation, splits and leakage controls |

## Data pipeline

```bash
aii ingest                 # fetch questions by month since 2019-01-01 (configs/pipeline.yaml); resumable
aii ingest --max-requests 50 --start 2024-01-01   # smaller run
aii update                 # re-fetch questions with new activity (edits, duplicate closures)
aii fetch-linked           # fetch duplicate/linked targets outside the window (context only)
aii preprocess             # raw pages -> data/processed/*.parquet + manifest.json
aii split                  # time-based train/val/test split + split_manifest.json
aii status                 # checkpoint and snapshot summary
```

## Analysis (Milestone 3)

```bash
pip install -e ".[ml]"
aii embed                  # sentence-transformer embeddings (local, cached per snapshot)
aii clusters --k 40        # semantic clusters fit on the train split
aii anomalies              # weekly anomaly signals with FDR control
aii anomaly-eval           # injection study + null calibration
aii retrieval-eval         # BM25 vs embedding related-question retrieval
aii graph                  # build, validate and analyse the knowledge graph
aii graph-neighbors question:485393   # explore a node's typed neighbours
aii drift                  # monthly semantic/categorical drift with calibrated alerts
aii drift-eval             # controlled no-drift / injected-topic tests
aii ppi-reproduce          # GraphSAGE benchmark reproduction (PPI, paper Table 1)
aii link-predict           # GraphSAGE vs BM25 / embedding / Jaccard on related questions
```

Results and caveats: [docs/milestones.md](docs/milestones.md). Graph schema: [docs/graph_schema.md](docs/graph_schema.md). Drift: [docs/drift_monitoring.md](docs/drift_monitoring.md). GraphSAGE: [docs/benchmark_reproduction.md](docs/benchmark_reproduction.md), [docs/link_prediction.md](docs/link_prediction.md).

Without an API key the Stack Exchange quota is **300 requests/day** (100 questions each). The
ingestion stops cleanly at the quota reserve and resumes from its checkpoint on the next run.
With a free key from [stackapps.com](https://stackapps.com/apps/oauth/register), the quota is 10,000 requests/day.

### Data layout (`data/`, git-ignored)

```
data/
├── raw/stackexchange/apple/questions/
│   ├── creation/<YYYY-MM>/page_0001.json.gz      # verbatim API pages + request metadata
│   ├── activity/activity_<run>/page_0001.json.gz # incremental updates
│   └── by_id/ids_<run>/page_0001.json.gz         # linked questions fetched by id
├── state/ingest_apple.json                       # checkpoint: chunk progress, update watermark, run log
└── processed/
    ├── questions.parquet          # one row per question (latest version), UTC timestamps
    ├── question_links.parquet     # duplicate-closure and body-link edges with known_at times
    ├── question_entities.parquet  # devices, OS versions, error codes, tech entities, categories + evidence
    ├── splits.parquet             # question_id -> train/val/test
    ├── manifest.json              # snapshot id, counts, data-quality report
    └── split_manifest.json        # boundaries, counts, id hashes
```

Processed tables are always rebuilt from raw pages; nothing is edited in place.

## Repository layout

```
ml/            data pipeline and ML (runs without the API, database or frontend)
  data/          Stack Exchange client, checkpointed ingestion, raw store
  preprocessing/ HTML cleanup, entity extraction, normalisation, temporal splits, clusters
  anomaly_detection/ weekly aggregation, detectors, FDR screening, evaluation
  retrieval/     BM25, embeddings, related-question index and explanations
  graph/         heterogeneous graph schema, construction, analysis, snapshots
  drift/         drift metrics, calibrated alerts, controlled evaluation
  models/        GraphSAGE link prediction (PyTorch Geometric)
  experiments/   PPI reproduction, link-prediction experiment
  evaluation/    ranking metrics and bootstrap CIs
backend/       FastAPI app (api/, core/, db/, schemas/, services/), loader, demo data, job runner, tests
migrations/    Alembic migrations (PostgreSQL + pgvector)
frontend/      Next.js dashboard (JavaScript, Tailwind, Recharts, React Flow)
configs/       pipeline settings and transparent entity/category dictionaries
docs/          research design, data policy, API, milestone log, per-component write-ups
tests/         pipeline/ML tests (synthetic fixtures; no network)
```

## Data use

Content is CC BY-SA, from Ask Different via the official API. This repository does not
redistribute collected data. Only author display names and profile links are stored, for
attribution. No content is sent to external LLMs. See [docs/data_policy.md](docs/data_policy.md).
