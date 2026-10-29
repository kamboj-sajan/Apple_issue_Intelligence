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
| 2. Data pipeline | Code and tests done; real ingestion run in progress (quota-limited) |
| 3. Anomaly detection, BM25 and embedding baselines | Not started |
| 4. Knowledge graph | Not started |
| 5. GraphSAGE (benchmark reproduction and link prediction) | Not started |
| 6. Drift monitoring | Not started |
| 7. API and dashboard | Not started |
| 8. Final validation | Not started |

Detailed progress and limitations: [docs/milestones.md](docs/milestones.md).

## Quick start

Requires Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate     # or: uv venv --python 3.12 .venv
pip install -e ".[dev]"
cp .env.example .env                                   # optional: add STACKEXCHANGE_KEY
pytest -q                                              # offline tests on synthetic fixtures
```

## Data pipeline

```bash
aii ingest                 # fetch questions by month since 2019-01-01 (configs/pipeline.yaml); resumable
aii ingest --max-requests 50 --start 2024-01-01   # smaller run
aii update                 # re-fetch questions with new activity (edits, duplicate closures)
aii preprocess             # raw pages -> data/processed/*.parquet + manifest.json
aii split                  # time-based train/val/test split + split_manifest.json
aii status                 # checkpoint and snapshot summary
```

Without an API key the Stack Exchange quota is **300 requests/day** (100 questions each). The
ingestion stops cleanly at the quota reserve and resumes from its checkpoint on the next run.
With a free key from [stackapps.com](https://stackapps.com/apps/oauth/register), the quota is 10,000 requests/day.

### Data layout (`data/`, git-ignored)

```
data/
├── raw/stackexchange/apple/questions/
│   ├── creation/<YYYY-MM>/page_0001.json.gz      # verbatim API pages + request metadata
│   └── activity/activity_<run>/page_0001.json.gz # incremental updates
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
  preprocessing/ HTML cleanup, entity extraction, normalisation, temporal splits
configs/       pipeline settings and transparent entity/category dictionaries
docs/          research design, data policy, milestone log
tests/         pytest suite (synthetic fixtures only; no network)
```

## Data use

Content is CC BY-SA, from Ask Different via the official API. This repository does not
redistribute collected data. Only author display names and profile links are stored, for
attribution. No content is sent to external LLMs. See [docs/data_policy.md](docs/data_policy.md).
