# API

FastAPI app at `backend/app/main.py`. Interactive docs are at `http://localhost:8000/docs` while the
server runs. Every response carries `meta.data_source`:

* `real`: calculated from collected Ask Different data (snapshot id in `meta.snapshot_id`);
* `demo`: the small synthetic dataset loaded with `aii-db load-demo`;
* `empty`: nothing has been loaded yet.

Errors always look like `{"error": {"code": "...", "message": "...", "details": ...}}`. Codes are
`validation_error` (422), `not_found` (404), `job_already_running` (409), `not_available` (404) and
`internal_error` (500).

## Endpoints

| Method | Path | Purpose | Main parameters |
|--------|------|---------|-----------------|
| GET | `/health` | liveness, database check, data source | |
| GET | `/overview` | totals, latest analysis time, freshness, top categories | |
| GET | `/questions` | paginated questions | `limit` (≤100), `offset`, `q` (title or id), `start_date`, `end_date`, `tag`, `device`, `os`, `category`, `sort` |
| GET | `/questions/{id}` | question, rule-based entities with evidence, CC BY-SA attribution | |
| GET | `/issues/trending` | category volume change + most recent signals | `weeks` (1–104), `limit` |
| GET | `/issues/series` | weekly observed vs expected for an issue group | `group_type`, `group_value`, dates |
| GET | `/issues/groups` | groups with a stored series (for pickers) | `group_type`, `q` |
| GET | `/anomalies` | ranked, filterable signals | `group_type`, `group_value`, `device`, `os`, `tag`, dates, `min_score`, `signal_type`, `sort`, pagination |
| GET | `/anomalies/{id}` | signal with explanation, ±1 year series, evidence questions, limitations | |
| GET | `/graph/neighbors/{node_id}` | bounded neighbourhood for visualisation | `depth` (1–2), `limit` (edges per node, ≤100), `max_nodes` (≤300), `node_types`, `edge_types` |
| GET | `/graph/search` | find nodes by label | `q`, `node_type` |
| GET | `/related-questions/{id}` | BM25 / embedding / GraphSAGE recommendations with evidence | `methods`, `k` (≤10) |
| GET | `/metrics/model` | link-prediction, retrieval, PPI, anomaly and drift evaluation; splits; leakage controls | |
| GET | `/drift` | monthly drift metrics and alerts, cluster movements, new terms, novelty, evaluation | `metric`, `start_month`, `end_month`, `alerts_only` |
| POST | `/jobs/ingest` | start ingestion (`creation`, `update`, `fetch-linked`) | body `{mode, max_requests}` |
| POST | `/jobs/analyze` | rerun pipeline steps, then reload the database | body `{steps, load_database}` |
| GET | `/jobs`, `/jobs/{id}` | job status, current step, log tail | |

### Explanations

* **Anomalies:** the triggering week, the 26-week baseline period, observed vs expected with the site
  total, the statistical test, the spike/sustained/too-recent label, whether the rolling detector agrees,
  data-quality flags, co-occurring tags and entities, the most-viewed questions of that week (with links),
  and a limitations statement.
* **Related questions:** shared tags, shared entities by type, overlapping terms ranked by IDF, the
  embedding cosine, direct graph edges between the pair, and any *known* duplicate or link between them
  (which no method used as input).

## Serving choices (differences from the evaluation)

* Related questions are **precomputed** at load time: the top 10 *earlier* questions per method for
  every question. BM25 is fit on the whole snapshot. GraphSAGE re-ranks the 100 most similar earlier
  questions by embedding, using the graph as of the snapshot. The evaluated model instead ranked all
  candidates with the per-query graph.
* Lexical evidence uses the first 600 characters of each question body.

## Background jobs: design and limitations

Jobs run the existing CLI steps (`python -m ml.cli <step>`, then `python -m backend.app.cli load`) as
**subprocesses from a worker thread**. Requests are never blocked; status and a rolling log tail are
stored in the `jobs` table.

* One job per type at a time; a second request gets HTTP 409.
* Jobs live inside the API process. A restart marks running jobs `interrupted` and they aren't resumed;
  rerun them, since ingestion itself resumes from its checkpoint.
* Run the API with **one worker** (the Docker image does). With several workers each would have its own
  runner. At larger scale, replace this with a proper queue (e.g. RQ or Celery).
* While `load` replaces the tables (one transaction), API reads may wait briefly for it to finish.
* Ingestion needs network access and, for more than 300 requests a day, `STACKEXCHANGE_KEY`.
