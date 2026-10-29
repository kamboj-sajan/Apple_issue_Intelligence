# Milestone Log

Numbers below come from actual runs. They will change as ingestion continues.

## Milestone 1: Research design (done)

* [research_design.md](research_design.md): hypotheses H1–H5, scope, split and leakage rules,
  evaluation protocols, GraphSAGE PPI reproduction plan (Table 1 numbers checked against the paper).
* [data_policy.md](data_policy.md): licensing, attribution, API etiquette, privacy.

## Milestone 2: Data pipeline (code done, collection partial)

### What exists

| Component | File |
|-----------|------|
| API client (spacing, `backoff`, retries, throttle/quota) | `ml/data/stackexchange_client.py` |
| Checkpointed creation-chunk, activity-update and by-id ingestion | `ml/data/ingest.py` |
| Verbatim raw page store | `ml/data/raw_store.py` |
| HTML cleanup / normalisation / link extraction | `ml/preprocessing/text.py` |
| Rule-based entities and categories | `ml/preprocessing/entities.py`, `configs/entities.yaml` |
| Dedup, UTC timestamps, quality report, links table | `ml/preprocessing/normalize.py` |
| Week-aligned temporal splits | `ml/preprocessing/splits.py` |
| CLI | `ml/cli.py` (`aii ...`) |

### First real ingestion (2026-10-29, anonymous quota)

| Item | Value |
|------|-------|
| Raw pages stored | 282 |
| Questions (unique) | 26,391 |
| Created range | 2019-01-01 → 2021-12-15 (Dec 2021 partially collected) |
| Months complete | 35 (2019-01 … 2021-11) |
| Snapshot id | `f7a2cf045a3f` |
| Questions closed as duplicate | 1,264 (1,439 duplicate links) |
| Body links to other questions | 1,734 |
| Link targets not yet in corpus | 1,828 unique ids (mostly pre-2019 canonical questions) |
| Entity matches | device 32,085 · tech 33,987 · os 15,504 · category 41,149 · error 549 |
| Uncategorised questions | 5,434 (20.6 %) |
| Split (in-window) | train 18,423 · val 3,992 (from 2020-11-30) · test 3,976 (from 2021-05-24) |

The run stopped at the quota reserve (`stopped_reason: quota_reserve`), as designed. No retries or
errors occurred.

### Fixes made after inspecting real data

* A bare 8-hex-digit "error code" rule matched memory addresses and register dumps in crash logs
  (`0x00000000`, `0xff000000` were the most frequent "errors"). The rule was removed, and
  uninformative codes (`0`, `1`, `-1`, all-zero hex) are now ignored. Error matches fell from
  1,254 to 549, and the most frequent are now real codes (diskutil `-69743`, `-69877`, `NSPOSIXErrorDomain`).
* Only 366 of 1,439 duplicate links pointed to a collected question. Added `aii fetch-linked`
  (fetch by id, 100 per request). Such questions are flagged `in_research_window = False`, so they
  serve as link-prediction context without inflating weekly volumes or shifting split boundaries.

### Commands

```bash
aii ingest            # continue the creation crawl (resumes at 2021-12 page 4)
aii fetch-linked      # ~19 requests for the missing link targets
aii update            # activity-based refresh (captures later duplicate closures)
aii preprocess && aii split
pytest -q             # 33 tests
```

### Tests

33 passed (`pytest -q`); `ruff check` and `ruff format --check` clean. Covered: client retries,
backoff, throttling and key redaction; pagination; checkpoint resume; idempotent re-runs; activity
updates; by-id fetching; HTML/Unicode cleanup; link extraction; entity rules; dedup; UTC
handling; missing values; research-window flags; split ordering, week alignment and determinism.

### Known limitations

* **Collection incomplete:** 2021-12 onwards is not yet collected. At the anonymous 300/day quota
  the rest of 2019–2026 needs about one more day; an app key makes it a single run.
* **Activity update mode is untested on live data** (only on fixtures). Until it runs, duplicate
  closures that happened after a page was fetched are missing.
* **Body-link timing is conservative:** a link is assumed to exist from the question's last edit (or
  creation), because the API does not expose when a link was added.
* **Rules are high-recall, imperfect precision:** e.g. `terminal` also matches the word in prose; "error 2"
  is generic. Rule hits are evidence for explanations, not ground truth.
* `content_license` is missing for 1,774 questions (the API omitted it); those default to the site licence.
* The first ISO week (2018-12-31) contains only part of the week's questions. Aggregation must treat
  edge weeks as partial.
* Deleted questions disappear from pagination; a resumed chunk could skip an item if a deletion shifts
  page boundaries between runs. Rare, and a later `update` run reconciles it.

### Prerequisites for Milestone 3

1. Finish collection through the present (`aii ingest` over 1–2 quota days) and run `aii fetch-linked`
   and `aii update`.
2. Install ML extras: `pip install -e ".[ml]"` (statsmodels, scipy, sentence-transformers, rank-bm25).
3. Choose the sentence-transformer model (plan: `all-MiniLM-L6-v2`, local inference) and cache embeddings
   keyed by snapshot id.
