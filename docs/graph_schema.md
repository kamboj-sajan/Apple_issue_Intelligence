# Knowledge Graph Schema

Source: `ml/graph/schema.py` (types and validation) and `ml/graph/build.py` (construction rules).
Configuration: `graph:` section of `configs/pipeline.yaml`.

## Identifiers and time

* **Node id** = `<node_type>:<value>`. Values are source ids or canonical entity values from
  `configs/entities.yaml`, so ids are stable across rebuilds:
  `question:488267`, `tag:wifi`, `device:macbook-pro`, `device:iphone:15`, `os:macos:14`,
  `error:error:-36`, `tech:time-machine`, `category:battery-power`, `week:2023-09-04`.
* Every node and edge has **`known_at`** (UTC), the earliest time the fact was observable.
  `filter_as_of(nodes, edges, t)` keeps facts with `known_at < t` and reproduces the graph as it
  could have been built at `t`. Validation rejects any edge known before one of its endpoints.

## Node types

| Type | Value | `known_at` | Attributes |
|------|-------|------------|------------|
| `question` | Stack Exchange question id | creation time | title (`label`), link, split, in_research_window, primary_category |
| `tag` | tag name | first tagged question | |
| `device` | family (`iphone`) or family:model (`iphone:15`) | first mention | |
| `os` | `macos:10.15`, `macos:14`, `ios:17`, … | first mention | |
| `error` | `<prefix>:<code>` (e.g. `error:-36`, `domain:nscocoaerrordomain`) | first mention | |
| `tech` | technical entity (`time-machine`, `filevault`, …) | first mention | |
| `category` | rule-based issue category | first assignment | |
| `week` | Monday of the ISO week | week start | |

## Edge types

| Type | Endpoints | Weight | Rule |
|------|-----------|--------|------|
| `QUESTION_HAS_TAG` | question → tag | 1.0 | explicit Stack Exchange tag |
| `QUESTION_MENTIONS_DEVICE` | question → device | 1.0 title/tags, 0.6 body, 0.4 code | entity rule match |
| `QUESTION_MENTIONS_VERSION` | question → os | same | entity rule match |
| `QUESTION_CONTAINS_ERROR` | question → error | same | entity rule match |
| `QUESTION_MENTIONS_TECH` | question → tech | same | entity rule match *(extension)* |
| `QUESTION_IN_CATEGORY` | question → category | rule score ÷ the question's max score | category rules |
| `QUESTION_POSTED_IN_WINDOW` | question → week | 1.0 | creation week *(extension)* |
| `QUESTION_SEMANTICALLY_SIMILAR` | question → earlier question | cosine | top-10 earlier questions with cosine ≥ 0.6 |
| `QUESTION_TEMPORALLY_NEAR_QUESTION` | question → earlier question | shared × e^(−days/3) | ≤ 7 days apart, ≥ 2 shared specific entities/tags, top 5 |
| `ISSUE_SPIKED_DURING_WINDOW` | tag/device/os/error/tech/category → week | −log10(q) | Milestone 3 anomaly signal; known at the end of the week |
| `QUESTION_DUPLICATE_OF` | question → question | 1.0 | community duplicate closure; **label edge** |
| `QUESTION_LINKS_TO` | question → question | 1.0 | hyperlink in body; **label edge** |

Edges are stored directed (`src` → `dst`). The NetworkX view is an undirected `MultiGraph` keyed by
edge type, with `src`/`dst` kept as attributes.

### Why these thresholds

* **Semantic edges (cosine ≥ 0.6, top-10, earlier only).** On snapshot `3d642dc498c8`, random
  question pairs have a 99.9th-percentile cosine of 0.534. Duplicate pairs have a median of 0.627 and
  a 25th percentile of 0.533. A typical question's 10th-most-similar earlier question scores about
  0.57. So 0.6 avoids near-random links and keeps the graph sparse (median question degree 15).
* **Temporal edges.** Questions posted in the same week are only linked if they share at least two
  *specific* entities or tags. Entities or tags that appear on more than 5 % of questions (e.g. the
  `macos` tag) don't count, which avoids linking every question to every other.
* **Label edges** are excluded from graphs used for link prediction
  (`to_networkx(..., include_label_edges=False)`). Milestone 5 adds training-time label edges only
  when their `known_at` precedes the training cutoff.

## Leakage notes

* Semantic and temporal edges always point from a question to **earlier** questions, and are known
  at the later question's creation time.
* Embeddings come from a frozen pretrained model, so they carry no label information.
* **Known limitation:** question text is the latest revision (the API doesn't return revision
  history without extra calls). Edits made after a cutoff can therefore appear in the text of
  questions created before it. This affects every text-based method equally.

## Analysis outputs (`data/artifacts/graph/<snapshot>/`)

| File | Content |
|------|---------|
| `nodes.parquet`, `edges.parquet` | full graph tables (any snapshot = `known_at` filter) |
| `summary.json` | validation, counts, degree summary, components, communities, top bridges |
| `degree_distribution.csv` | log-binned degree histogram per node type |
| `communities.parquet`, `community_summary.parquet` | Louvain (seed 0) on the question projection; top tags/OS/devices and time span |
| `bridges_{tag,os,device}.csv` | communities connecting values that rarely co-occur on a question |
| `snapshots.csv` | structure at each 1 January since 2020 and at the latest question |

Bridge score: `min(share_a, share_b) × (1 − overlap_ab)`, where `overlap_ab` is the global
overlap coefficient |A∩B| / min(|A|,|B|). Jaccard was tried first and rejected: it ranked
`keyboard + macos` as a bridge just because `macos` is very common.
