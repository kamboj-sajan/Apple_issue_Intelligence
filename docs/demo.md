# Demonstration walkthrough

A 10-minute tour of the system using real results from snapshot `9e8dce73e94a`. Start the stack
(`docker compose up -d --build`, then load results with `aii-db load` or
`docker compose exec api python -m backend.app.cli load`) and open <http://localhost:3000>.
Every page shows a banner saying whether the data is **real** or the synthetic **demo** dataset.

## 1. Overview: what is in the system

`/`. 42,555 questions from 2019-01-01 to 2026-10-27, 21 statistically significant signals, the latest
analysis time and the last collection run. The category chart shows the most active rule-based issue
categories over the last 12 months.

## 2. Emerging issues: the macOS Catalina surge

`/issues?anomaly=b7a9657b4827b818`

* The weekly chart for `os → macos:10.15` shows the week of **2019-10-07**: **124 questions observed vs
  6.8 expected** (FDR q ≈ 6e-16), labelled **sustained** because a change point starts that week.
* The explanation panel gives the 26-week baseline period, the test used, and data-quality checks.
  It lists real questions from that week, e.g. "Mac stuck in restart cycle after failed macOS Catalina
  update", each linking to Ask Different.
* Compare `os → macos:11` (Big Sur, 2020-11-09: 45 vs 3.3) and the September iOS releases. The
  limitations box says why these are discussion patterns, not defects.
* Try the filters: `Type = spike`, or `Tag = backup` (a spike in August 2022).

API: `GET /anomalies?os=macos:10.15`, `GET /anomalies/b7a9657b4827b818`.

## 3. Related questions: a December 2025 duplicate of a 2015 question

`/related?id=485393`. "How to disable Safari Sidebar pop out in Full Screen, Mac OS Tahoe 26.2"
(December 2025).

* **All three methods rank question 209114 first.** That is a much older question ("How to turn off
  Auto-show sidebar in fullscreen Safari 9"), and it *is* the community-chosen duplicate (badge "known
  duplicate"). None of the methods used that label as input.
* The evidence under each recommendation shows the shared tag (`safari`), shared terms (sidebar, disable,
  …), embedding cosine 0.823, and the graph relations between the two questions.
* Further down the lists the methods disagree. BM25 prefers word overlap ("BBEdit sidebar…"); the
  embedding and GraphSAGE lists stay closer to the full-screen sidebar theme.

API: `GET /related-questions/485393?k=5`.

## 4. Knowledge graph: exploring the neighbourhood

`/graph?node=question:485393`. The question is connected to its tags (`safari`, `macos-tahoe`), the
OS node `macos:26`, categories, similar earlier questions, and other Tahoe 26.2 questions from the same
week (temporal edges). The dashed edge marked **duplicate** is the known label. Double-click a node to
expand it; use the filters to hide node or edge types. Views are capped so large neighbourhoods stay
responsive (try `tag:macos`).

## 5. Drift monitoring: what changed, and when

`/drift`

* The embedding-MMD chart spikes in **2019-10** (Catalina), well above the calibrated threshold.
* Alerts table: the OS-version mix alerts only around major release cycles (2019-10, 2021-11, 2023-10),
  because its threshold (z = 12.8) is calibrated on history where the OS mix shifts every year.
* New vocabulary: `m1` (2020-11), `m3` (2023-11), **`tahoe` (2025-09)**, plus each point release.
* The cluster table shows, in October 2019, the iTunes/Music cluster nearly tripling, the month
  Catalina replaced iTunes with the Music app.
* The controlled evaluation shows 0 % false alarms on same-month random splits for every metric except
  vocabulary (5 %), and an 80 % detection rate for a topic making up 20 % of a month.

## 6. Model evaluation: the honest result

`/models`

* GraphSAGE MRR **0.441** vs embedding cosine **0.435**. The paired difference is **not significant**
  (+0.006 [−0.005, +0.017]). Both clearly beat BM25 (0.341) and the Jaccard heuristic (0.084).
* The selection table shows why: without a raw-text residual, GraphSAGE reached only 0.306 on
  validation. The no-message-passing ablation performs almost exactly like full GraphSAGE.
* The PPI reproduction (0.580 ± 0.018 vs the paper's 0.598) shows the GraphSAGE implementation itself
  behaves as published.
* Splits and leakage controls are listed at the bottom of the page.

## 7. API and background jobs

* Interactive API docs: <http://localhost:8000/docs>.
* Start a job: `curl -X POST localhost:8000/jobs/analyze -H 'Content-Type: application/json' -d
  '{"steps":["split"],"load_database":false}'`, then follow it with `GET /jobs/{job_id}`. The API keeps
  answering while the job runs as a subprocess.
