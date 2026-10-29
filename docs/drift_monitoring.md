# Drift Monitoring

Code: `ml/drift/` (`metrics.py`, `pipeline.py`, `evaluation.py`). Commands: `aii drift`, `aii drift-eval`.
Configuration: the `drift:` section of `configs/pipeline.yaml`. Results below are for snapshot `9e8dce73e94a`.

Drift monitoring asks whether **what people ask about** changes over time. It is deliberately
separate from volume anomaly detection (Milestone 3): a topic can change without a volume spike,
and a volume spike can happen without a change in content.

## Method

* **Windows:** calendar months. Weekly windows are too small by 2025, at about 25 questions a week.
  Each month is compared with the pooled **three preceding months**.
* **Equal sample sizes:** both samples are subsampled to `min(n_ref, n_cur, 300)`. Months with
  fewer than 50 questions are reported as *insufficient data* (2026-07 to 2026-10).
* **Metrics:**

  | Metric | What it compares | Limitation |
  |--------|------------------|------------|
  | `centroid_cosine` | 1 − cosine between mean MiniLM embeddings | misses changes that keep the mean fixed (e.g. a topic splitting) |
  | `mmd_rbf` | unbiased MMD² with an RBF kernel (median-distance bandwidth) | sensitive to everything; scale is hard to interpret |
  | `tags`, `categories`, `os`, `devices`, `clusters` | Jensen–Shannon divergence of value frequencies | the reference's top 200 values plus `__other__`; rule-based values inherit rule errors |
  | `vocabulary` | JSD of title-term document frequencies | term sets, not meaning |

* **Sample-size correction:** for every metric and month, 100 permutations of window membership give a
  null distribution, a standardised score `z` and a permutation p-value. Raw JSD and MMD grow with
  sampling noise, so `z` is what is compared across months.
* **Calibrated alerts:** each metric's threshold is the **95th percentile of its z-scores over the
  calibration months** (all months before the validation boundary, 2022-06). An alert needs
  `z > threshold`, permutation `p < 0.01`, and 50 or more questions per side. "Alert" therefore means
  more drift than this site usually shows, not "any change". For example, the OS-version threshold is
  z = 12.8, because the OS mix shifts every release cycle.
* **Clusters:** shares of the 40 training-period clusters are tracked month by month (`cluster_movements`).
  A question is **novel** when its nearest training centroid is below the 5th percentile of training
  questions (cosine 0.350). Each quarter, novel questions are grouped (average-linkage, cosine). A group
  counts as a **candidate new cluster** if it has 5 or more questions and is at least as coherent as the
  tightest 10 % of training clusters (mean within-cluster distance 0.643, calibrated rather than chosen).
* **New vocabulary:** title terms used by 5 or more questions in a month and by at most 1 in the previous 12 months.

## Results (87 months evaluated)

| Month | Alerting metrics | Plausible context |
|-------|------------------|-------------------|
| 2019-10 | centroid, MMD, OS | macOS Catalina release (also the largest volume signal) |
| 2021-11 | categories, OS | macOS Monterey cycle |
| 2021-12 | categories | |
| 2022-03 | clusters | |
| 2022-09 | categories | iOS 16 / macOS Ventura cycle |
| 2022-11 | centroid, MMD | |
| 2023-06 | devices, MMD, vocabulary | WWDC 2023 |
| 2023-10 | categories, centroid, OS | macOS Sonoma cycle |
| 2025-09 | vocabulary | "tahoe" appears (macOS 26 Tahoe) |

* **Alert rate:** 2.3 % of metric-months in the calibration period and 2.6 % afterwards. Below 5 %, because
  alerts also need p < 0.01.
* **Cluster movements:** in October 2019 the update/upgrade cluster (×2.9) and the **iTunes/Music** cluster
  (×2.8) grew sharply; Catalina replaced iTunes with the Music app that month. Time Machine/backup
  questions jumped in August 2022 (×2.9), matching the backup-tag volume signal from Milestone 3.
* **New terms** track releases: `m1` (2020-11), version numbers such as `11.2`, `12.3.1`, `13.1`,
  `m3` (2023-11), `tahoe` (2025-09).
* **Novelty:** the novel-question rate stays near 5 % every year (4.2–5.8 %). Out-of-distribution
  questions are not increasing. Only 4 small candidate new clusters were found (5–6 questions: Apple
  Numbers/PDF questions in 2019–20, a Catalina group in early 2020). Most novel questions are scattered
  outliers, not a coherent new topic.
* **H5 (volume vs semantic drift):** Spearman correlation between monthly MMD z and the number of
  volume signals is **ρ = 0.19 (p = 0.08, 87 months)**. They coincide in big release months (2019-10)
  but are mostly different signals, so H5 is supported.

## Controlled evaluation (`aii drift-eval`)

* **No drift:** 20 calibration months, each split at random into two halves. False-alarm rate is
  **0 %** for every metric except vocabulary (**5 %**, 1 of 20).
* **Injected topic surge:** a fraction of a month's sample is replaced by questions from one cluster
  (15 trials per fraction). Detection rate:

  | Surge | Centroid | MMD | Clusters | Tags | Categories | Vocabulary | Devices | OS |
  |-------|----------|-----|----------|------|------------|------------|---------|----|
  | 5 % | 0.07 | 0.00 | 0.07 | 0.00 | 0.00 | 0.00 | 0.07 | 0.00 |
  | 10 % | 0.21 | 0.21 | 0.29 | 0.07 | 0.14 | 0.00 | 0.00 | 0.00 |
  | 20 % | 0.80 | 0.80 | 0.80 | 0.33 | 0.27 | 0.13 | 0.07 | 0.00 |
  | 40 % | 1.00 | 1.00 | 0.93 | 0.93 | 0.71 | 0.64 | 0.29 | 0.00 |

  Embedding and cluster metrics reliably catch a topic that takes over **20 %** of a month and miss
  5–10 % shifts. The OS metric never fires on topic surges, by design: its calibrated threshold only
  reacts to release-scale OS shifts.

## Limitations

* Thresholds come from 2019–2022 history. If the site's character changes, recalibration is needed.
* With 300-question samples, small shifts (under 10 %) are not detectable. Late-2026 months have too few
  questions to evaluate at all.
* Clusters and centroids are fixed from the training period, so a topic that grows inside an existing
  cluster shows up as a share change, not as novelty.
* Alert months are matched to release cycles only for plausibility. These are discussion patterns,
  not evidence of defects.
