# Limitations

All known limitations in one place, gathered from every milestone. Each is either measured in this
project or a deliberate design trade-off; none is hidden in the results.

## What the outputs mean

* **Signals are not defects.** An anomaly says that *posted questions* about a topic rose above
  expectation. It does not show a product defect, a cause, or how many people are affected. Ask
  Different users are a small, self-selected group of technical users.
* **Release weeks dominate.** The strongest signals line up with OS releases (Catalina 2019-10, Big Sur
  2020-11, September iOS releases). Annual release cycles aren't modelled per group, so these show up as
  surges. That is real, but not surprising.
* **The site is shrinking.** Weekly volume fell about 6× between 2019 and 2025. Detectors model each
  group's *share* of site volume to compensate, but by 2025–26 many groups are too small to test, and
  monthly drift windows from mid-2026 onwards have too few questions (under 50) to evaluate.

## Data

* **Latest revision only.** The API returns the current text of each question, not its revision
  history. Edits made after an evaluation cutoff can therefore appear in text features. This affects
  every text-based method equally.
* **Incomplete labels.** Duplicate closures and body links are the only "related" labels, and many
  truly related pairs were never linked. Unlinked pairs are not assumed unrelated, and classification
  metrics (AUC, AP) are lower bounds because sampled negatives may include unlabelled duplicates.
* **Body-link timing is approximate.** A link is assumed to exist from the question's last edit or
  creation; the API doesn't say when a link was added.
* **Updates need re-running.** `aii update` catches later edits and closures, but only when someone
  runs it.
* **Deleted questions** disappear from the API, and a resumed collection chunk can skip an item if page
  boundaries shift between runs.
* `content_license` is missing for 3,097 questions (the API omitted it).

## Entity extraction and categories

* Rules are transparent but imperfect. They favour recall, so precision suffers: `terminal` also
  matches the word in prose, and "error 2" is generic. About 21 % of questions get no category.
* An early rule treated memory addresses in crash logs as error codes. It was found on real data and
  removed. Other rules may have similar, undiscovered failure modes.

## Anomaly detection

* **Small groups are hard to monitor.** With FDR control across about 1,500 groups, a doubling in a group
  with fewer than 5 questions a week is caught only about 18 % of the time (injection study).
* The negative-binomial test is conservative (3.1 % false positives at a nominal 5 %). The rolling robust
  z flags too often (6.8 %), so it is used only as supporting evidence.
* Change-point detection looks at the whole series. It labels signals as spike or sustained after the
  fact and never raises alerts. Weak shifts are often labelled spikes (46 % correct at ×2).

## Related questions and GraphSAGE

* **GraphSAGE does not beat plain embedding cosine at ranking.** Test MRR is 0.441 vs 0.435; the paired
  difference is +0.006 [−0.005, +0.017], which is not significant. It is significantly better at pair
  classification (AP +0.064), but most of that comes from supervised training; message passing adds +0.012.
* Why the graph adds little: semantic edges come from the same embeddings, entity and tag hubs blur
  neighbourhoods under mean aggregation, labels are few (3,257 training pairs), and later questions
  have sparse neighbourhoods.
* The no-message-passing ablation uses one seed; three would firm up the "graph adds little" conclusion.
* Validation for model selection used week-level as-of graphs (faster). Test evaluation used strict
  per-query times.
* **Serving is an approximation.** The dashboard shows GraphSAGE as a re-ranking of the embedding top
  100 using the snapshot graph. The evaluation ranked all earlier questions with per-query graphs.
* The benchmark reproduction (PPI micro-F1 0.580 ± 0.018 vs the paper's 0.598) used one learning rate
  rather than the paper's sweep. It validates the implementation, not Ask Different performance.

## Drift monitoring

* Alert thresholds come from 2019–2022 history and need recalibrating if the site changes character.
* With 300-question samples, topic shifts under 10 % of a month are not detected. A 20 % surge is
  caught about 80 % of the time.
* Clusters are fixed from the training period. A topic growing inside an existing cluster shows up
  as a share change, not as novelty. Only 4 small new-cluster candidates were found.

## Engineering

* **Background jobs** run in the API process: one job per type at a time, a single API worker, and no
  resume after a restart (jobs are marked `interrupted`; ingestion itself resumes from its checkpoint).
* **No authentication** on the API or dashboard. Compose binds every port to 127.0.0.1; don't expose
  the stack publicly as is.
* The API image is 2.8 GB because it includes the CPU ML stack, so jobs can rerun the pipeline.
* One active snapshot in the database; loading replaces everything inside one transaction.
* The full link-prediction experiment takes about 68 minutes on a laptop CPU; loading the database
  about 3.5 minutes.
* Without an API key, Stack Exchange allows 300 requests a day per IP. A fresh full collection needs
  a free key (about 500 requests).
