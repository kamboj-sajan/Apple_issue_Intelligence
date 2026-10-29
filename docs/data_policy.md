# Data Use, Licensing and Privacy

## Source and licence

* Data comes from **Ask Different** (<https://apple.stackexchange.com/>) through the official
  **Stack Exchange API v2.3**. User contributions are licensed under **CC BY-SA** (version per
  post, recorded in the `content_license` field).
* Attribution: any place this project displays question content links back to the original
  question URL and shows the author display name linked to their public profile, as required by
  Stack Exchange's attribution guidance. That is the only reason those two fields are stored.
* This repository **does not redistribute** collected data. `data/` is git-ignored; anyone can
  regenerate it with `aii ingest`. Test fixtures under `tests/fixtures/` are *synthetic*, written by
  hand to mimic the API's JSON shape, and are labelled as such.
* Review the current [Stack Exchange API Terms of Use](https://stackexchange.com/legal/api-terms-of-use)
  before running large collections. This project is a non-commercial research/portfolio project
  and does not train generative language models on the content.

## API etiquette implemented

* Requests are sequential with a configurable minimum interval (default 0.5 s), far below the
  documented 30 requests/second limit.
* The `backoff` field in any response is honoured before the next request to the same method.
* Throttle and transient errors (HTTP 429/5xx, `throttle_violation`) are retried with exponential
  backoff; quota exhaustion stops the run cleanly and the checkpoint allows resuming the next day.
* Anonymous quota is 300 requests/day; with an app key (`STACKEXCHANGE_KEY`) it is 10,000.

## Privacy

* Not collected: user ids, account ids, reputation, profile images, emails, comments, answers' authors.
* No content is sent to external LLM APIs. Embeddings are computed locally with an open
  sentence-transformer model.
* Outputs describe *candidate issues* and *discussion patterns*, never confirmed defects.
