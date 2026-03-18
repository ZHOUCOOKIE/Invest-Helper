# API Contract

Authoritative

TL;DR
- OpenAPI schema lives at `GET /openapi.json`.
- Swagger UI lives at `GET /docs`.
- This file summarizes current route families and business semantics; exact field schema should be cross-checked with OpenAPI and code.

## Core Invariants

- All opinion outputs must remain traceable to `source_url`, `raw_posts`, and `post_extractions`.
- Daily digest replay is by `(profile_id, digest_date)` and weekly digest replay is by `(profile_id, report_kind, anchor_date)`.
- Public API currently operates on the system profile `id=1` for digest and dashboard flows.
- Historical `approved/rejected` records are read-tolerant; legacy `library_tags` is ignored.

## Health And Runtime

- `GET /health`
- `GET /extractor-status`

## Assets And KOLs

- `GET /assets`
- `POST /assets`
- `POST /assets/upsert`
- `GET /kols`
- `POST /kols`
- `GET /kols/{kol_id}/assets-summary`
- `GET /kols/{kol_id}/views`
- `GET /assets/{asset_id}/views`
- `GET /assets/{asset_id}/views/feed`
- `GET /assets/{asset_id}/views/timeline`
- `GET /assets/{asset_id}/views/{view_id}/post-detail`

Behavior notes:
- asset view grouping uses the latest row per `(kol_id, asset_id, horizon)`
- timeline and post-detail endpoints preserve the link back to source post and extraction context

## Ingest And Import

- `GET /ingest/x/import/template`
- `POST /ingest/x/following/import`
- `POST /ingest/x/convert`
- `GET /ingest/x/raw-posts/preview`
- `POST /ingest/x/import`
- `POST /raw-posts/extract-batch`
- `POST /raw-posts/{raw_post_id}/extract`
- `GET /ingest/x/progress`
- `POST /ingest/x/retry-failed`
- `POST /ingest/x/retry-pending-all`

Import/output notes:
- `POST /ingest/x/import` returns inserted IDs, deduped IDs, and `pending_failed_dedup_*` fields for raw posts whose latest extraction is still failed semantics
- `GET /ingest/x/progress` reports extraction progress counts globally or per author
- `POST /ingest/x/retry-failed` targets latest failed-semantics rows only

## Async Extract Jobs

- `POST /extract-jobs`
- `GET /extract-jobs/{job_id}`

Job payload notes:
- job creation deduplicates requested `raw_post_ids`
- in-flight jobs may be reused when queued/running coverage already exists
- job status includes `ai_call_used`, `openai_call_attempted_count`, `success_count`, `failed_count`, `skipped_count`, `max_concurrency_used`, `last_error_summary`

## Extractions

- `GET /extractions`
- `GET /extractions/stats`
- `GET /extractions/{extraction_id}`
- `POST /extractions/{extraction_id}/re-extract`
- `POST /extractions/{extraction_id}/approve`
- `POST /extractions/{extraction_id}/approve-batch`
- `POST /extractions/{extraction_id}/reject`

List semantics:
- list returns only the latest extraction per `raw_post`
- `status` filter supports `pending`, `approved`, `rejected`, `library`, `all`
- keyword filter is `q`
- final ordering is by business post time descending with fallback `raw_post.posted_at -> extraction.created_at`, then `id`

Replacement semantics:
- `POST /extractions/{id}/re-extract` creates a new extraction for the same `raw_post`
- if the new result is a valid AI extraction, older extraction rows for that raw post can be deleted
- if the new result is failed semantics, old rows are kept

## Extraction JSON Contract

Normalized business keys:
- `as_of`
- `source_url`
- `islibrary`
- `hasview`
- `asset_views`
- `library_entry`

Canonical order:
- top level: `as_of, source_url, islibrary, hasview, asset_views, library_entry`
- `asset_views[*]`: `symbol, market, stance, horizon, confidence, summary`

Field constraints:
- `market`: `CRYPTO|STOCK|ETF|FOREX|OTHER`
- `stance`: `bull|bear|neutral`
- `horizon`: `intraday|1w|1m|3m|1y`
- `asset_views[*].confidence` below `70` is dropped by normalization
- `hasview` is recomputed from final `asset_views`
- `islibrary=1` with invalid `library_entry` is downgraded to `islibrary=0`
- current library validation still requires `library_entry.summary == "测试"`

Storage and tolerance:
- `parsed_model_output` is stored as ordered `JSON`
- parse-failed outputs still create `pending` rows and write error context into `last_error` and compact `meta`
- read path tolerates historical rows and ignores legacy `library_tags`

## Dashboard

- `GET /dashboard`

Current response families include:
- pending extraction queue
- top assets and recent view counts
- clarity metrics and clarity ranking
- recent extraction stats
- active KOLs

## Daily Digest

- `POST /digests/generate`
- `GET /digests`
- `GET /digests/dates`
- `GET /digests/{digest_id}`

Current semantics:
- query `date` is required for generate and read
- optional `to_ts` overrides generated timestamp during generation
- only recent 3 days are supported
- write path overwrites the single row for `(profile_id=1, digest_date)`
- read/list paths auto-purge expired rows
- source post business time fallback is `as_of -> posted_at -> created_at`

## Weekly Digest

- `POST /weekly-digests/generate`
- `GET /weekly-digests`
- `GET /weekly-digests/dates`

Current semantics:
- `kind` is required and must be `recent_week`, `this_week`, or `last_week`
- optional `date` is the reference date; defaults to today
- optional `to_ts` overrides generated timestamp
- `recent_week` window is `[date-6d, date]`
- `this_week` starts from the latest Sunday through `date`
- `last_week` is the previous full Sunday-Saturday window
- stale rows for the same `kind` are auto-purged when their `anchor_date` no longer matches the currently expected anchor
- current weekly AI prompt emphasizes near-term market impact, explicit consensus/disagreement rules, and author-grouped short-term trading observations

## Portfolio Advice

- `POST /portfolio/advice`

Request body:
- `user_goal`
- `holdings[]`
- each holding supports `asset_id`, `symbol`, optional `name`, optional `market`
- optional `holding_reason_text`
- optional `sell_timing_text`
- `support_citations[]` and `risk_citations[]`

Citations support:
- required `source_url`, `summary`
- optional `extraction_id`, `author_handle`, `stance`, `horizon`, `confidence`, `as_of`

Response semantics:
- `status=ok`: AI aggregation succeeded
- `status=skipped_no_api_key`: no `OPENAI_API_KEY`, deterministic fallback returned
- `status=failed`: AI call failed, fallback returned with `error`

Persistence:
- request-time only
- dedicated replay/version table is `Not Implemented`

## Admin Repair And Cleanup

- `DELETE /admin/extractions/pending`
- `DELETE /admin/kols/{kol_id}`
- `DELETE /admin/assets/{asset_id}`
- `DELETE /admin/digests`
- `DELETE /admin/digests/{digest_id}`
- `DELETE /admin/extractions/{extraction_id}`
- `DELETE /admin/kol-views/{kol_view_id}`
- `POST /admin/extractions/refresh-wrong-extracted-json`
- `POST /admin/extractions/recompute-statuses`
- `POST /admin/fix/approved-missing-views`
- `POST /admin/extractions/cleanup-json`

Admin guardrail:
- destructive endpoints require `confirm=YES`
- dry-run flags are supported on the repair endpoints above
