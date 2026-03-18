# STATUS

## Implemented

### Extraction
- Prompt SSOT is [extraction_prompt.py](/home/zhoucookie/code/investpulse/apps/api/services/prompts/extraction_prompt.py).
- Prompt currently renders `author_handle`, `url`, `posted_at`, `content_text` into one user message.
- Normalized extraction core keys are `as_of`, `source_url`, `islibrary`, `hasview`, `asset_views`, `library_entry`.
- Canonical key order is fixed:
  - top level: `as_of, source_url, islibrary, hasview, asset_views, library_entry`
  - `asset_views[*]`: `symbol, market, stance, horizon, confidence, summary`
- Enum constraints are enforced:
  - `market`: `CRYPTO|STOCK|ETF|FOREX|OTHER`
  - `stance`: `bull|bear|neutral`
  - `horizon`: `intraday|1w|1m|3m|1y`
- `asset_views` normalization keeps only rows with `confidence >= 70`.
- `hasview` is recomputed from the final normalized `asset_views`.
- `asset_views[*].summary` and `library_entry.summary` must pass Chinese-summary validation.
- Symbol normalization supports ASCII ticker forms and CJK asset names; invalid symbols are dropped.
- `islibrary=1` with invalid or missing `library_entry` is downgraded to `islibrary=0` instead of failing the row.
- Current library contract is still strict and test-shaped:
  - `library_entry` shape is `{tag, summary}`
  - `tag` enum is `macro|industry|thesis|strategy|risk|events`
  - `summary` must equal exact literal `测试`
- `parsed_model_output` is stored as ordered DB `JSON`.

### Review And Failure Semantics
- `hasview=0` auto-rejects.
- Auto-approve requires `hasview=1` and confidence threshold `>= 80`.
- Auto-review writes compact meta such as `auto_policy_applied`, `auto_review_reason`, `auto_review_threshold`, `model_confidence`.
- Parse-failed model output is still persisted as an extraction row with DB `status=pending`.
- Retry/progress logic treats parse-failed pending rows as failed semantics.
- `POST /extractions/{id}/re-extract` is replacement-style:
  - a valid new AI result can delete older extraction rows for the same `raw_post`
  - `kol_views` referenced only by deleted rows are also removed
  - if the new extraction is still failed semantics, old rows remain

### Ingest And Jobs
- X import, convert, preview, following-import, batch extract, async extract jobs, retry-failed, and retry-pending-all are implemented.
- `/ingest/x/import` returns `pending_failed_dedup_count`, `pending_failed_dedup_ids`, and `pending_failed_reason_breakdown`.
- `/extract-jobs/{job_id}` exposes `ai_call_used`, OpenAI attempt counters, and concurrency stats.

### Views, Dashboard, And Assets
- Assets, KOLs, KOL asset summaries, asset views, asset feed, asset timeline, and post-detail endpoints are implemented.
- Dashboard aggregates pending extraction queue, top assets, clarity, active KOLs, recent extraction stats, and latest views.

### Daily And Weekly Digests
- Daily digest replay is unique by `(profile_id, digest_date)` and current API uses the system profile `id=1`.
- Daily digest retention window is recent 3 days; read/list paths purge expired rows.
- Weekly digest replay is unique by `(profile_id, report_kind, anchor_date)`.
- Weekly digest kinds are `recent_week`, `this_week`, `last_week`.
- Weekly digest stale rows are auto-purged when their anchor no longer matches the expected current anchor for that kind.
- Weekly digest AI input is aggregated by day as `ai_input_by_day`.
- Weekly digest AI prompt is independent from daily digest AI prompt.
- Current weekly AI prompt emphasizes:
  - focus on intraday and next `1w` to `1m`
  - `3m+` views only as brief tail context
  - explicit consensus vs disagreement criteria
  - `trading_observations` grouped by author when short-term trade actions exist

### Portfolio Advice
- `POST /portfolio/advice` is implemented as request-time AI aggregation over user-supplied holdings and citations.
- When `OPENAI_API_KEY` is missing, the API returns deterministic fallback advice with `status=skipped_no_api_key`.
- When the AI call fails, the API returns fallback advice with `status=failed` instead of hard-failing the request.

### Local Tooling
- `scripts/investpulse` starts and stops API and Web in background, runs API migrations on start, writes logs to `logs/`, and manages PID files in `.runtime/`.
- Root `make verify` runs migrations, ruff, lint, API tests, web tests, and root-level tests.

## Not Implemented
- Event reminder entity and reminder lifecycle workflow.
- Portfolio advice historical persistence and replay entity.
- Public profile-management API endpoints. Profile tables and server-side rules exist, but no public route family is exposed.

## Legacy / Boundary Notes
- Historical records that still contain legacy `library_tags` are tolerated and ignored on read.
- Reddit-related remnants are legacy only and are not an active product direction.
