# API Contract

## Extraction Prompt SSOT
- Prompt rules and template live only in `apps/api/services/prompts/extraction_prompt.py`.
- Current prompt body is Chinese instructions and currently renders these input fields into the single user message: `author_handle`, `url`, `posted_at`, `content_text`.
- Runtime uses `render_prompt_bundle(...)` only, and sends a single `messages` item (`role=user`) from `apps/api/services/extraction.py`.
- `platform` still exists in the render function signature, but the current template does not emit a `platform:` line.
- `lang` prompt field is removed (it was only a language hint and not used by business logic/audit/replay).

## Model Output JSON (required core keys)
Top-level model output must include these 6 core keys:
- `as_of` (`YYYY-MM-DD`)
- `source_url` (string; expected to use input post URL)
- `islibrary` (`0|1`, integer)
- `hasview` (`0|1`, integer)
- `asset_views` (array, can be empty)
- `library_entry` (`null` or object)

Canonical key order (read/write):
- top-level: `as_of, source_url, islibrary, hasview, asset_views, library_entry` (`extracted_json` may append `meta`)
- `asset_views[*]`: `symbol, market, stance, horizon, confidence, summary`

## Field Rules
- Enum strict output (model must output exact enum values; server does not run keyword/synonym alias normalization for these fields):
  - `market`: `CRYPTO|STOCK|ETF|FOREX|OTHER` (legacy auto enum removed)
  - `stance`: `bull|bear|neutral`
  - `horizon`: `intraday|1w|1m|3m|1y`
- `asset_views[*]`:
  - exact shape: `{symbol, market, stance, horizon, confidence, summary}`
  - `confidence`: integer; normalize keeps only `>=70` (prompt text asks model to output `80..100`)
  - `summary`: Chinese only, and prompt requires a short reason chain
  - direct mention is strict: the post text itself must contain the asset ticker/name/symbol string
  - `hasview=1` requires a real forward-looking or action-oriented investment claim; hypothetical, educational, sarcastic, meme-only content should stay `hasview=0`
- `library_entry`:
  - when `islibrary=0`: must be `null` (normalize forces null)
  - when `islibrary=1`: must be `{tag,summary}`
  - `tag`: enum `macro|industry|thesis|strategy|risk|events`
  - `summary`: Chinese only and must equal exact literal `测试`

## Normalize Guarantees
- parse/repair keeps existing behavior (BOM/code fence/outer object/outermost JSON extraction)
- no alias-map keyword/synonym normalization for `market/stance/horizon`
- `asset_views` with `confidence<70` are dropped
- `hasview` is recomputed from final `asset_views` (`1` if non-empty else `0`)
- `islibrary=1` + invalid/missing `library_entry` => downgrade to `islibrary=0`, `library_entry=null`
- `extracted_json.meta` is stored in compact form:
  - keep auto-review result fields such as `auto_approved/auto_rejected/auto_review_reason/auto_review_threshold/model_confidence/auto_policy_applied`
  - keep exceptional replay/status fields only when applicable, such as `library_downgraded(_reason)`, `parse_error(_reason)`, `fallback_reason`, `dummy_fallback`, `retry_source`, `truncated`, `raw_truncated`
  - drop runtime/debug counters and null/false placeholders during read/write normalization
  - if nothing meaningful remains, `meta` is omitted entirely on read/write normalization
- `parsed_model_output` is persisted as ordered JSON (DB type `JSON`, not `JSONB`) and cleanup strips `meta` from that stored parsed payload

## Auto Review
- if `hasview=0`: auto reject (`hasview_zero`)
- auto approve requires `hasview=1` and confidence path meeting threshold (`80`)
- `meta.auto_policy_applied` is recorded (current value: `threshold_asset`)
- reject reason key is `meta.auto_review_reason` (legacy `auto_reject_reason` is removed)
- user-triggered `POST /extractions/{id}/re-extract` goes through the same standard auto-review path after runtime validation succeeds
- there is no special keep-pending bypass for `islibrary=1` user-triggered rows; if the normalized payload yields `hasview=0`, it is auto rejected by the standard rules

## X Import And Extract Jobs (Current)
- `POST /ingest/x/import`
  - returns:
    - `inserted_raw_post_ids`
    - `dedup_existing_raw_post_ids`
    - `pending_failed_dedup_count`
    - `pending_failed_dedup_ids`
    - `pending_failed_reason_breakdown`
  - dedup raw posts whose latest extraction is still failed semantics (`status=pending` + error semantics) are surfaced via `pending_failed_dedup_*`
  - dedup raw posts that already have successful / approved / rejected terminal results are not surfaced in `pending_failed_dedup_ids`
- `POST /extract-jobs`
  - creates an async extract job for a deduplicated list of `raw_post_ids`
  - in-flight jobs reuse the existing job when all requested ids are already covered by queued/running work
- `GET /extract-jobs/{job_id}`
  - returns async job counters including:
    - `ai_call_used`
    - `openai_call_attempted_count`
    - `max_concurrency_used`
    - `last_error_summary`
  - UI should prefer `ai_call_used` as the uploaded-to-AI count, then fall back to `openai_call_attempted_count`, then `success_count + failed_count`
- `GET /ingest/x/progress`
  - returns current global or per-author counts for `extracted_success/pending/failed/no_extraction`
- `POST /ingest/x/retry-failed`
  - retries latest failed-semantics rows only, skipping already-successful or terminal-reviewed raw posts
- `POST /ingest/x/retry-pending-all`
  - creates one async job covering all current pending raw posts that still match enabled X KOL scope

## Parse Failure Semantics
- when model returns content but parse fails (for example invalid JSON/truncated output after retries), extraction record is still created as `pending`
- failure reason is written to `last_error` and/or `meta.parse_error(_reason)`
- state classifier treats such `pending` record as failed semantics for progress/retry flows

## Extractions List Ordering (Current)
- `GET /extractions`
  - list order is deterministic by business post start time desc:
    - primary: `raw_post.posted_at`
    - fallback: `post_extractions.created_at`
  - tie-breaker: extraction `id` desc
- `GET /extractions/stats`
  - returns current repository-level counts for:
    - `raw_posts_count`
    - `post_extractions_count`
    - `duplicate_raw_post_count` (how many `raw_post` currently have multiple extraction rows)
- `POST /extractions/{id}/re-extract`
  - creates a new replacement extraction for the same `raw_post`
  - once the new AI result is accepted by runtime validation, it immediately goes through the standard auto-review flow
  - when the new extraction is a valid AI result (not failed parse/request/dummy fallback), it replaces all prior extraction results for that `raw_post`
    - deletes all older extraction rows for that `raw_post`, including previously manual-approved rows
    - deletes related `kol_views` that are referenced only by those deleted rows
    - if the replacement row is still `pending`, `raw_post.review_status` is reset to `unreviewed`
  - if the new extraction itself is failed semantics, old rows are kept

## Admin Repair Endpoints (Current)
- `POST /admin/extractions/refresh-wrong-extracted-json`
  - scans approved rows whose `auto_applied_kol_view_ids` exist but `extracted_json.asset_views` is empty/missing
  - rebuilds `asset_views` from referenced `kol_views`
  - rewrites `meta` into compact form if meaningful keys remain, otherwise removes `meta`
- `POST /admin/extractions/cleanup-json`
  - rewrites historical `extracted_json` into the current normalized read shape
  - rewrites `parsed_model_output` into ordered core JSON and removes parsed `meta`
- `POST /admin/extractions/recompute-statuses`
  - recomputes current auto-review status using normalized `extracted_json`, optionally preferring `parsed_model_output` when it yields a stronger decision

## Backward Read Tolerance
- read path tolerates old records and legacy fields without rewriting historical approval status
- legacy `library_tags` is ignored on read

## Daily Digest API (Current)
- `POST /digests/generate`
  - required query:
    - `date` (`YYYY-MM-DD`)
  - optional query:
    - `to_ts` (optional ISO datetime, used as `generated_at`)
  - behavior:
    - only supports recent `3` days (today and previous `2` days); out-of-window date returns `400`
    - regenerate and overwrite the single digest row for the same `profile_id + digest_date`
    - current endpoint writes with fixed system profile (`profile_id=1`)
    - `version` field is persisted but current write path keeps it at `1`
    - digest window is fixed to `[date-1d 00:00 UTC, date+1d 00:00 UTC)`
    - source post time fallback priority: `as_of -> posted_at -> created_at`
- `GET /digests`
  - required query: `date`
  - only recent `3` days are readable; older dates return `404`
  - returns the replay digest for fixed system profile (`profile_id=1`) on that date
- `GET /digests/dates`
  - returns replayable digest dates within recent `3` days for fixed system profile (`profile_id=1`) (desc)
  - request path auto-purges digests older than `3` days
- `GET /digests/{digest_id}`
  - replay by primary key (older-than-3-days records are auto-purged and return `404`)

## Weekly Digest API (Current)
- `POST /weekly-digests/generate`
  - required query:
    - `kind` (`recent_week|this_week|last_week`)
  - optional query:
    - `date` (`YYYY-MM-DD`, reference date; default today)
    - `to_ts` (optional ISO datetime, used as `generated_at`)
  - behavior:
    - uses weekly-digest specific AI prompt (daily and weekly prompt pipelines are separated)
    - AI input is aggregated by day (`ai_input_by_day`), each day carries a single date bucket
    - `recent_week`: `[date-6d, date]`
    - `this_week`: from latest Sunday to `date` (not necessarily full 7 days)
    - `last_week`: previous full Sunday-Saturday week
    - before write, stale rows for same `(profile_id, report_kind)` whose `anchor_date` is not current expected anchor are purged
    - overwrite by unique key `(profile_id, report_kind, anchor_date)` with `version=1`
- `GET /weekly-digests`
  - required query: `kind`, `anchor_date`
  - request path auto-purges stale rows for the same `report_kind` before lookup
  - returns replay digest for fixed system profile (`profile_id=1`)
- `GET /weekly-digests/dates`
  - required query: `kind`
  - request path auto-purges stale rows for the same `report_kind` before listing
  - returns replayable anchor dates (desc) for fixed system profile (`profile_id=1`)

## Portfolio Advice API (Current)
- `POST /portfolio/advice`
  - request body:
    - `holdings[]`:
      - `asset_id`, `symbol`, optional `name/market`
      - optional `holding_reason_text` (手写持仓理由)
      - optional `sell_timing_text` (手写卖出时机)
      - `support_citations[]` / `risk_citations[]`:
        - `source_url`, `summary` (required)
        - optional `extraction_id/author_handle/stance/horizon/confidence/as_of`
  - response:
    - `status`:
      - `ok`：AI聚合成功
      - `skipped_no_api_key`：未配置 `OPENAI_API_KEY`，返回规则版建议
      - `failed`：AI调用失败，返回规则版建议并带 `error`
    - `advice_summary`：组合级建议
    - `asset_advice[]`：逐资产建议与评价（含 `score/stance/suggestion/evaluation/key_risks/key_triggers`）
  - persistence:
    - current behavior is request-time compute only; no dedicated replay table for portfolio advice
