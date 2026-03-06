# API Contract

## Extraction Prompt SSOT
- Prompt rules and template live only in `apps/api/services/prompts/extraction_prompt.py`.
- Runtime uses `render_prompt_bundle(...)` only, and sends a single `messages` item (`role=user`) from `apps/api/services/extraction.py`.
- `lang` prompt field is removed (it was only a language hint and not used by business logic/audit/replay).

## Model Output JSON (required core keys)
Top-level model output must include these 6 core keys:
- `as_of` (`YYYY-MM-DD`)
- `source_url` (must equal input post URL)
- `islibrary` (`0|1`, integer)
- `hasview` (`0|1`, integer)
- `asset_views` (array, can be empty)
- `library_entry` (`null` or object)

## Field Rules
- Enum strict output (model must output exact enum values; server does not run keyword/synonym alias normalization for these fields):
  - `market`: `CRYPTO|STOCK|ETF|FOREX|OTHER` (legacy auto enum removed)
  - `stance`: `bull|bear|neutral`
  - `horizon`: `intraday|1w|1m|3m|1y`
- `asset_views[*]`:
  - exact shape: `{symbol, market, stance, horizon, confidence, summary}`
  - `confidence`: integer; normalize keeps only `>=70` (prompt text asks model to output `80..100`)
  - `summary`: Chinese only
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
- internal observability remains in `extracted_json.meta`

## Auto Review
- if `hasview=0`: auto reject (`hasview_zero`)
- auto approve requires `hasview=1` and confidence path meeting threshold (`80`)
- `meta.auto_policy_applied` is recorded (current values: `threshold_asset|no_auto_review_user_trigger`)

## Backward Read Tolerance
- read path tolerates old records and legacy fields without rewriting historical approval status
- legacy `library_tags` is ignored on read

## Daily Digest API (Current)
- `POST /digests/generate`
  - required query:
    - `date` (`YYYY-MM-DD`)
  - optional query:
    - `profile_id` (default `1`)
    - `to_ts` (optional ISO datetime, used as `generated_at`)
  - behavior:
    - regenerate and overwrite the single digest row for the same `profile_id + digest_date`
    - digest window is fixed to `[date-1d 00:00 UTC, date+1d 00:00 UTC)`
    - source post time fallback priority: `as_of -> posted_at -> created_at`
- `GET /digests`
  - required query: `date`
  - optional query: `profile_id` (default `1`)
  - returns the replay digest for this `profile_id + digest_date`
- `GET /digests/dates`
  - optional query: `profile_id` (default `1`)
  - returns all replayable digest dates for that profile (desc)
- `GET /digests/{digest_id}`
  - replay by primary key
