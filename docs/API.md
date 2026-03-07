# API Contract

## Extraction Prompt SSOT
- Prompt rules and template live only in `apps/api/services/prompts/extraction_prompt.py`.
- Runtime uses `render_prompt_bundle(...)` only, and sends a single `messages` item (`role=user`) from `apps/api/services/extraction.py`.
- `lang` prompt field is removed (it was only a language hint and not used by business logic/audit/replay).

## Model Output JSON (required core keys)
Top-level model output must include these 6 core keys:
- `as_of` (`YYYY-MM-DD`)
- `source_url` (string; expected to use input post URL)
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
    - overwrite by unique key `(profile_id, report_kind, anchor_date)` with `version=1`
- `GET /weekly-digests`
  - required query: `kind`, `anchor_date`
  - returns replay digest for fixed system profile (`profile_id=1`)
- `GET /weekly-digests/dates`
  - required query: `kind`
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
