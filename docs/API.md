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
- `assets` (non-empty array of `{symbol, market}`)
- `asset_views` (array, can be empty)
- `library_entry` (`null` or object)

## Field Rules
- Enum strict output (model must output exact enum values; server does not run keyword/synonym alias normalization for these fields):
  - `market`: `CRYPTO|STOCK|ETF|FOREX|OTHER` (legacy auto enum removed)
  - `stance`: `bull|bear|neutral`
  - `horizon`: `intraday|1w|1m|3m|1y`
- `assets`:
  - if no valid tradable target: `[{"symbol":"NoneAny","market":"OTHER"}]`
- `asset_views[*]`:
  - exact shape: `{symbol, market, stance, horizon, confidence, summary}`
  - `confidence`: integer; normalize keeps only `>=70`
  - `summary`: Chinese only
- `library_entry`:
  - when `islibrary=0`: must be `null` (normalize forces null)
  - when `islibrary=1`: must be `{confidence,tags,summary}`
  - `confidence`: `0..100`, and library valid gate is `>=70`
  - `tags`: length `1..2`, enum `macro|industry|thesis|strategy|risk|events`
  - `summary`: Chinese only

## Normalize Guarantees
- parse/repair keeps existing behavior (BOM/code fence/outer object/outermost JSON extraction)
- no alias-map keyword/synonym normalization for `market/stance/horizon`
- `asset_views` with `confidence<70` are dropped
- `assets=[]` becomes `[{"symbol":"NoneAny","market":"OTHER"}]`
- `islibrary=1` + invalid/missing `library_entry` => downgrade to `islibrary=0`, `library_entry=null`
- internal observability remains in `extracted_json.meta`

## Auto Review
- asset branch (`islibrary=0`): threshold logic based on normalized asset view confidence path
- library branch (`islibrary=1`): uses `library_entry.confidence`
  - `>=70` approved
  - `<70` rejected
- `meta.auto_policy_applied` is recorded (for example: `threshold_asset|threshold_library|noneany_asset_forced_reject|no_auto_review_user_trigger`)

## Backward Read Tolerance
- read path tolerates old records and legacy fields without rewriting historical approval status
- legacy `library_tags` is ignored on read
