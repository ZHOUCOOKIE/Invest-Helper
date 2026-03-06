# STATUS

## Implemented
- Extraction prompt rules upgraded to strict 6-key JSON: `as_of/source_url/islibrary/assets/asset_views/library_entry`.
- Prompt SSOT is `apps/api/services/prompts/extraction_prompt.py` only.
- Structured schema and normalize logic aligned to `islibrary` contract.
- `assets` final shape is object array `{symbol, market}`.
- NoneAny sentinel enforced: `[{"symbol":"NoneAny","market":"OTHER"}]`.
- `market` enum is `CRYPTO|STOCK|ETF|FOREX|OTHER`; legacy auto enum value is removed.
- `stance/horizon/market` must be model-direct exact enum output; server does not apply alias-map keyword/synonym normalization.
- `asset_views` keeps only `confidence>=70`.
- Chinese summary validation is enforced for:
  - `asset_views[*].summary`
  - `library_entry.summary`
- Library boundary (A2) enforced with new contract:
  - `library_entry` final shape is `{confidence, tags, summary}`
  - `confidence` is required int in `0..100`; when `islibrary=1` it must be `>=70`
  - `tags` is required array length `1..2` with enum `macro|industry|thesis|strategy|risk|events`
  - invalid/missing `library_entry` downgrades to `islibrary=0`
- Auto review:
  - asset path keeps existing threshold flow
  - library path uses `library_entry.confidence` (`>=70` approve, `<70` reject)
  - writes `meta.auto_policy_applied`

## Not Implemented
- None.

## Notes
- Legacy read tolerance remains for historical records; old `library_tags` is ignored.
