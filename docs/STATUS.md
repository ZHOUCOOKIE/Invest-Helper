# STATUS

## Implemented
- Extraction prompt rules upgraded to strict 6-key JSON: `as_of/source_url/islibrary/hasview/asset_views/library_entry`.
- Prompt SSOT is `apps/api/services/prompts/extraction_prompt.py` only.
- Structured schema and normalize logic aligned to `islibrary` contract.
- Top-level contract is `as_of/source_url/islibrary/hasview/asset_views/library_entry`.
- `market` enum is `CRYPTO|STOCK|ETF|FOREX|OTHER`; legacy auto enum value is removed.
- `stance/horizon/market` must be model-direct exact enum output; server does not apply alias-map keyword/synonym normalization.
- `asset_views` keeps only `confidence>=70`.
- Chinese summary validation is enforced for:
  - `asset_views[*].summary`
  - `library_entry.summary`
- Library boundary current contract:
  - `library_entry` final shape is `{tag, summary}`
  - `tag` enum: `macro|industry|thesis|strategy|risk|events`
  - `library_entry.summary` must be exact `测试`
  - invalid/missing `library_entry` downgrades to `islibrary=0`
- Auto review:
  - `hasview=0` auto reject
  - auto approve requires `hasview=1` + threshold flow (`80`)
  - writes `meta.auto_policy_applied`
- Daily Digest replay contract:
  - single-row overwrite by `profile_id + digest_date`
  - generation API params are `date/profile_id/to_ts` only
  - read API params are `date/profile_id` only

## Not Implemented
- None.

## Notes
- Legacy read tolerance remains for historical records; old `library_tags` is ignored.
- Prompt text currently asks model output `asset_views.confidence>=80`; runtime normalize threshold is `>=70`, and auto-review threshold is `>=80`.
