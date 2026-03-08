# STATUS

## Implemented
- Extraction prompt rules upgraded to strict 6-key JSON: `as_of/source_url/islibrary/hasview/asset_views/library_entry`.
- Prompt SSOT is `apps/api/services/prompts/extraction_prompt.py` only.
- Structured schema and normalize logic aligned to `islibrary` contract.
- Top-level contract is `as_of/source_url/islibrary/hasview/asset_views/library_entry`.
- Canonical key order is fixed:
  - top-level: `as_of,source_url,islibrary,hasview,asset_views,library_entry` (`meta` optional at tail)
  - `asset_views[*]`: `symbol,market,stance,horizon,confidence,summary`
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
  - reject reason key is `meta.auto_review_reason` (legacy `auto_reject_*` removed)
- Parse-failure semantics:
  - parse-failed extraction remains `pending` at row status
  - error is captured in `last_error`/`meta.parse_error(_reason)`
  - classifier treats these rows as failed semantics for progress/retry
- `parsed_model_output` DB type is `JSON` (ordered), not `JSONB`
- Meta normalization drops legacy/debug keys:
  - `auto_reject_reason`, `auto_reject_threshold`
  - `parse_unwrapped_extracted_json`, `parse_unwrapped_key`
  - `provider_detected`, `output_mode_used`, `parse_strategy_used`
  - `ruleset_version`, `extraction_mode`, `repaired`, `raw_len`
- Daily Digest replay contract:
  - single-row overwrite by `profile_id + digest_date`
  - generation API params are `date/to_ts` only (`profile_id` not exposed; current fixed `1`)
  - read API params are `date` only (`profile_id` not exposed; current fixed `1`)
  - retention window is fixed to recent `3` days (today and previous `2` days)
  - read/list paths auto-purge digests older than `3` days
  - `daily_digests.version` is currently written as `1`
- Weekly Digest replay contract:
  - weekly report kinds: `recent_week|this_week|last_week`
  - write key: `(profile_id, report_kind, anchor_date)` overwrite
  - `this_week` starts at latest Sunday; `last_week` is previous full Sunday-Saturday
  - weekly AI prompt pipeline is independent from daily digest
  - weekly AI input aggregation is by day (`ai_input_by_day`)
  - generate/read/list paths auto-purge stale rows whose `anchor_date` is not the expected current anchor for that `report_kind`
- Extractions listing:
  - `/extractions` response order is by business post time desc (`raw_post.posted_at`, fallback `extraction.created_at`), then by `id` desc
- Portfolio holdings advice:
  - added `POST /portfolio/advice` for AI aggregation on user-provided holdings context
  - request supports `holding_reason_text/sell_timing_text/support_citations/risk_citations`
  - when `OPENAI_API_KEY` is missing or AI request fails, API returns deterministic fallback advice (no hard failure)
- Web robustness updates:
  - daily/weekly digest pages retry short polling after generate-call proxy errors to recover eventual-success writes
  - weekly page shows request-aware API error context (`request_id`)
  - ingest page handles polling abort/unmount/manual-stop errors with explicit user-facing messages

## Not Implemented
- Event reminder entity and reminder lifecycle workflow.
- Portfolio advice 历史版本化存储与回放实体。

## Notes
- Legacy read tolerance remains for historical records; old `library_tags` is ignored.
- Prompt text currently asks model output `asset_views.confidence>=80`; runtime normalize threshold is `>=70`, and auto-review threshold is `>=80`.
