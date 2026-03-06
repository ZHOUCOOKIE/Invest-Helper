# Traceability And Replay

Authoritative

TL;DR
- 所有输出都必须可回链证据。
- Digest 回放键为 `profile_id + digest_date + version`。
- 回放操作命令以 `docs/RUNBOOK.md` 为准；本文定义语义与约束。

## Evidence Chain (Storage-Level)

1. `raw_posts`
- 原始来源字段：`platform`, `external_id`, `url`, `content_text`, `posted_at`, `raw_json`

2. `post_extractions`
- 关联：`raw_post_id`
- 审计：`prompt_version`, `prompt_text`, `prompt_hash`, `raw_model_output`, `parsed_model_output`
- Prompt 组装来源：`apps/api/services/prompts/extraction_prompt.py`（`prompt_version` 当前为 `extract_v1`）
- 状态：`status`, `extracted_json`, `last_error`

3. `kol_views`
- 观点证据：`source_url`, `kol_id`, `asset_id`, `horizon`, `stance`, `confidence`, `as_of`

4. `daily_digests`
- 回放核心：`profile_id`, `digest_date`, `version`, `content`

## Replay Semantics

- 唯一键：`(profile_id, digest_date, version)`。
- `POST /digests/generate` 对同一 `profile_id + digest_date` 递增 `version`。
- `GET /digests` 不传 `version` 时返回该日该 profile 最新版本。
- `GET /digests/{digest_id}` 支持按主键直接回放。
- `GET /digests/dates?profile_id=...` 返回可回放日期集合。

## Time-Field Policy

Digest 时间字段回退顺序：
- `as_of`（preferred）
- `posted_at`
- `created_at`

输出元数据包含 `time_field_used`。

## Verification Rule

- 任何文档提到“可追溯/可回放”时，必须与本文一致。
- 任何运行命令示例必须引用 `docs/RUNBOOK.md`，避免重复版本。

## Not Implemented

- Event reminder entity and replay lifecycle.
