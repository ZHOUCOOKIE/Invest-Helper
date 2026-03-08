# Traceability And Replay

Authoritative

TL;DR
- 所有输出都必须可回链证据。
- Digest 当前以唯一 `profile_id + digest_date` 单条覆盖存储（当前 API 固定使用 `profile_id=1`）。
- 回放操作命令以 `docs/RUNBOOK.md` 为准；本文定义语义与约束。

## Evidence Chain (Storage-Level)

1. `raw_posts`
- 原始来源字段：`platform`, `external_id`, `url`, `content_text`, `posted_at`, `raw_json`

2. `post_extractions`
- 关联：`raw_post_id`
- 审计：`prompt_version`, `prompt_text`, `prompt_hash`, `raw_model_output`, `parsed_model_output`
- Prompt 组装来源：`apps/api/services/prompts/extraction_prompt.py`（`prompt_version` 当前为 `extract_v1`）
- 状态：`status`, `extracted_json`, `last_error`
- `parsed_model_output` 存储类型：`JSON`（保序，非 `JSONB`）
- `parsed_model_output` 当前规范键顺序：`as_of, source_url, islibrary, hasview, asset_views, library_entry`

3. `kol_views`
- 观点证据：`source_url`, `kol_id`, `asset_id`, `horizon`, `stance`, `confidence`, `as_of`

4. `daily_digests`
- 回放核心：`profile_id`, `digest_date`, `content`
- `content` 关键字段：`post_summaries`, `ai_input_by_author`, `ai_analysis`, `metadata`

5. `weekly_digests`
- 回放核心：`profile_id`, `report_kind`, `anchor_date`, `content`
- `content` 关键字段：`post_summaries`, `ai_input_by_day`, `ai_analysis`, `metadata`

6. `portfolio/advice`（请求级）
- 证据载体：`holdings[].support_citations[]` / `holdings[].risk_citations[]`
- 证据关键字段：`source_url`, `summary`（可选 `extraction_id/author_handle/stance/horizon/confidence/as_of`）
- 当前语义：请求即计算，不写入独立回放实体（Not Implemented: advice 历史版本化存储）

## Replay Semantics

- 唯一键：`(profile_id, digest_date)`。
- `POST /digests/generate` 对同一 `profile_id + digest_date` 覆盖重生成。
- `POST /digests/generate` 当前有效参数：`date`、`to_ts`（`profile_id` 当前未暴露，固定为 `1`）。
- `GET /digests` 读取该日系统 profile 的单条日报（参数：`date`）。
- `GET /digests/{digest_id}` 支持按主键直接回放。
- `GET /digests/dates` 返回系统 profile 可回放日期集合。
- 日报保留窗口固定为近 `3` 天（今天及往前 `2` 天）；超出窗口数据会在读/列出路径自动清理。
- `daily_digests.version` 当前写入路径固定为 `1`，不做多版本累积。
- 周报回放：
  - `POST /weekly-digests/generate` 按 `(profile_id, report_kind, anchor_date)` 覆盖重生成。
  - `GET /weekly-digests` 按 `kind + anchor_date` 回放。
  - `GET /weekly-digests/dates` 返回对应 `kind` 的可回放锚点日期集合。

## Time-Field Policy

Digest 摘要流排序时间字段回退顺序：
- `as_of`（preferred）
- `posted_at`
- `created_at`

## Verification Rule

- 任何文档提到“可追溯/可回放”时，必须与本文一致。
- 任何运行命令示例必须引用 `docs/RUNBOOK.md`，避免重复版本。

## Extraction Failure Semantics

- 当模型返回内容但解析失败（例如 invalid JSON / truncated output after retries）时，仍会创建 extraction 记录。
- 该记录数据库 `status` 保持 `pending`，并写入 `last_error` 与相关 parse meta。
- 运行时分类会将其归为 failed 语义（用于进度统计、重试筛选）。

## Not Implemented

- Event reminder entity and replay lifecycle.
