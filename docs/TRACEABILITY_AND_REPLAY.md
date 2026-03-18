# Traceability And Replay

Authoritative

TL;DR
- 所有结论都必须能回链到原始帖子和 extraction 证据。
- Daily digest 与 weekly digest 都是覆盖式回放，不是历史多版本累积。
- 运行命令看 `docs/RUNBOOK.md`，本文只定义语义。

## Evidence Chain

### 1. `raw_posts`
- 原始来源字段：`platform`, `external_id`, `url`, `content_text`, `posted_at`, `raw_json`
- 这是最底层来源证据。

### 2. `post_extractions`
- 关联键：`raw_post_id`
- 审计字段：`prompt_version`, `prompt_text`, `prompt_hash`, `raw_model_output`, `parsed_model_output`
- 结果字段：`status`, `extracted_json`, `last_error`
- `parsed_model_output` 存储为有序 `JSON`
- `extracted_json.meta` 为精简形态，只保留影响自动审核、失败解释、回放状态的关键字段

### 3. `kol_views`
- 观点物化字段：`kol_id`, `asset_id`, `stance`, `horizon`, `confidence`, `summary`, `source_url`, `as_of`
- 这是 dashboard、资产页和 digest 的直接观点证据层。

### 4. `daily_digests`
- 回放键：`profile_id`, `digest_date`
- 主要内容：`post_summaries`, `ai_input_by_author`, `ai_analysis`, `metadata`

### 5. `weekly_digests`
- 回放键：`profile_id`, `report_kind`, `anchor_date`
- 主要内容：`post_summaries`, `ai_input_by_day`, `ai_analysis`, `metadata`

### 6. `portfolio/advice`
- 当前是请求级证据聚合，不落独立回放表
- 证据来自 `holdings[].support_citations[]` 与 `holdings[].risk_citations[]`
- 当前支持字段：`source_url`, `summary`，以及可选的 `extraction_id`, `author_handle`, `stance`, `horizon`, `confidence`, `as_of`

## Daily Digest Replay

- 唯一键是 `(profile_id, digest_date)`。
- 当前 public API 固定使用系统 profile `id=1`。
- `POST /digests/generate` 对同一日覆盖生成，不做多版本累积。
- `GET /digests` 通过 `date` 回放。
- `GET /digests/{digest_id}` 通过主键回放。
- `GET /digests/dates` 返回当前可回放日期。
- 保留窗口固定为最近 3 天；读和列举路径会自动清理过期行。

## Weekly Digest Replay

- 唯一键是 `(profile_id, report_kind, anchor_date)`。
- 当前 public API 固定使用系统 profile `id=1`。
- `POST /weekly-digests/generate` 对同一 `kind + anchor_date` 覆盖生成。
- `GET /weekly-digests` 通过 `kind + anchor_date` 回放。
- `GET /weekly-digests/dates` 返回对应 `kind` 的可回放锚点日期。
- 读、列举、生成路径会自动清理锚点已失效的陈旧记录。

## Time Field Policy

Digest 使用的业务时间回退顺序：
- `as_of`
- `posted_at`
- `created_at`

这条顺序同时影响 digest 摘要流入窗与排序语义。

## Re-Extract Replacement Semantics

- `POST /extractions/{id}/re-extract` 不是简单追加。
- 当新的 extraction 成为有效 AI 结果时，同一 `raw_post` 的旧 extraction 行可能被删除。
- 仅被这些旧行引用的 `kol_views` 也会被删除。
- 如果新的 extraction 仍属 failed semantics，旧行会保留。

## Parse Failure Semantics

- 模型返回内容但解析失败时，仍然会落一条 extraction。
- 该记录数据库 `status` 仍可能是 `pending`。
- 错误被记录到 `last_error` 与 parse 相关 meta。
- 进度统计与重试筛选会把这类 pending 行按 failed semantics 对待。

## Verification Rule

- 任何“可追溯”“可回放”的文档表述都必须与本文一致。
- 命令不要在本文重复维护，统一引用 `docs/RUNBOOK.md`。

## Not Implemented

- Event reminder entity and replay lifecycle.
- Portfolio advice historical persistence and replay.
