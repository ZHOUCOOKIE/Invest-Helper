# API

Authoritative

TL;DR
- 本文是 API 端点与示例请求的权威入口。
- 本地启动与回放流程以 `docs/RUNBOOK.md` 为准。
- 能力边界以 `docs/STATUS.md` 为准。

## Contract Entry

- OpenAPI JSON: `http://localhost:8000/openapi.json`
- Swagger UI: `http://localhost:8000/docs`

## Health

```bash
curl -s "http://localhost:8000/health" | jq
```

## Dashboard

```bash
curl -s "http://localhost:8000/dashboard?days=7&window=7d&limit=10&assets_window=24h&profile_id=1" | jq
```

Top-level fields include:
- `pending_extractions_count`
- `assets`
- `active_kols_7d`
- `clarity_ranking`
- `top_assets`

## Digest Endpoints

- Generate: `POST /digests/generate`
- Get latest/version by query: `GET /digests`
- List digest dates: `GET /digests/dates`
- Get by id: `GET /digests/{digest_id}`

Example (generate):
```bash
curl -s -X POST "http://localhost:8000/digests/generate?date=2026-02-24&days=7&profile_id=1" | jq
```

## Ingest Endpoints

- `POST /ingest/manual`
- `POST /ingest/x/convert`
- `POST /ingest/x/import`
- `POST /ingest/x/following/import`
- `GET /ingest/x/raw-posts/preview`
- `GET /ingest/x/progress`
- `POST /ingest/x/retry-failed`
- `POST /raw-posts`

Example (`/ingest/manual`):
```bash
curl -s -X POST "http://localhost:8000/ingest/manual" \
  -H "Content-Type: application/json" \
  --data '{
    "platform":"x",
    "author_handle":"alice",
    "url":"https://x.com/alice/status/1",
    "content_text":"BTC trend looks constructive"
  }' | jq
```

## Extraction And Review Endpoints

- `POST /raw-posts/{raw_post_id}/extract`
- `POST /raw-posts/extract-batch`
- `POST /extract-jobs`
- `GET /extract-jobs/{job_id}`
- `GET /extractions`
- `GET /extractions/{extraction_id}`
- `POST /extractions/{extraction_id}/approve`
- `POST /extractions/{extraction_id}/approve-batch`
- `POST /extractions/{extraction_id}/reject`
- `POST /extractions/{extraction_id}/re-extract`

Example (`/raw-posts/extract-batch`):
```bash
curl -s -X POST "http://localhost:8000/raw-posts/extract-batch" \
  -H "Content-Type: application/json" \
  --data '{"raw_post_ids":[1,2,3],"mode":"pending_only"}' | jq
```

Extraction contract (code-enforced):
- 顶层字段：`assets/stance/horizon/confidence/summary/source_url/as_of/asset_views/content_kind/library_entry`
- `assets` 最终始终为对象数组（`{symbol,name,market}`）；无具体标的时为 `[{symbol:"NoneAny",name:null,market:"OTHER"}]`
- `summary` 与 `asset_views[*].summary` 必须中文；若非中文，触发一次 summary 中文纠正重试（与 parse 重试共享额外预算=1，互斥）。
- 额外重试预算总计固定 1（`meta.extra_retry_budget_total=1`）：
  - text_json 截断重试（`truncated_output`）或 invalid_json 纠错重试（二选一）；
  - 或 summary 中文纠正重试；
  - 不可叠加。
- OpenRouter `text_json` 下，当 `parse_error_reason=invalid_json` 且额外预算未使用时，会触发一次 invalid_json 纠错重试；仍失败时 `last_error` 会进入 `parse_error_invalid_json_after_retry`。
- invalid_json 纠错重试提示会强制“只输出合法 JSON + 字符串必须正确转义 + 禁止未转义裸引号”。
- `content_kind`:
  - `asset`: 正常资产观点
  - `library`: 无具体标的但高价值的投资分析/方法论/风险/事件解读（用于 Library 主展示）
- `library_entry`（仅 `content_kind=library` 有效）：
  - 结构：`{confidence: int(0..100), tags: string[1..2], summary: string|null}`
  - `tags` 允许值：`macro/industry/thesis/strategy/risk/events`
  - `content_kind=library` 时必须满足：`confidence>=80`、`tags` 长度 `1..2` 且枚举合法
- 顶层 `library_tags` 已移除；唯一真相是 `library_entry.tags`。旧数据若仍含 `library_tags`，读取时会忽略，不再写入新 extraction。
- 若模型仍输出顶层 `library_tags`，normalize 会剥离并写 `meta.library_tags_stripped=true`。
- `content_kind=library` 且 `library_entry` 无效/缺失时，不走失败兜底：normalize 降级为 `content_kind=asset`，并写 `meta.library_downgraded=true` 与 `meta.library_downgrade_reason`：
  - `low_library_confidence`
  - `invalid_library_tags`
  - `invalid_library_shape`
- `content_kind=library` 时最终强制 `asset_views=[]`；若模型输出了非空 `asset_views`，服务端会 normalize 清空并写 meta（`library_asset_views_cleared/library_asset_views_original_count/library_asset_views_final_count`）
- `content_kind=library` 时最终强制 `assets=[NoneAny]`。
- `content_kind=asset` 时会执行资产收敛：
  - `asset_views` 仅保留 `confidence>=70`
  - `assets` 与最终 `asset_views` 同步（symbols 去重，顺序按 `asset_views`）
  - 若过滤后 `asset_views=[]`，则 `assets=[NoneAny]`
- `asset_views[*]` 必含：`symbol/stance/horizon/confidence/summary`。模型若输出 `reasoning/drivers`，服务端会忽略，不触发失败。
- OpenRouter 模式下走 `text_json`，并将解析与语言约束结果写入 `extracted_json.meta`（如 `provider_detected/output_mode_used/summary_language/*`）
- `event_tags` 不再作为输出契约字段；模型误输出会被 normalize 丢弃。
- symbol 范围与格式（prompt + normalize 约束）：
  - A股/港股：中文股票全名（不要代码）
  - 美股/海外股票：英文 ticker 或英文名（可在 `name` 带中文）
  - ETF/指数：常见代码或标准简称（如 `IGV`, `SPX/标普500`）
  - 加密：常见代码/交易对（如 `BTC/比特币`, `BTC/USDT`）
- `GET /extractions` 默认 latest-only（每个 `raw_post` 仅返回最新 extraction）；`show_history=true` 才返回历史。
- 读取兼容：旧记录缺字段时 API 渲染默认 `content_kind="asset"`、`library_entry=null`，且不改变原有 `status`；旧记录中的 `library_tags` 会被忽略。

Auto-review（code-enforced）：
- `trigger in {"auto","bulk"}` 执行阈值：
  - `content_kind="library"` 使用 `library_entry.confidence`；
  - `content_kind="asset"` 使用顶层 `confidence`；
  - 阈值均为 `>= 70` 通过，`<70` 拒绝。
- `trigger="user"` 不执行阈值自动审核。
- `content_kind="asset"` 且 `assets==[NoneAny]` 强制拒绝。
- `content_kind="library"`（仅当存在有效 `library_entry`）允许 `assets==[NoneAny]`，按阈值（auto/bulk）或 pending（user）。
- `POST /extractions/{id}/re-extract`（user trigger）会重新走 parse/normalize 并产生新的 extraction；在 latest-only 列表中成为该 `raw_post` 的最新记录。
- `extracted_json.meta.auto_policy_applied` 可观测策略值：
  - `threshold_asset`
  - `threshold_library`
  - `noneany_asset_forced_reject`
  - `no_auto_review_user_trigger`

## Extractor Status

- `GET /extractor-status`
- 用于查看当前运行时模型与 OpenRouter 基础地址（环境变量优先于 settings 默认）。

Example:
```bash
curl -s "http://localhost:8000/extractor-status" | jq
```

## Profile / Asset / KOL Endpoints

- Profiles: `GET /profiles`, `GET /profiles/{profile_id}`, `PUT /profiles/{profile_id}/kols`, `PUT /profiles/{profile_id}/markets`
- Assets: `GET /assets`, `POST /assets`, `POST /assets/upsert`, aliases endpoints
- KOLs: `GET /kols`, `POST /kols`
- Views: `POST /kol-views`, `GET /assets/{asset_id}/views`, feed/timeline endpoints

## Admin Endpoints (Dev/Ops)

- Extraction repair/cleanup under `/admin/extractions/*`
- Hard delete endpoints under `/admin/*`

## Not Implemented

- Event reminder scheduling/triggering API.
- Prediction-market integration API.
