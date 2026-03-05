# Status

Authoritative

TL;DR
- 本文是代码事实级能力账本。
- 只记录“已实现/未实现/非目标”，不承载运行步骤。
- 最后核对日期：2026-02-27。

## Implemented Capability Map

Ingest
- `POST /ingest/manual`
- `POST /ingest/x/convert`
- `POST /ingest/x/import`
- `POST /ingest/x/following/import`
- `GET /ingest/x/raw-posts/preview`
- `GET /ingest/x/progress`
- `POST /ingest/x/retry-failed`
- `POST /raw-posts`

Extract / Review
- 单条、批量、异步任务抽取。
- 审核：approve/reject/re-extract/approve-batch。
- 抽取审计元数据持久化（prompt/model IO/tokens/latency）。
- Prompt 输出约束已 enforce：JSON-only 指令、schema 校验/normalize、OpenRouter `text_json` 解析修复。
- summary 中文约束已 enforce：检测（顶层 `summary` + `asset_views[*].summary`）+ 一次纠正重试，失败写入 `last_error` 与 `extracted_json.meta`。
- text_json `invalid_json` 纠错重试已实现：当 `parse_error_reason=invalid_json` 且额外预算未使用时，触发一次纠错重试；失败写 `parse_error_invalid_json_after_retry`。
- 额外重试预算总计=1 已 enforce：截断重试 / invalid_json 重试 / summary 中文纠正重试互斥，不可叠加。
- invalid_json 重试提示已收敛：强制合法 JSON、字符串正确转义、禁止未转义裸引号。
- `assets` 归一化已 enforce：最终统一为对象数组；字符串数组会转为 `{symbol,name:null,market:"AUTO"}` 并写 meta。
- 输出字段收敛已实现：不再要求 `reasoning/event_tags`；`asset_views` 最终仅保留 `symbol/stance/horizon/confidence/summary`，模型误输出 `reasoning/drivers/event_tags` 会被忽略。
- Prompt 顶层输出收敛已实现：system hard rule 明确禁止顶层 `stance/horizon/confidence/summary/reasoning/event_tags`（仅允许在 `asset_views` 中出现）。
- symbol 范围约束已实现（prompt 层）：A股/港股中文全名；美股 `ticker/中文名`；ETF/指数；加密；黄金/原油。
- `NoneAny` 规则已 enforce：
  - `assets=[NoneAny]` + `content_kind=asset` -> 自动拒绝；
  - `content_kind=library` 允许走阈值自动审核；
  - `NoneAny` 与其他 symbol 混合 -> `last_error=noneany_mixed_with_symbols`。
- `content_kind/library_entry` 已实现（A2 规则）：
  - `content_kind` 支持 `asset|library` 作为主展示类型；
  - `library_entry={confidence,tags,summary}` 是 `content_kind=library` 的唯一有效入口。
- `library_entry` 强约束已实现：
  - `confidence>=80` 且 `tags` 长度 `1..2` 且枚举合法才保留；
  - 不满足则 normalize 清空为 `null`，并写 `meta.library_entry_dropped/library_entry_drop_reason`（不会因该项直接失败）。
- `content_kind=library` 降级策略已实现：
  - 若 `library_entry` 缺失/无效，则 normalize 降级为 `content_kind=asset`；
  - 写 `meta.library_downgraded=true` 与 `meta.library_downgrade_reason`（`low_library_confidence/invalid_library_tags/invalid_library_shape`）。
- Library 清空策略已实现：`content_kind=library` 时 `asset_views` 最终强制 `[]`，并写 meta：`library_asset_views_cleared/library_asset_views_original_count/library_asset_views_final_count`。
- Asset 收敛策略已实现：`content_kind=asset` 时只保留 `confidence>=70` 的 `asset_views`，并将 `assets` 同步为相同 symbols（去重同序）；过滤后为空则 `assets=[NoneAny]`。
- Auto-review 策略可观测：`meta.auto_policy_applied`（`threshold_asset|threshold_library|noneany_asset_forced_reject|no_auto_review_user_trigger`）。
- Auto-review 阈值源已实现：`content_kind=library` 使用 `library_entry.confidence`，`content_kind=asset` 使用顶层 `confidence`。
- 读取向后兼容已实现：旧版 `extracted_json` 缺失 `content_kind/library_entry` 时，API 渲染默认值且不改变既有 `approved/rejected` 状态；旧数据含 `library_tags` 会被忽略。
- 抽取运行时并发/节流已 enforce：`max_concurrency`、`max_rpm`、`batch_size`、`batch_sleep_ms`、retry backoff。
- 管理端修复与清理接口（admin endpoints）。

Serve
- Assets/KOLs/Profiles APIs。
- Dashboard（包含 clarity ranking 与 contributor evidence）。
- Digest 生成、读取、日期列表、按 profile+date 版本化回放。

Storage
- PostgreSQL + SQLAlchemy + Alembic。
- 核心实体：`raw_posts`, `post_extractions`, `kol_views`, `daily_digests`, profile 相关表。
- 测试库防污染护栏已实现：tests 仅允许 `DATABASE_URL_TEST`/`DATABASE_URL` 指向含 `test` 的库/Schema（否则 fail-fast）。

User Config
- Profile KOL 权重与启停。
- Profile market filters。

Observability
- `/health`
- request id middleware
- extraction status/job counters

## Not Implemented

- Event calendar entity。
- Reminder scheduling/triggering。
- Notification delivery channels。
- Prediction-market integration。

## Non-goal / Legacy

- Reddit 不是当前核心目标流水线。
- 相关遗留字段/UI 文案仅按“待移除遗留”处理。
