# InvestPulse

Authoritative

TL;DR
- InvestPulse 从帖子中抽取结构化投资观点，提供可追溯、可回放的看板与 Daily Digest。
- 文档单一入口：`docs/INDEX.md`。
- 统一验收命令：`make verify`。

## Product Scope (Code-Verified)

Implemented
- X 导入与原始帖子入库（manual/convert/import/following/retry）。
- 抽取与审核流程（single/batch/async jobs，approve/reject/re-extract）。
- OpenRouter 抽取默认模型：`deepseek/deepseek-v3.2`（可被环境变量 `OPENAI_MODEL` 覆盖）。
- Prompt 限制已在代码 enforce：JSON-only 输出、OpenRouter `text_json` 模式、reasoning 中文检测与一次纠正重试。
- 抽取并发与节流已在批量/异步任务生效：`max_concurrency` + `max_rpm` + `batch_size` + `batch_sleep_ms` + retry backoff。
- Profile 规则（KOL 权重、market 过滤）。
- 按 `profile_id + digest_date + version` 的 Digest 版本化生成与回放。
- 证据链字段贯通 `raw_posts -> post_extractions -> kol_views -> daily_digests`。

Not Implemented
- Event reminder scheduling/triggering。
- Prediction-market integration。

Non-goal (current)
- Reddit 不是当前核心产品流水线。

## Start Here

- 本地运行与回放：`docs/RUNBOOK.md`
- 开发/测试/lint/迁移：`docs/DEV_WORKFLOW.md`
- API 端点与示例：`docs/API.md`
- 可追溯与回放策略：`docs/TRACEABILITY_AND_REPLAY.md`
- 能力边界账本：`docs/STATUS.md`
- 术语：`docs/GLOSSARY.md`
