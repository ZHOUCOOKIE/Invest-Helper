# InvestPulse

[English](./README.md) | [简体中文](./README.zh-CN.md)

InvestPulse 是一个面向投研/交易场景的“证据可追溯”信号看板系统。  
系统将多源帖子流转成结构化观点，支持人工/自动审核，并提供按日期可回放的日报与周报（当前 API 固定系统 profile `id=1`）。

## UI Preview

![InvestPulse Dashboard Preview](./docs/images/dashboard-preview.png)

## 项目背景

传统“看帖做判断”的流程存在三个核心问题：
- 信息分散：观点散落在多个来源，难以统一聚合。
- 证据断链：结论与原始链接脱节，复盘时难以追责与回看。
- 回放困难：同一天的结论会被覆盖，难以稳定复现当时决策上下文。

InvestPulse 的设计目标是把“帖子 -> 抽取 -> 审核 -> 观点 -> Digest 回放”做成可审计的数据链路。

## 项目预期

- 建立可追溯信号生产线：任何观点都能回链到 `source_url/raw_posts/post_extractions`。
- 提供稳定回放能力：日报/周报按键值覆盖生成，支持确定性读取。
- 提升研究效率：将非结构化内容沉淀为 `asset_views` 与摘要输入，降低人工整理成本。
- 保持工程可维护：测试、迁移、校验流程标准化，支持本地与 CI 一致验收。

## 项目现状（按当前代码与未提交改动核对）

- 已实现 X 数据导入、抽取、审核、观点沉淀、日报/周报生成与回放。
- 抽取结果执行统一 normalize 合约：
  - 顶层键：`as_of/source_url/islibrary/hasview/asset_views/library_entry`
  - `asset_views` 仅保留 `confidence >= 70`，并据此回算 `hasview`
  - `islibrary=1` 且 `library_entry` 无效时自动降级 `islibrary=0`
- 自动审核规则已落地：
  - `hasview=0` 自动拒绝
  - 自动通过阈值路径为 `>=80`
  - 自动拒绝原因写入 `meta.auto_review_reason`
- 解析失败语义已固化：记录仍以 `pending` 存储，并在进度/重试分类中视作失败语义。
- 回放语义已固化：
  - 日报：近 `3` 天保留窗口，超窗在读/列举路径自动清理
  - 周报：按 `report_kind` 自动清理非当前锚点版本，避免同类历史锚点混入当前回放集合
- 列表排序已对齐业务时间：`/extractions` 按 `raw_post.posted_at` 降序（缺失回退 `created_at`）。
- 前端健壮性增强：
  - 日报/周报“生成后短暂查询失败”场景增加短轮询恢复
  - ingest 页面对手动/卸载中断轮询的错误提示更明确
  - 周报页 API 错误信息包含 `request_id`，便于排障

## 产品能力面

- API: FastAPI + SQLAlchemy + Alembic
- Web: Next.js dashboard/review/kols/digest/weekly-digest/portfolio pages
- Storage: PostgreSQL (dev + test), Redis

Main routes:
- `/dashboard`
- `/portfolio`
- `/ingest`
- `/extractions`
- `/assets`
- `/kols`
- `/digests/[date]`
- `/weekly-digests`
- `/health`

## 前景与未来发展

中短期方向（部分为规划项）：
- 抽取质量评估体系：引入更细粒度评测与回归基线，降低语义漂移风险。
- Profile 维度增强：在固定系统 profile 之外，逐步开放更完整的 profile 规则与回放隔离能力。
- 组合建议回放化：将 `POST /portfolio/advice` 从“请求即计算”扩展到“可版本化存储与回放”（Not Implemented）。
- 提醒与事件生命周期：补齐 event/reminder 实体与完整流程（Not Implemented）。
- 持续收敛遗留路径：对非核心/遗留能力（如 Reddit 相关遗留字段）保持只兼容不扩写，逐步清理。

## Quick Start

```bash
# 1) Infra
docker compose up -d db db_test redis

# 2) API
cd apps/api
uv sync
ENV=local DEBUG=true DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/investpulse uv run uvicorn main:app --reload
```

In another shell:

```bash
# 3) Web
cd apps/web
pnpm install
pnpm dev
```

Unified verification (required after changes):

```bash
DATABASE_URL_TEST=postgresql+asyncpg://postgres:postgres@localhost:5433/investpulse_test make verify
```

## Documentation Map (SSOT)

- [docs/INDEX.md](./docs/INDEX.md): documentation authority and navigation
- [docs/RUNBOOK.md](./docs/RUNBOOK.md): run and replay commands
- [docs/DEV_WORKFLOW.md](./docs/DEV_WORKFLOW.md): dev/test/lint/migrate/verify workflow
- [docs/API.md](./docs/API.md): API contract and examples
- [docs/TRACEABILITY_AND_REPLAY.md](./docs/TRACEABILITY_AND_REPLAY.md): evidence and replay semantics
- [docs/STATUS.md](./docs/STATUS.md): implemented capability ledger

## Architecture Snapshot

```text
X/Source Posts
   -> raw_posts
   -> extraction (prompt + model + normalize)
   -> post_extractions
   -> review/auto-review
   -> kol_views
   -> daily_digests (unique profile_id + digest_date, current API writes profile_id=1)
   -> dashboard / digest replay
```

## Roadmap
- Improve extraction quality controls and evaluation benchmarks.
- Add richer profile-level signal ranking and filtering.
- Strengthen digest explainability with clearer evidence summaries.
- Prepare event/reminder lifecycle capabilities (currently not implemented).
- Continue reducing legacy/non-core paths outside the X-centric workflow.

## License

MIT. See [LICENSE](./LICENSE).
