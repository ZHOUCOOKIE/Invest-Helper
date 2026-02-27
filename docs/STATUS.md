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
- 管理端修复与清理接口（admin endpoints）。

Serve
- Assets/KOLs/Profiles APIs。
- Dashboard（包含 clarity ranking 与 contributor evidence）。
- Digest 生成、读取、日期列表、按 profile+date 版本化回放。

Storage
- PostgreSQL + SQLAlchemy + Alembic。
- 核心实体：`raw_posts`, `post_extractions`, `kol_views`, `daily_digests`, profile 相关表。

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
