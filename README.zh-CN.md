# InvestPulse

[English](./README.md) | [简体中文](./README.zh-CN.md)

InvestPulse 是一个基于多源帖子抽取的投资信号看板系统，强调证据可追溯与 Digest 可回放。  
系统将非结构化帖子转成结构化信号，并支持按 `digest_date` 做日报生成与回放（当前 API 使用系统 profile `id=1`）。

## 产品价值

- 所有关键输出可回链证据来源，便于审计与复盘。
- 将帖子流转化为标准化 `asset_views`，提升可消费性。
- 支持 profile 维度的观点聚合与按日期 Digest 回放。
- 提供可落地的本地运行与统一验收流程。

## 当前实现（以代码为准）

- 已支持 X 流程下的导入与原始帖子入库。
- 已支持单条、批量、异步抽取流程。
- Prompt 单一事实来源在 `apps/api/services/prompts/extraction_prompt.py`。
- 运行时归一化协议：`as_of/source_url/islibrary/hasview/asset_views/library_entry`。
- `market/stance/horizon` 使用严格枚举。
- 自动审核逻辑：
  - `hasview=0` 自动拒绝
  - `hasview=1` 且满足置信度路径走阈值审核（`80`）
- 证据链贯通 `raw_posts -> post_extractions -> kol_views -> daily_digests`。
- Digest 按唯一 `profile_id + digest_date` 行覆盖重生成并回放（当前写入路径使用 `profile_id=1`）。

## 产品界面与技术栈

- API：FastAPI + SQLAlchemy + Alembic
- Web：Next.js（dashboard/review/kols/digest/weekly-digest/portfolio）
- 存储：PostgreSQL（dev/test）+ Redis

主要页面路由：
- `/dashboard`
- `/portfolio`
- `/ingest`
- `/extractions`
- `/assets`
- `/kols`
- `/digests/[date]`
- `/weekly-digests`

## 快速开始

```bash
# 1) 启动依赖服务
docker compose up -d db db_test redis

# 2) 启动 API
cd apps/api
uv sync
ENV=local DEBUG=true DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/investpulse uv run uvicorn main:app --reload
```

另开终端：

```bash
# 3) 启动 Web
cd apps/web
pnpm install
pnpm dev
```

变更后统一验收（必跑）：

```bash
DATABASE_URL_TEST=postgresql+asyncpg://postgres:postgres@localhost:5433/investpulse_test make verify
```

## 文档入口（SSOT）

- [docs/INDEX.md](./docs/INDEX.md)：文档导航与权威关系
- [docs/RUNBOOK.md](./docs/RUNBOOK.md)：本地运行与回放命令
- [docs/DEV_WORKFLOW.md](./docs/DEV_WORKFLOW.md)：开发/测试/lint/迁移/verify
- [docs/API.md](./docs/API.md)：API 协议与示例
- [docs/TRACEABILITY_AND_REPLAY.md](./docs/TRACEABILITY_AND_REPLAY.md)：可追溯与回放语义
- [docs/STATUS.md](./docs/STATUS.md)：能力边界账本

## 架构快照

```text
X/来源帖子
   -> raw_posts
   -> extraction (prompt + model + normalize)
   -> post_extractions
   -> review/auto-review
   -> kol_views
   -> daily_digests（唯一 profile_id + digest_date，当前 API 写入 profile_id=1）
   -> dashboard / digest replay
```

## 未来发展

- 完善抽取质量评估与质量闸门。
- 增强 profile 维度信号排序与筛选能力。
- 强化 Digest 可解释性与证据摘要表达。
- 规划事件提醒生命周期能力（当前未实现）。
- 继续收敛和清理非 X 核心流程的遗留路径。

## License

MIT，详见 [LICENSE](./LICENSE)。
