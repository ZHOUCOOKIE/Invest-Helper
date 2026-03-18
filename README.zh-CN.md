# InvestPulse

[English](./README.md) | [简体中文](./README.zh-CN.md)

InvestPulse 是一个面向投研与交易工作流的投资信号看板系统，核心强调证据可追溯与摘要可回放。系统会把多源帖子转成结构化观点，支持审核流程，并提供日报与周报回放能力。

## 界面预览

![InvestPulse Dashboard Preview](./docs/images/dashboard-preview.png)

## 项目背景

直接基于社交帖子做投资判断，通常会在三个环节出现问题：

- 信息分散在多个来源，难以稳定聚合。
- 结论与原始证据链接脱节，审计和复盘成本高。
- 每日结论容易被覆盖，历史回放不可靠。

InvestPulse 的目标是把 `帖子 -> 抽取 -> 审核 -> 观点 -> Digest 回放` 这条链路做成既可追溯又可运营的系统。

## 项目愿景

- 保证所有关键输出都能回链到 `source_url`、`raw_posts` 和 `post_extractions`。
- 把非结构化帖子流沉淀为结构化 `asset_views` 和 Digest 输入。
- 让日报和周报成为可回放资产，而不是一次性结果。
- 保持本地开发、测试和 CI 验收流程简单且可执行。

## 当前状态

- 已实现 X 导入、原始帖子入库、抽取、审核、观点沉淀，以及 Digest 生成与回放。
- 抽取结果统一归一化为 `as_of/source_url/islibrary/hasview/asset_views/library_entry`。
- `asset_views` 仅保留 `confidence >= 70` 的记录，并基于最终结果回算 `hasview`。
- 当 `library_entry` 无效或缺失时，`islibrary=1` 会自动降级为 `islibrary=0`，不会直接让记录失败。
- `hasview=0` 会自动拒绝。
- 基于置信度的自动通过使用 `>=80` 阈值路径。
- 拒绝原因统一写入 `meta.auto_review_reason`。
- 模型输出解析失败时，记录仍以 `pending` 落库，但在重试与进度统计中按失败语义处理。
- 手动 `POST /extractions/{id}/re-extract` 现在是“替换式重提取”：当新的 AI 结果有效时，同一原帖下更早的 extraction 以及仅被这些旧记录引用的 `kol_views` 会被删除。
- `/ingest/x/import` 现在会通过 `pending_failed_dedup_*` 返回“已去重但最新 extraction 仍属失败语义”的原帖，供后续异步抽取任务只针对仍需 AI 处理的行继续执行。
- 日报使用 3 天保留窗口，并会在读取和列举时自动清理过期数据。
- 周报会按 `report_kind` 自动清理锚点日期已过期的陈旧记录，避免旧锚点混入当前回放结果。
- `/extractions` 按业务时间倒序排列，优先使用 `raw_post.posted_at`，缺失时回退到 `created_at`。
- Web 端已经加入 Digest 生成后的恢复轮询、审核队列的仓库级统计，以及优先基于 `ai_call_used` 展示的 AI 上传进度。
- `scripts/investpulse` 可用于后台启动/停止 API 与 Web，并在启动时自动执行迁移、管理本地日志与 PID 文件。

## 产品能力面

- API：FastAPI + SQLAlchemy + Alembic
- Web：Next.js 的 dashboard、ingest、extractions、assets、KOL、daily digest、weekly digest、portfolio 页面
- 存储：PostgreSQL（dev/test）与 Redis

主要路由：

- `/dashboard`
- `/portfolio`
- `/ingest`
- `/extractions`
- `/assets`
- `/kols`
- `/digests/[date]`
- `/weekly-digests`
- `/health`

## 前景与发展方向

InvestPulse 现在已经具备“可追溯信号运营层”的基础能力。下一阶段重点会放在抽取质量控制、profile 维度决策视图增强，以及把更多输出从“请求时生成”升级为“可版本化回放”。

规划方向：

- 增加更强的抽取评估与回归控制。
- 扩展 profile 维度的排序、筛选与回放隔离能力。
- 把组合建议从请求时计算升级为可版本化存储与回放。`Not Implemented`
- 增加事件与提醒生命周期能力。`Not Implemented`
- 持续收敛 X 之外的遗留和非核心路径。

## 快速开始

```bash
# 1) 基础设施
docker compose up -d db db_test redis

# 2) API
cd apps/api
uv sync
ENV=local DEBUG=true DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/investpulse uv run uvicorn main:app --reload
```

另开一个终端：

```bash
# 3) Web
cd apps/web
pnpm install
pnpm dev
```

变更后的统一验收：

```bash
DATABASE_URL_TEST=postgresql+asyncpg://postgres:postgres@localhost:5433/investpulse_test make verify
```

可选的后台启动助手：

```bash
mkdir -p ~/.local/bin
ln -sf /home/zhoucookie/code/investpulse/scripts/investpulse ~/.local/bin/investpulse
investpulse start
```

## 文档导航

- [docs/INDEX.md](./docs/INDEX.md)：文档权威关系与导航入口
- [docs/RUNBOOK.md](./docs/RUNBOOK.md)：本地运行与回放命令
- [docs/DEV_WORKFLOW.md](./docs/DEV_WORKFLOW.md)：开发、测试、lint、迁移与 verify 流程
- [docs/API.md](./docs/API.md)：API 协议与示例
- [docs/TRACEABILITY_AND_REPLAY.md](./docs/TRACEABILITY_AND_REPLAY.md)：可追溯与回放语义
- [docs/STATUS.md](./docs/STATUS.md)：已实现与未实现能力账本

## 架构快照

```text
X/来源帖子
   -> raw_posts
   -> extraction (prompt + model + normalize)
   -> post_extractions
   -> review/auto-review
   -> kol_views
   -> daily_digests / weekly_digests
   -> dashboard / digest replay
```

## License

MIT。详见 [LICENSE](./LICENSE)。
