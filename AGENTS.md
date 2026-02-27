# AGENTS.md

## 项目目标（一句话）

InvestPulse 是一个基于多源帖子抽取与证据可追溯的投资信号看板系统，支持按 profile/日期生成并回放版本化 Daily Digest。

## 不可破坏约束

- 可追溯：所有观点输出必须能回链到 `source_url` / `raw_posts` / `post_extractions`。
- 可回放：Digest 必须保留 `profile_id + digest_date + version`，不得破坏按版本回放。
- 测试绝不碰开发库：测试只能连接 test DB（`DATABASE_URL_TEST`），非 test 库必须直接 fail。
- 文档与代码一致：文档只写已实现行为，未实现能力必须显式标注 `Not Implemented`。

## 代码修改守则

- 小步修改，避免无关重构。
- 尽量先补/改测试再改实现。
- 新增配置同步更新 `apps/api/.env.example`。
- 不引入非必要依赖；优先复用现有 `FastAPI + SQLAlchemy + Alembic + Next.js` 栈。

## 权威命令

- 每次改动后的统一验收（必须）：`make verify`
- Dev 基础设施：`docker compose up -d db redis`
- Dev API：`cd apps/api && uv sync && uv run alembic upgrade head && uv run uvicorn main:app --reload --port 8000`
- Dev Web：`cd apps/web && pnpm install && pnpm dev`
- API 测试（强制 test DB）：`./scripts/test_api.sh`
- 全量测试：`pnpm test`
- 迁移：`cd apps/api && uv run alembic upgrade head`
- Lint：
  - Web: `cd apps/web && pnpm lint`
  - API: `cd apps/api && uv run ruff check .`

## Repo 导航

- `apps/api/main.py`: FastAPI 路由与核心工作流
- `apps/api/services/`: digest/profile/extraction 业务逻辑
- `apps/api/models.py`: 数据模型与约束
- `apps/api/alembic/`: 数据库迁移
- `apps/api/tests/`: API 测试（含 DB 安全护栏）
- `apps/web/app/`: Dashboard/Ingest/Extractions/Assets/Profile/Digest 页面
- `scripts/`: 仓库级运维脚本（如 `test_api.sh`, `reset_db_local.sh`）
- `docs/`: 状态盘点、文档审计与专题说明

## 非目标/弃用处理原则

- Reddit 相关流程当前不是产品目标。若代码仍有遗留字段/选项，只能标注为“遗留/待移除”，不得作为核心能力继续扩写。
