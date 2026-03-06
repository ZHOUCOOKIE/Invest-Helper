# AGENTS.md

## 项目目标（一句话）

InvestPulse 是一个基于多源帖子抽取与证据可追溯的投资信号看板系统，支持按 profile/日期生成并回放版本化 Daily Digest。

## 不可破坏约束

- 可追溯：所有观点输出必须能回链到 `source_url` / `raw_posts` / `post_extractions`。
- 可回放：Digest 必须保留 `profile_id + digest_date + version`，不得破坏按版本回放。
- 测试绝不碰开发库：测试只能连接 test DB（`DATABASE_URL_TEST`），非 test 库必须直接 fail。
- 文档与代码一致：文档只写已实现行为，未实现能力必须显式标注 `Not Implemented`。
- 抽取语义不可漂移：`assets` 最终形态必须为对象数组；无标的用 `NoneAny` 哨兵对象；library 语义通过 `islibrary + library_entry` 表达。
- Library 边界强约束（A2）：`library_entry` 为入库唯一有效入口，且必须满足：
  - `confidence >= 70`
  - `tags` 长度 `1..2`
  - `tags` 枚举固定为 `macro/industry/thesis/strategy/risk/events`
  - `islibrary=1` 且 `library_entry` 无效/缺失时，必须 normalize 降级为 `islibrary=0`（不走失败兜底）
- 向后兼容：历史 `approved/rejected` 记录不得回写重算；旧记录若仍含 `library_tags` 必须忽略且不得报错，不得改变其状态展示。

## 文档单一事实来源（SSOT）

- 文档导航入口：`docs/INDEX.md`
- 本地运行/回放：`docs/RUNBOOK.md`
- 开发/测试/lint/迁移：`docs/DEV_WORKFLOW.md`
- API 合约与示例：`docs/API.md`
- 可追溯与回放语义：`docs/TRACEABILITY_AND_REPLAY.md`
- 能力边界账本：`docs/STATUS.md`
- 术语：`docs/GLOSSARY.md`

## 代码修改守则

- 小步修改，避免无关重构。
- 尽量先补/改测试再改实现。
- 新增配置同步更新 `apps/api/.env.example`。
- 不引入非必要依赖；优先复用现有 `FastAPI + SQLAlchemy + Alembic + Next.js` 栈。

## 权威命令

- 每次改动后的统一验收（必须）：`make verify`
- 本地运行与回放命令：`docs/RUNBOOK.md`
- 开发/测试/迁移/lint 命令：`docs/DEV_WORKFLOW.md`

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
