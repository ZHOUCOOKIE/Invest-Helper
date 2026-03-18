# Documentation Index

Authoritative

TL;DR
- 文档以代码事实为准。
- 运行与回放命令看 `docs/RUNBOOK.md`。
- 开发、测试、迁移与验收看 `docs/DEV_WORKFLOW.md`。

## SSOT Map

- `README.md`
  - 项目概览、能力边界、入口导航。
- `README.zh-CN.md`
  - 中文概览，与 `README.md` 保持同一事实层级。
- `docs/RUNBOOK.md`
  - 本地运行、后台启动、常用检查、digest 回放命令。
- `docs/DEV_WORKFLOW.md`
  - 依赖安装、测试、lint、迁移、`make verify`、test DB 护栏。
- `docs/API.md`
  - 当前 API 端点族、关键请求参数、返回语义与示例。
- `docs/TRACEABILITY_AND_REPLAY.md`
  - 证据链、digest 保留与回放语义、替换式 re-extract 语义。
- `docs/STATUS.md`
  - 已实现能力、约束、`Not Implemented` 清单。
- `docs/GLOSSARY.md`
  - 统一术语定义。
- `docs/DOC_AUDIT.md`
  - 文档收敛与 SSOT 维护说明。

## Reference Docs

- `docs/PROMPT_AND_FLOW_REASONING_ZH.md`
  - extraction prompt 与 normalize / auto-review 流程说明。
- `docs/public_checklist.md`
  - 对外发布前检查清单。
- `apps/api/README.md`
  - API 服务级入口。
- `apps/web/README.md`
  - Web 服务级入口。

## Conflict Rule

- 若多个文档描述冲突，优先级按本页 `SSOT Map` 解释。
- 命令示例不在多处重复维护；运行命令以 `docs/RUNBOOK.md` 为准。
- 未实现能力必须明确标为 `Not Implemented`，不得按已实现能力描述。
