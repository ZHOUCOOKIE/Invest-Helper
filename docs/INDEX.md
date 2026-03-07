# Documentation Index

Authoritative

TL;DR
- 本文件是仓库文档导航与权限分层的单一入口。
- `make verify` 是变更后的统一验收命令。
- 若文档冲突，以本文件声明的权威文档为准。

## SSOT Map

- `README.md`（Authoritative）
  - 项目目标、能力边界、文档入口。
- `docs/RUNBOOK.md`（Authoritative）
  - 本地运行、端口/URL、端到端操作与回放命令。
- `docs/DEV_WORKFLOW.md`（Authoritative）
  - 开发流程、测试/迁移/lint/verify 命令与 test DB 规则。
- `docs/API.md`（Authoritative）
  - API 端点族与可执行示例。
- `docs/TRACEABILITY_AND_REPLAY.md`（Authoritative）
  - 证据链、当前回放约束与数据语义。
- `docs/STATUS.md`（Authoritative）
  - 已实现/未实现能力账本（仅代码事实）。
- `docs/GLOSSARY.md`（Reference）
  - 统一术语定义。
- `docs/DOC_AUDIT.md`（Authoritative）
  - 文档合并审计与决策记录。

## Additional References

- `docs/PROMPT_AND_FLOW_REASONING_ZH.md`（Reference）
- `docs/public_checklist.md`（Reference）
- `apps/api/README.md`（Service-local Reference）
- `apps/web/README.md`（Service-local Reference）
