# Documentation Audit

Authoritative

TL;DR
- 本次审计目标：把重复文档收敛为更少的 SSOT 文档，并删除确认冗余文档。
- 合并日期：2026-02-27。
- 合并后 `README/AGENTS/RUNBOOK/DEV_WORKFLOW/API` 不再各自维护重复流程。

## Consolidation Summary

| Source | Action | SSOT Target | Reason |
|---|---|---|---|
| `RUNBOOK_LOCAL.example.txt` | Deleted | `docs/RUNBOOK.md` | 与 `docs/RUNBOOK.md` 内容重叠且不再需要兼容 stub |
| `repo_tree.txt` | Deleted | `docs/INDEX.md` + `AGENTS.md` | 无引用，目录导航信息已由 SSOT 文档覆盖 |
| `README.md` quickstart details | Merged by reference | `docs/RUNBOOK.md` + `docs/DEV_WORKFLOW.md` | 避免启动/测试命令多点维护 |
| `AGENTS.md` command duplicates | Normalized | `docs/INDEX.md` + SSOT docs | 消除与 runbook/workflow 的冲突风险 |
| `docs/TRACEABILITY_AND_REPLAY.md` replay curls | De-duplicated | `docs/RUNBOOK.md` | 回放操作命令只保留一处权威版本 |
| `docs/STATUS.md` runtime commands | De-duplicated | `docs/RUNBOOK.md` + `docs/DEV_WORKFLOW.md` | `STATUS` 仅保留能力账本 |

## SSOT Ownership After Merge

- `README.md`: 项目目标与边界、文档入口。
- `docs/RUNBOOK.md`: 本地启动、端到端流程、回放命令、运维清理。
- `docs/DEV_WORKFLOW.md`: 开发测试流程、`make verify`、test DB 安全护栏。
- `docs/API.md`: API 端点与请求示例。
- `docs/TRACEABILITY_AND_REPLAY.md`: 证据链与版本化回放语义。
- `docs/STATUS.md`: 仅记录实现状态与非目标。
- `docs/GLOSSARY.md`: 统一术语。
- `docs/INDEX.md`: 导航与权威分层。

## Validation Performed

- 全仓引用检查：`git grep -n "RUNBOOK_LOCAL.example.txt\|repo_tree.txt"`
- 导航检查：`docs/INDEX.md` 包含全部 SSOT 文档路径。
- 随机流程抽检（单一权威）：
  - 本地启动：仅 `docs/RUNBOOK.md`
  - API 测试：仅 `docs/DEV_WORKFLOW.md`
  - Digest 回放命令：仅 `docs/RUNBOOK.md`

## Notes

- 本次仅做文档整理与引用更新，不涉及代码重构。
- 未实现能力继续使用 `Not Implemented` 标注，不做能力扩写。
