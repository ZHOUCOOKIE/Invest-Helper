# Documentation Audit

Authoritative

TL;DR
- This file tracks document-level quality issues and resolutions.
- Goal: single source of truth, no conflicting run/test/replay instructions.
- Last full audit: 2026-02-27.

## Audit Matrix

| File | Issues Found | Action |
|---|---|---|
| `README.md` | Quickstart and docs navigation not fully centralized; no authority label/TL;DR | Rewritten with authority label, concise boundary, links to `docs/INDEX.md` |
| `docs/STATUS.md` | Lacked clear authority marker and concise scope framing | Rewritten as authoritative capability ledger |
| `docs/RUNBOOK.md` | Missing full minimal chain and preconditions | Expanded with end-to-end flow, replay, and evidence checks |
| `docs/DEV_WORKFLOW.md` | Verify command existed but doc discipline/triage incomplete | Consolidated around `make verify` + failure handling |
| `docs/API.md` | Endpoint families only, insufficient executable examples | Added runnable curl examples + sample response notes |
| `docs/TRACEABILITY_AND_REPLAY.md` | Good baseline but no TL;DR/authority and fewer executable checks | Rewritten with clear chain, replay commands, version policy |
| `docs/PROMPT_AND_FLOW_REASONING_ZH.md` | Scope boundary not explicit as specialized reference | Kept as reference; linked from index with explicit positioning |
| `apps/api/README.md` | Partial overlap with root/docs | Retained as service-local reference; should defer to docs index |
| `apps/web/README.md` | Partial overlap with root/docs | Retained as app-local reference; should defer to docs index |
| `RUNBOOK_LOCAL.txt` / `RUNBOOK_LOCAL.example.txt` | Local-note semantics may conflict with SSoT usage | Marked as non-authoritative in `docs/INDEX.md` and README flow |
| `docs/public_checklist.md` | Needed consistency with local-runbook policy | Kept, with local note policy wording aligned |

## Top Conflict Themes Resolved

1. No docs index entry point.
2. Missing authority levels.
3. Quickstart duplication across files.
4. Inconsistent run/test verification entry.
5. API docs lacking executable examples.
6. Replay/traceability fragmented across docs.
7. Not Implemented markers not centralized.
8. Local runbook ambiguity.
9. Prompt deep-dive scope ambiguity.
10. Terminology inconsistency.

## Outcome Rules

- `make verify` is authoritative post-change acceptance command.
- Capability boundary lives in `docs/STATUS.md`.
- Operational execution lives in `docs/RUNBOOK.md`.
- Development lifecycle lives in `docs/DEV_WORKFLOW.md`.
- Replay/evidence rules live in `docs/TRACEABILITY_AND_REPLAY.md`.
- API examples live in `docs/API.md`.
- Terms live in `docs/GLOSSARY.md`.
