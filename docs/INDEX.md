# Documentation Index

TL;DR
- This is the single navigation entry for repository docs.
- Use `make verify` as the authoritative acceptance command after any change.
- If documents conflict, trust files marked `Authoritative`.
- `RUNBOOK_LOCAL*.txt` are local notes, not repository source of truth.

## Authority Levels

- Authoritative: normative and must stay aligned with code.
- Reference: factual deep-dive or API details; may be narrower in scope.
- Notes: local context or supplementary discussion.

## Core Paths

- Authoritative: [README](../README.md)
  - Project overview, quickstart, high-level boundaries.
- Authoritative: [Dev Workflow](./DEV_WORKFLOW.md)
  - Development lifecycle, verify command, failure handling.
- Authoritative: [Runbook](./RUNBOOK.md)
  - End-to-end operational steps to produce and replay daily outputs.
- Authoritative: [Status](./STATUS.md)
  - Current capabilities, boundaries, implemented/non-implemented map.
- Authoritative: [Traceability And Replay](./TRACEABILITY_AND_REPLAY.md)
  - Evidence chain and replay/version policy.

## API And Technical Reference

- Reference: [API](./API.md)
  - Endpoint families, curl examples, OpenAPI usage.
- Reference (ZH): [Prompt And Flow Reasoning](./PROMPT_AND_FLOW_REASONING_ZH.md)
  - Extraction prompt/runtime reasoning compliance snapshot.
- Reference: [Glossary](./GLOSSARY.md)
  - Canonical terminology.

## Governance And Audit

- Authoritative: [Doc Audit](./DOC_AUDIT.md)
  - File-level audit findings and actions.
- Reference: [Public Checklist](./public_checklist.md)
  - Checklist before publishing repository.

## Local Notes Policy

- Notes (Deprecated as authority): `RUNBOOK_LOCAL.txt`, `RUNBOOK_LOCAL.example.txt`
- These files can keep machine/local examples, but must not be used as authoritative repository docs.
