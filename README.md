# InvestPulse

[English](./README.md) | [简体中文](./README.zh-CN.md)

InvestPulse is an evidence-traceable investment signal dashboard built from multi-source posts.  
It extracts structured signals, supports review workflows, and provides Daily Digest generation/replay by `profile_id + digest_date`.

## Why InvestPulse

- Keep signal extraction auditable with source-linked evidence.
- Turn unstructured post streams into structured `asset_views`.
- Support profile-scoped decision views and date-scoped digest playback.
- Keep operations practical for local development and CI verification.

## Current Implementation (Code-Verified)

- Ingestion and raw post persistence for X workflows.
- Extraction pipeline with single, batch, and async execution paths.
- Prompt SSOT in `apps/api/services/prompts/extraction_prompt.py`.
- Runtime normalize contract: `as_of/source_url/islibrary/hasview/asset_views/library_entry`.
- Strict enum handling for `market`, `stance`, and `horizon`.
- Auto-review flow:
  - non-library + `hasview=0` => auto reject
  - non-library + valid confidence path => threshold review (`70`)
  - `islibrary=1` => auto approve (`library_flag`)
- Traceability chain across `raw_posts -> post_extractions -> kol_views -> daily_digests`.
- Replay-ready digests stored by `profile_id + digest_date`.

## Product Surfaces

- API: FastAPI + SQLAlchemy + Alembic
- Web: Next.js dashboard/review/profile/digest pages
- Storage: PostgreSQL (dev + test), Redis

Main routes:
- `/dashboard`
- `/ingest`
- `/extractions`
- `/assets`
- `/profile`
- `/digests/[date]`

## Quick Start

```bash
# 1) Infra
docker compose up -d db db_test redis

# 2) API
cd apps/api
uv sync
ENV=local DEBUG=true DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/investpulse uv run uvicorn main:app --reload
```

In another shell:

```bash
# 3) Web
cd apps/web
pnpm install
pnpm dev
```

Unified verification (required after changes):

```bash
DATABASE_URL_TEST=postgresql+asyncpg://postgres:postgres@localhost:5433/investpulse_test make verify
```

## Documentation Map (SSOT)

- [docs/INDEX.md](./docs/INDEX.md): documentation authority and navigation
- [docs/RUNBOOK.md](./docs/RUNBOOK.md): run and replay commands
- [docs/DEV_WORKFLOW.md](./docs/DEV_WORKFLOW.md): dev/test/lint/migrate/verify workflow
- [docs/API.md](./docs/API.md): API contract and examples
- [docs/TRACEABILITY_AND_REPLAY.md](./docs/TRACEABILITY_AND_REPLAY.md): evidence and replay semantics
- [docs/STATUS.md](./docs/STATUS.md): implemented capability ledger

## Architecture Snapshot

```text
X/Source Posts
   -> raw_posts
   -> extraction (prompt + model + normalize)
   -> post_extractions
   -> review/auto-review
   -> kol_views
   -> daily_digests (profile_id + digest_date)
   -> dashboard / digest replay
```

## Roadmap

- Improve extraction quality controls and evaluation benchmarks.
- Add richer profile-level signal ranking and filtering.
- Strengthen digest explainability with clearer evidence summaries.
- Prepare event/reminder lifecycle capabilities (currently not implemented).
- Continue reducing legacy/non-core paths outside the X-centric workflow.

## License

MIT. See [LICENSE](./LICENSE).
