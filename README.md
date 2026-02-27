# InvestPulse

Authoritative

TL;DR
- InvestPulse aggregates investment views from X posts, then serves dashboard and versioned daily digests.
- Core guarantees: traceability, replay/versioning, strict test DB isolation, docs aligned to code.
- Single acceptance command after changes: `make verify`.
- Full documentation navigation: `docs/INDEX.md`.

## What This Repo Does

- Ingest and normalize posts (`/ingest/manual`, `/ingest/x/*`, `/raw-posts`).
- Extract structured asset views and support review flows.
- Serve dashboard metrics and clarity ranking.
- Generate and replay profile-scoped, versioned daily digests.

## Current Capability Boundary

Implemented
- X import/convert/import-preview/retry flows.
- Extraction: single, batch, async jobs.
- Review lifecycle: approve/reject/re-extract.
- Profile-scoped rules: KOL weights + market filters.
- Versioned digest replay by date/profile/version/id.
- Traceability fields across raw post, extraction, view, digest.

Not Implemented
- Event reminder/scheduler and lead-time notifications.
- Prediction-market data fusion.

Non-goal (current)
- Reddit is not a core product pipeline.

## Quickstart

1. `cp apps/api/.env.example apps/api/.env`
2. `docker compose up -d db db_test redis`
3. `cd apps/api && uv sync && uv run alembic upgrade head`
4. `cd apps/api && uv run uvicorn main:app --reload --port 8000`
5. `cd apps/web && pnpm install && pnpm dev`
6. Run verification: `make verify`

## Common Commands

- Full verification: `make verify`
- API tests (test DB guarded): `./scripts/test_api.sh`
- Web tests: `pnpm test:web`
- Lint: `pnpm lint`
- API migration: `cd apps/api && uv run alembic upgrade head`

## Documentation

- [Documentation Index](docs/INDEX.md)
- [Status](docs/STATUS.md)
- [Runbook](docs/RUNBOOK.md)
- [Dev Workflow](docs/DEV_WORKFLOW.md)
- [API](docs/API.md)
- [Traceability And Replay](docs/TRACEABILITY_AND_REPLAY.md)
- [Glossary](docs/GLOSSARY.md)
- [Prompt/Flow Snapshot (ZH)](docs/PROMPT_AND_FLOW_REASONING_ZH.md)
