# Status

Authoritative

TL;DR
- This document is the code-based capability ledger.
- It defines implemented vs non-implemented scope.
- If another doc conflicts with this file, update that doc.

Last verified: 2026-02-27

## Runtime Deliverables

- Infra: `docker compose up -d db db_test redis`
- API: `cd apps/api && uv sync && uv run alembic upgrade head && uv run uvicorn main:app --reload --port 8000`
- Web: `cd apps/web && pnpm install && pnpm dev`
- Acceptance: `make verify`

## Implemented Capability Map

Ingest
- `POST /ingest/manual`
- `POST /ingest/x/convert`
- `POST /ingest/x/import`
- `POST /ingest/x/following/import`
- `GET /ingest/x/raw-posts/preview`
- `GET /ingest/x/progress`
- `POST /ingest/x/retry-failed`
- `POST /raw-posts`

Normalize / Extract / Review
- Extraction: single, batch, async job modes.
- Review endpoints: approve/reject/re-extract and batch approve.
- Runtime extraction audit persistence (`prompt_*`, model IO/tokens/latency).
- Admin repair endpoints for operational recovery.

Serve
- Assets/KOLs/profiles endpoints.
- Dashboard with clarity ranking and contributor evidence.
- Digest generate/read/list dates, versioned per profile+date.

Storage
- PostgreSQL + SQLAlchemy + Alembic.
- Core entities: `raw_posts`, `post_extractions`, `kol_views`, `daily_digests`, profile tables.

User Config
- Profile KOL weight/enabled controls.
- Profile market filters.

Observability
- `/health`
- Request id middleware.
- Extraction status/job counters.

## Higher-Standard Behaviors Already Implemented

- Digest replay includes explicit version fetch and date list.
- Digest time-field fallback chain: `as_of -> posted_at -> created_at`.
- Clarity ranking includes score and top contributors.
- Batch extraction includes skip/resume semantics.

## Not Implemented

- Event calendar entity.
- Reminder scheduling/triggering.
- Notification delivery channels.
- Prediction-market integration.

## Non-goal / Legacy

- Reddit is not a core target pipeline in current scope.
- Any remaining Reddit UI/wording is legacy and should be removed gradually.
