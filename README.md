# InvestPulse

[English](./README.md) | [简体中文](./README.zh-CN.md)

InvestPulse is an evidence-traceable investment signal dashboard built for research and trading workflows. It turns source posts into structured views, supports review workflows, and provides replayable daily and weekly digests.

## UI Preview

![InvestPulse Dashboard Preview](./docs/images/dashboard-preview.png)

## Background

Investment decisions built directly from social posts usually break down in three places:

- Information is fragmented across sources and hard to aggregate consistently.
- Conclusions often lose the original evidence link, which weakens auditability and post-trade review.
- Daily conclusions are easy to overwrite or regenerate, making historical replay unreliable.

InvestPulse is designed to turn the flow of `post -> extraction -> review -> view -> digest replay` into a traceable and operational system.

## Vision

- Keep every output traceable to `source_url`, `raw_posts`, and `post_extractions`.
- Convert unstructured post streams into structured `asset_views` and digest inputs.
- Make daily and weekly summaries replayable instead of disposable.
- Keep the workflow practical for local development, testing, and CI verification.

## Current State

- X ingestion, raw post persistence, extraction, review, asset view materialization, and digest generation are implemented.
- The extraction contract is normalized to `as_of/source_url/islibrary/hasview/asset_views/library_entry`.
- `asset_views` keeps only rows with `confidence >= 70`, and `hasview` is recomputed from the final normalized result.
- Invalid or missing `library_entry` downgrades `islibrary=1` to `islibrary=0` instead of failing the record.
- The current library branch is still intentionally strict and test-shaped: `library_entry.summary` must equal `测试`.
- `hasview=0` is auto rejected.
- Confidence-based auto approval uses the `>=80` threshold path.
- Rejection reasons are recorded under `meta.auto_review_reason`.
- Parse-failed model outputs are still persisted as `pending`, while retry and progress flows classify them as failed semantics.
- Manual `POST /extractions/{id}/re-extract` now acts as replacement: once the new AI result is valid, older extraction rows for the same raw post are deleted together with owned `kol_views`.
- `/ingest/x/import` now reports deduplicated raw posts whose latest extraction is still failed semantics via `pending_failed_dedup_*`, so follow-up extract jobs can target only the rows that still need AI work.
- Daily digests use a 3-day retention window and purge expired rows on read/list paths.
- Weekly digests purge stale rows whose anchor date no longer matches the current expected anchor for the selected `report_kind`.
- Weekly digest AI input is aggregated by day, and the current prompt emphasizes intraday to `1w/1m` impact plus author-grouped short-term trading observations.
- `/extractions` is ordered by business post time descending, using `raw_post.posted_at` and falling back to `created_at`.
- The web UI includes recovery polling for digest generation requests that may succeed in the backend after a proxy-side failure, extraction repository stats on the review queue, and AI upload progress derived from `ai_call_used` when available.
- `scripts/investpulse` can start/stop API and Web in the background, run migrations on start, and manage local logs/PID files.
- Profile tables and server-side profile rule loading exist, but public profile-management routes are not exposed yet.

## Product Surface

- API: FastAPI + SQLAlchemy + Alembic
- Web: Next.js dashboard, ingest, extractions, assets, KOL, daily digest, weekly digest, and portfolio pages
- Storage: PostgreSQL for dev/test and Redis

Main routes:

- `/dashboard`
- `/portfolio`
- `/ingest`
- `/extractions`
- `/assets`
- `/kols`
- `/digests/[date]`
- `/weekly-digests`
- `/health`

## Outlook

InvestPulse is already useful as a traceable signal operations layer. The next stage is to improve extraction quality controls, strengthen profile-based decision surfaces, and make more output types replayable instead of request-only.

Planned directions:

- Add stronger extraction evaluation and regression controls.
- Expand profile-scoped ranking, filtering, and replay isolation.
- Version and replay portfolio advice instead of keeping it request-time only. `Not Implemented`
- Introduce event and reminder lifecycle support. `Not Implemented`
- Continue shrinking legacy and non-core paths outside the X-centric workflow.

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

Unified verification after changes:

```bash
DATABASE_URL_TEST=postgresql+asyncpg://postgres:postgres@localhost:5433/investpulse_test make verify
```

Optional background helper:

```bash
mkdir -p ~/.local/bin
ln -sf /home/zhoucookie/code/investpulse/scripts/investpulse ~/.local/bin/investpulse
investpulse start
```

## Documentation Map

- [docs/INDEX.md](./docs/INDEX.md): documentation authority and navigation
- [docs/RUNBOOK.md](./docs/RUNBOOK.md): local run and replay commands
- [docs/DEV_WORKFLOW.md](./docs/DEV_WORKFLOW.md): development, test, lint, migration, and verify workflow
- [docs/API.md](./docs/API.md): API contract and examples
- [docs/TRACEABILITY_AND_REPLAY.md](./docs/TRACEABILITY_AND_REPLAY.md): traceability and replay semantics
- [docs/STATUS.md](./docs/STATUS.md): implemented and not implemented capability ledger

## Architecture Snapshot

```text
X/Source Posts
   -> raw_posts
   -> extraction (prompt + model + normalize)
   -> post_extractions
   -> review/auto-review
   -> kol_views
   -> daily_digests / weekly_digests
   -> dashboard / digest replay
```

## License

MIT. See [LICENSE](./LICENSE).
