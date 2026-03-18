# InvestPulse Web

Reference

TL;DR
- Next.js frontend for dashboard, ingest, review, assets, kols, daily digest, weekly digest, and portfolio pages.
- For repository-level authoritative docs, use `docs/INDEX.md`.
- Runtime/test commands are maintained in:
  - `docs/RUNBOOK.md` (run/replay)
  - `docs/DEV_WORKFLOW.md` (test/lint/migrate/verify)

## API Binding

- `next.config.ts` rewrites `/api/:path*` to `http://localhost:8000/:path*`.
- Start API before using web pages.

## Main Routes

- `/dashboard`
- `/portfolio`
- `/ingest`
- `/extractions`
- `/extractions/[id]`
- `/assets`
- `/assets/[id]`
- `/kols`
- `/digests/[date]`
- `/weekly-digests`
- `/health`

Current UI notes:
- digest pages include recovery polling after generate-call proxy failures
- ingest progress prefers `ai_call_used` when the API provides it
- extractions page shows repository totals from `/extractions/stats`

## Scope Note

- Non-X UI remnants are legacy, not a core product target.
