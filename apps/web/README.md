# InvestPulse Web

Reference

TL;DR
- Next.js frontend for dashboard, ingest, review, assets, profile, and digest pages.
- For repository-level authoritative docs, use `docs/INDEX.md`.
- Post-change acceptance command is `make verify` from repo root.

## Start Web

```bash
cd apps/web
pnpm install
pnpm dev
```

Default URL: `http://localhost:3000`

## API Binding

- `next.config.ts` rewrites `/api/:path*` to `http://localhost:8000/:path*`.
- Start API before using web pages.

## Main Routes

- `/dashboard`
- `/ingest`
- `/extractions`
- `/extractions/[id]`
- `/assets`
- `/assets/[id]`
- `/profile`
- `/digests/[date]`
- `/health`

## Scope Note

- Reddit-related UI remnants are legacy, not a core product target.
