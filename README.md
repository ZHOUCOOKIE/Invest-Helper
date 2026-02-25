# InvestPulse

Personal investment signal dashboard (KOL views -> per-asset aggregation -> digest).

## Local dev
- Copy env template:
  - `cp apps/api/.env.example apps/api/.env` and fill your own keys (do not commit).
- Start infra:
  - `docker compose up -d db redis`
- API:
  - `cd apps/api && uv sync && uv run alembic upgrade head && uv run uvicorn main:app --reload --port 8000`
- Web:
  - `cd apps/web && pnpm install && pnpm dev`

## Security
- Never commit `.env`, `RUNBOOK_LOCAL.txt`, `twitter-*.json`.
