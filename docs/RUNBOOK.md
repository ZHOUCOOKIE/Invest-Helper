# Runbook

Authoritative

TL;DR
- Use this file to run one complete daily-output cycle and replay it.
- Use `make verify` before/after changes.
- For traceability details, see `docs/TRACEABILITY_AND_REPLAY.md`.

## Preconditions

- `.env` exists at `apps/api/.env`.
- Docker running.
- Ports 3000, 5432, 5433, 6379, 8000 available.

## Minimal End-to-End Flow

1. Start stack
```bash
docker compose up -d db db_test redis
```

2. Start API
```bash
cd apps/api
uv sync
uv run alembic upgrade head
uv run uvicorn main:app --reload --port 8000
```

3. Start Web
```bash
cd apps/web
pnpm install
pnpm dev
```

4. Optional ingest batch
```bash
curl -s -X POST "http://localhost:8000/ingest/x/import" \
  -H "Content-Type: application/json" \
  --data @/tmp/x_import.json | jq
```

5. Optional extract batch
```bash
curl -s -X POST "http://localhost:8000/raw-posts/extract-batch" \
  -H "Content-Type: application/json" \
  --data '{"raw_post_ids":[1,2,3],"mode":"pending_only"}' | jq
```

6. Generate daily digest
```bash
curl -s -X POST "http://localhost:8000/digests/generate?date=2026-02-24&days=7&profile_id=1" | jq
```

7. Check dashboard and digest pages
- `http://localhost:3000/dashboard`
- `http://localhost:3000/digests/2026-02-24`

## Replay And Evidence Checks

- Latest by date/profile
```bash
curl -s "http://localhost:8000/digests?date=2026-02-24&profile_id=1" | jq
```

- Fixed version
```bash
curl -s "http://localhost:8000/digests?date=2026-02-24&profile_id=1&version=1" | jq
```

- By digest id
```bash
curl -s "http://localhost:8000/digests/1" | jq
```

- Evidence links
```bash
curl -s "http://localhost:8000/digests?date=2026-02-24&profile_id=1" | jq '..|.source_url? // empty'
```

## Acceptance Command

```bash
make verify
```

## Not Implemented

- Event reminder trigger/scheduler runbook.
