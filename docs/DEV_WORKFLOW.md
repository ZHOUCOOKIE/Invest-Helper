# Dev Workflow

Authoritative

TL;DR
- `make verify` is the mandatory acceptance command.
- API tests must run on test DB only (`DATABASE_URL_TEST` guard).
- Keep docs/code aligned in every PR.

## Authoritative Command

```bash
make verify
```

This runs:
- `docker compose up -d db db_test redis`
- API migration check (`alembic upgrade head` on dev DB)
- API lint (`ruff check`)
- web lint (`pnpm lint`)
- API tests (`./scripts/test_api.sh`)
- web tests (`pnpm test:web`)
- monorepo tests (`pnpm test`)

## Local Setup

```bash
docker compose up -d db db_test redis
cd apps/api && uv sync
cd apps/web && pnpm install
```

## Focused Commands

- API tests only: `./scripts/test_api.sh`
- web tests only: `pnpm test:web`
- lint only: `pnpm lint`
- API migration only: `cd apps/api && uv run alembic upgrade head`

## Common Failures And Fixes

- `Refusing to run tests on non-test database URL`
  - Set `DATABASE_URL_TEST` to database/schema containing `test`.
- Alembic/env parsing failure
  - Ensure `.env` has valid booleans (`DEBUG=true|false`) and rerun `make verify`.
- DB connection refused
  - Run `docker compose up -d db db_test redis`.
- web lint/test dependency missing
  - Run `cd apps/web && pnpm install`.
- Port conflict (3000/8000/5432/5433/6379)
  - Stop conflicting process or change local binding.

## Documentation Discipline

- Update `docs/STATUS.md` when capability boundary changes.
- Keep endpoint examples in `docs/API.md` executable.
- Keep replay/traceability rules in `docs/TRACEABILITY_AND_REPLAY.md` aligned.
