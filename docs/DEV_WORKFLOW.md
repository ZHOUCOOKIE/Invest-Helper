# DEV WORKFLOW

Authoritative

## Install

```bash
pnpm install
cd /home/zhoucookie/code/investpulse/apps/api && uv sync
```

## Migrations

```bash
cd /home/zhoucookie/code/investpulse/apps/api
ENV=local DEBUG=false DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/investpulse uv run alembic upgrade head
```

## Lint

```bash
pnpm lint
cd /home/zhoucookie/code/investpulse/apps/api && uv run ruff check .
```

## Tests

API tests:
```bash
./scripts/test_api.sh
```

Web tests:
```bash
pnpm test:web
```

Root-level verification:
```bash
DATABASE_URL_TEST=postgresql+asyncpg://postgres:postgres@localhost:5433/investpulse_test make verify
```

What `make verify` currently does:
- starts `db`, `db_test`, `redis`
- runs API migrations against dev DB
- runs API ruff checks
- runs web lint
- runs API tests
- runs web tests
- runs root `pnpm test`

## Test DB Guardrail

Tests must never run against a non-test database.

Current enforcement:
- `apps/api/tests/conftest.py` resolves `DATABASE_URL_TEST` first, then `DATABASE_URL`
- test startup exits immediately if the chosen URL does not contain `test` in DB name or schema
- test session sets:
  - `ENV=test`
  - `DEBUG=true`
  - `DATABASE_URL=<resolved test url>`
  - `DATABASE_URL_TEST=<resolved test url>`
- default test flow resets the public schema before and after the test session, then runs migrations

## Change Scope And Validation

- If you changed only 1-2 files, run the smallest relevant validation.
- If you changed a local module or route family, run targeted tests for that module.
- If you changed core business flow, migrations, or public API contracts, run `make verify`.

## Extraction Change Checklist

When changing extraction prompt or normalization, update all of:
- [extraction_prompt.py](/home/zhoucookie/code/investpulse/apps/api/services/prompts/extraction_prompt.py)
- [extraction.py](/home/zhoucookie/code/investpulse/apps/api/services/extraction.py)
- [main.py](/home/zhoucookie/code/investpulse/apps/api/main.py)
- affected tests under `/home/zhoucookie/code/investpulse/apps/api/tests`
- docs:
  - `/home/zhoucookie/code/investpulse/docs/API.md`
  - `/home/zhoucookie/code/investpulse/docs/STATUS.md`
  - `/home/zhoucookie/code/investpulse/docs/RUNBOOK.md`
  - `/home/zhoucookie/code/investpulse/docs/PROMPT_AND_FLOW_REASONING_ZH.md`

## Current Extraction Contract Snapshot

- normalized keys: `as_of/source_url/islibrary/hasview/asset_views/library_entry`
- `asset_views` keeps `confidence >= 70`
- `hasview` is recomputed from normalized `asset_views`
- `library_entry` shape is `{tag, summary}` and current strict summary literal is `测试`
- `parsed_model_output` uses ordered DB `JSON`
- parse-failed outputs persist as `pending` extraction rows and are treated as failed semantics by retry/progress logic
