# DEV WORKFLOW

## Core Commands
- Install deps: `pnpm install` and `cd apps/api && uv sync`
- API tests: `./scripts/test_api.sh`
- Web tests: `pnpm test:web`
- Lint: `pnpm lint` and `cd apps/api && uv run ruff check .`
- Migrate: `cd apps/api && ENV=local DEBUG=false DATABASE_URL=... uv run alembic upgrade head`
- Unified verification (required): `make verify`

## Test DB Guardrail
- Tests must use test DB only.
- Required env:
  - `ENV=test`
  - `DATABASE_URL_TEST=postgresql+asyncpg://.../investpulse_test`
  - `DATABASE_URL=$DATABASE_URL_TEST`

## Extraction Ruleset Update Checklist
When changing extraction rules, update all of:
- Prompt SSOT: `apps/api/services/prompts/extraction_prompt.py`
- Runtime normalize/schema: `apps/api/services/extraction.py`
- Auto-review path: `apps/api/main.py`
- Tests under `apps/api/tests/`
- Docs (`docs/API.md`, `docs/STATUS.md`, `docs/RUNBOOK.md`, `docs/PROMPT_AND_FLOW_REASONING_ZH.md`)

## Current Extraction Contract
- Top-level strict business keys: `as_of/source_url/islibrary/assets/asset_views/library_entry`
- Asset views keep `confidence>=70`
- Library review key: `library_entry.confidence`
