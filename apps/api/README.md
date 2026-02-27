# InvestPulse API

Reference

TL;DR
- API service uses FastAPI + SQLAlchemy + Alembic.
- For repository-level authoritative flow, start from `docs/INDEX.md`.
- Post-change acceptance command: `make verify` (run from repo root).

## Local Service Start

```bash
cd apps/api
uv sync
uv run alembic upgrade head
uv run uvicorn main:app --reload --port 8000
```

## Service-Scoped Test Command

From repo root:

```bash
./scripts/test_api.sh
```

This command enforces test DB guardrails (`DATABASE_URL_TEST` only).

## API Contract Entry

- OpenAPI JSON: `GET /openapi.json`
- Swagger UI: `GET /docs`
- Detailed endpoint examples: `docs/API.md`

## Data And Replay Pointers

- Traceability and replay policy: `docs/TRACEABILITY_AND_REPLAY.md`
- Capability boundary: `docs/STATUS.md`

## Not Implemented (API scope)

- Event reminder scheduling/triggering API.
