# InvestPulse API

Reference

TL;DR
- API service uses FastAPI + SQLAlchemy + Alembic.
- For repository-level authoritative flow, start from `docs/INDEX.md`.
- Runtime/test commands are maintained in:
  - `docs/RUNBOOK.md` (run/replay)
  - `docs/DEV_WORKFLOW.md` (test/lint/migrate/verify)

## API Contract Entry

- OpenAPI JSON: `GET /openapi.json`
- Swagger UI: `GET /docs`
- Detailed endpoint examples: `docs/API.md`

## Data And Replay Pointers

- Traceability and replay policy: `docs/TRACEABILITY_AND_REPLAY.md`
- Capability boundary: `docs/STATUS.md`

## Not Implemented (API scope)

- Event reminder scheduling/triggering API.
