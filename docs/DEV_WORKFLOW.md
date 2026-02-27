# Dev Workflow

Authoritative

TL;DR
- 本文是开发、测试、lint、迁移与验收命令的唯一权威来源。
- 统一验收命令：`make verify`。
- API 测试必须使用 test DB（`DATABASE_URL_TEST`/`DATABASE_URL` 必须包含 `test`）。

## One-Command Acceptance

```bash
make verify
```

`make verify` 当前执行顺序（来自 `Makefile`）：
- `verify-migrate`
- `verify-ruff`
- `lint`
- `verify-api`
- `verify-web`
- `pnpm test`

## Focused Commands

- API migration: `cd apps/api && ENV=local DEBUG=false DATABASE_URL=${DATABASE_URL:-postgresql+asyncpg://postgres:postgres@localhost:5432/investpulse} uv run alembic upgrade head`
- API lint: `cd apps/api && uv run ruff check .`
- API tests (guarded): `./scripts/test_api.sh`
- Web tests: `pnpm test:web`
- Monorepo tests: `pnpm test`
- Web lint: `pnpm lint`

## Test DB Guardrails

- `./scripts/test_api.sh` 会设置：
  - `ENV=test`
  - `DEBUG=true`
  - `DATABASE_URL_TEST` 默认 `postgresql+asyncpg://postgres:postgres@localhost:5433/investpulse_test`
  - `DATABASE_URL=$DATABASE_URL_TEST`
- `apps/api/tests/conftest.py` 会拒绝非 test 库：
  - `Refusing to run tests on non-test database URL`
- pytest session 会在前后重置 schema，并在测试前自动执行 `alembic upgrade head`。

## Common Failures

- `Refusing to run tests on non-test database URL`
  - 使用包含 `test` 的 DB/schema URL。
- DB connection refused
  - 先按 `docs/RUNBOOK.md` 启动 infra（`db/db_test/redis`）。
- Alembic/env parse failure
  - 检查 `.env`，特别是布尔值（如 `DEBUG=true|false`）。
- Web deps missing
  - 执行 `cd apps/web && pnpm install`。

## Documentation Update Rule

- 运行/回放变更：更新 `docs/RUNBOOK.md`
- API 合约变更：更新 `docs/API.md`
- 证据链/版本语义变更：更新 `docs/TRACEABILITY_AND_REPLAY.md`
- 能力边界变更：更新 `docs/STATUS.md`
