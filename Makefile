.PHONY: verify verify-api verify-web verify-ruff verify-migrate lint

verify: verify-migrate verify-ruff lint verify-api verify-web
	pnpm test

verify-api: verify-migrate verify-ruff
	./scripts/test_api.sh

verify-web:
	pnpm test:web

verify-ruff:
	cd apps/api && uv run ruff check .

verify-migrate:
	docker compose up -d db db_test redis
	cd apps/api && ENV=local DEBUG=false DATABASE_URL=$${DATABASE_URL:-postgresql+asyncpg://postgres:postgres@localhost:5432/investpulse} uv run alembic upgrade head

lint:
	pnpm lint
