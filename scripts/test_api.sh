#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

docker compose up -d db_test >/dev/null

for i in {1..30}; do
  if docker compose exec -T db_test pg_isready -U postgres -d investpulse_test >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

export ENV=test
export DEBUG=true
export DATABASE_URL_TEST="${DATABASE_URL_TEST:-postgresql+asyncpg://postgres:postgres@localhost:5433/investpulse_test}"
export DATABASE_URL="$DATABASE_URL_TEST"

cd "$ROOT_DIR/apps/api"
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q "$@"
