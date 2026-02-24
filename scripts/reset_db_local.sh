#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cat <<'WARN'
[WARNING] This will delete local database volumes and clear all data.
Only use for one-time LOCAL development reset.
WARN

cd "$ROOT_DIR"

docker compose down -v
docker compose up -d db redis

cd "$ROOT_DIR/apps/api"
if command -v uv >/dev/null 2>&1; then
  uv run alembic upgrade head
else
  alembic upgrade head
fi

echo "Local DB reset completed."
