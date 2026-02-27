# Runbook

Authoritative

TL;DR
- 本文是本地运行与回放的唯一权威操作手册。
- 兼容旧路径的 `RUNBOOK_LOCAL*.txt` 已合并到本文件。
- 开发测试与验收命令请使用 `docs/DEV_WORKFLOW.md`。

## Preconditions

- 已复制 `apps/api/.env.example` 到 `apps/api/.env`。
- Docker 可用。
- 本地端口可用：`3000`、`8000`、`5432`、`5433`、`6379`。

## Local Start

1. Start infra
```bash
docker compose up -d db db_test redis
```

2. Start API
```bash
cd apps/api
uv sync
# Run migration command from docs/DEV_WORKFLOW.md first.
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

3. Start Web
```bash
cd apps/web
pnpm install
pnpm dev
```

4. Entry URLs
- API Swagger: `http://localhost:8000/docs`
- API OpenAPI: `http://localhost:8000/openapi.json`
- Web dashboard: `http://localhost:3000/dashboard`
- Web digest page: `http://localhost:3000/digests/2026-02-24`

## One Daily Workflow

1. Optional X import
```bash
curl -s -X POST "http://localhost:8000/ingest/x/import" \
  -H "Content-Type: application/json" \
  --data @/tmp/x_import.json | jq
```

2. Optional extraction batch
```bash
curl -s -X POST "http://localhost:8000/raw-posts/extract-batch" \
  -H "Content-Type: application/json" \
  --data '{"raw_post_ids":[1,2,3],"mode":"pending_only"}' | jq
```

3. Generate digest
```bash
curl -s -X POST "http://localhost:8000/digests/generate?date=2026-02-24&days=7&profile_id=1" | jq
```

## Replay (Authoritative)

1. Latest by date/profile
```bash
curl -s "http://localhost:8000/digests?date=2026-02-24&profile_id=1" | jq
```

2. Specific version
```bash
curl -s "http://localhost:8000/digests?date=2026-02-24&profile_id=1&version=1" | jq
```

3. By digest id
```bash
curl -s "http://localhost:8000/digests/1" | jq
```

4. Evidence URLs in digest payload
```bash
curl -s "http://localhost:8000/digests?date=2026-02-24&profile_id=1" | jq '..|.source_url? // empty'
```

## Local Admin Cleanup (Dev Only)

- Delete pending extractions
```bash
curl -s -X DELETE "http://localhost:8000/admin/extractions/pending?confirm=YES&status=pending" | jq
```

- Delete digest by date/profile
```bash
curl -s -X DELETE "http://localhost:8000/admin/digests?confirm=YES&digest_date=2026-02-24&profile_id=1" | jq
```

## Local Destructive Reset

```bash
./scripts/reset_db_local.sh
```

Warning: this permanently deletes local dev data volumes.

## Not Implemented

- Event reminder runbook (scheduler/trigger lifecycle).
