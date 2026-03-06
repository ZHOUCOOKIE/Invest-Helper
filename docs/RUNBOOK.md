# RUNBOOK

## Start Local Services
```bash
docker compose up -d db db_test redis
```

## One-Command Reset Dev DB Only (Keep Test DB)
```bash
bash -lc 'cd /home/zhoucookie/code/investpulse && docker compose exec -T db psql -U postgres -d postgres -c "DROP DATABASE IF EXISTS investpulse WITH (FORCE);" && docker compose exec -T db psql -U postgres -d postgres -c "CREATE DATABASE investpulse;" && cd /home/zhoucookie/code/investpulse/apps/api && ENV=local DEBUG=false DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/investpulse uv run alembic upgrade head'
```

## API Run
```bash
cd /home/zhoucookie/code/investpulse/apps/api
ENV=local DEBUG=true DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/investpulse uv run uvicorn main:app --reload
```

## One-Command Start API + Web
```bash
bash -lc 'cd /home/zhoucookie/code/investpulse/apps/api && ENV=local DEBUG=true DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/investpulse uv run uvicorn main:app --reload & cd /home/zhoucookie/code/investpulse/apps/web && pnpm dev'
```

## Verify Extraction Output (latest)
```bash
curl -s "http://localhost:8000/extractions?limit=1" | jq '.[0].extracted_json'
```
Expected business keys:
- `as_of`
- `source_url`
- `islibrary`
- `hasview`
- `asset_views`
- `library_entry`

## Inspect Normalize/Auto-Review Meta
```bash
curl -s "http://localhost:8000/extractions?limit=5" | jq '.[] | {id,status,last_error,extracted_json:{as_of:.extracted_json.as_of,source_url:.extracted_json.source_url,islibrary:.extracted_json.islibrary,hasview:.extracted_json.hasview,asset_views:.extracted_json.asset_views,library_entry:.extracted_json.library_entry},meta:.extracted_json.meta}'
```

## Typical Checks
- Enum strictness:
  - `market` only: `CRYPTO|STOCK|ETF|FOREX|OTHER` (legacy auto enum removed)
  - `stance` only: `bull|bear|neutral`
  - `horizon` only: `intraday|1w|1m|3m|1y`
  - no alias-map keyword/synonym normalization for these enum fields
- Asset post:
  - `islibrary=0`
  - `asset_views` only contains `confidence>=70` (prompt 文本要求模型输出 `>=80`)
  - `hasview` 由归一化后的 `asset_views` 自动回填
- Library post:
  - `islibrary=1`
  - valid `library_entry={tag,summary}`
  - `library_entry.summary` 必须是 `测试`
- Invalid library entry:
  - downgraded to `islibrary=0`
  - `library_entry=null`

## Generate And Replay Digest
```bash
# generate (overwrite same profile_id + date)
curl -s -X POST "http://localhost:8000/digests/generate?date=2026-03-06&profile_id=1" | jq

# replay by date/profile
curl -s "http://localhost:8000/digests?date=2026-03-06&profile_id=1" | jq

# list replayable dates
curl -s "http://localhost:8000/digests/dates?profile_id=1" | jq

# replay by digest id
curl -s "http://localhost:8000/digests/1" | jq
```

## Full Validation
```bash
DATABASE_URL_TEST=postgresql+asyncpg://postgres:postgres@localhost:5433/investpulse_test make verify
```
