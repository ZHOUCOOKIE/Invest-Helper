# RUNBOOK

## Start Local Services
```bash
docker compose up -d db db_test redis
```

## Install Dependencies
```bash
pnpm install
cd /home/zhoucookie/code/investpulse/apps/api && uv sync
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

## Background Start/Stop Helper
One-time install into current WSL user PATH:
```bash
mkdir -p ~/.local/bin
ln -sf /home/zhoucookie/code/investpulse/scripts/investpulse ~/.local/bin/investpulse
```

Start in background and open browser tabs for Web + API docs:
```bash
investpulse start
```

Stop background services:
```bash
investpulse end
```

Check status / logs:
```bash
investpulse status
investpulse logs
investpulse logs api
investpulse logs web
investpulse restart
```
Behavior:
- API runs at `http://localhost:8000`
- Web runs at `http://localhost:3000`
- logs are written to `logs/api.log` and `logs/web.log`
- PID files are written to `.runtime/`
- `start` tries `docker compose up -d db db_test redis` first; if WSL Docker integration is unavailable, it falls back to Windows Docker Desktop when available
- API startup runs `alembic upgrade head` against dev DB `postgresql+asyncpg://postgres:postgres@localhost:5432/investpulse`
- supported env overrides:
  - `INVESTPULSE_API_PORT`
  - `INVESTPULSE_WEB_PORT`
  - `INVESTPULSE_DB_HOST`
  - `INVESTPULSE_DB_PORT`
  - `INVESTPULSE_OPEN_BROWSER=0` to disable auto-open browser tabs

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
Canonical key order:
- top-level: `as_of, source_url, islibrary, hasview, asset_views, library_entry` (`meta` optional at tail)
- `asset_views[*]`: `symbol, market, stance, horizon, confidence, summary`

## Inspect Normalize/Auto-Review Meta
```bash
curl -s "http://localhost:8000/extractions?limit=5" | jq '.[] | {id,status,last_error,extracted_json:{as_of:.extracted_json.as_of,source_url:.extracted_json.source_url,islibrary:.extracted_json.islibrary,hasview:.extracted_json.hasview,asset_views:.extracted_json.asset_views,library_entry:.extracted_json.library_entry},meta:.extracted_json.meta}'
```
Note:
- `meta` is intentionally compact now; many legacy/debug counters are removed during read normalization
- rows without meaningful meta will omit `extracted_json.meta`

## Inspect Extraction Repository Counts
```bash
curl -s "http://localhost:8000/extractions/stats" | jq
```
Returns:
- `raw_posts_count`
- `post_extractions_count`
- `duplicate_raw_post_count`

## Inspect Parsed Model Output (ordered)
```bash
curl -s "http://localhost:8000/extractions?limit=5" | jq '.[] | {id,parsed_model_output}'
```
Expected `parsed_model_output` order:
- `as_of, source_url, islibrary, hasview, asset_views, library_entry`
- stored as DB `JSON` (not `JSONB`) to keep insertion order

## Cleanup Historical Extraction JSON/Parsed JSON
```bash
# dry-run
curl -s -X POST "http://localhost:8000/admin/extractions/cleanup-json?confirm=YES&days=3650&limit=5000&dry_run=true" | jq

# apply
curl -s -X POST "http://localhost:8000/admin/extractions/cleanup-json?confirm=YES&days=3650&limit=5000&dry_run=false" | jq
```

## Refresh Approved Rows With Missing `asset_views`
```bash
# dry-run
curl -s -X POST "http://localhost:8000/admin/extractions/refresh-wrong-extracted-json?confirm=YES&days=365&limit=2000&dry_run=true" | jq

# apply
curl -s -X POST "http://localhost:8000/admin/extractions/refresh-wrong-extracted-json?confirm=YES&days=365&limit=2000&dry_run=false" | jq
```

## Recompute Extraction Statuses
```bash
# dry-run
curl -s -X POST "http://localhost:8000/admin/extractions/recompute-statuses?confirm=YES&days=365&limit=5000&dry_run=true" | jq

# apply
curl -s -X POST "http://localhost:8000/admin/extractions/recompute-statuses?confirm=YES&days=365&limit=5000&dry_run=false" | jq
```

## Force Replace One Extraction
```bash
curl -s -X POST "http://localhost:8000/extractions/123/re-extract" | jq
```
Behavior:
- creates one replacement extraction for the same `raw_post`
- when the new AI result is valid, older extraction rows for that `raw_post` are deleted together with `kol_views` referenced only by those deleted rows
- when the new extraction is still failed semantics, old rows are kept

## Inspect Async Extract Job
```bash
curl -s "http://localhost:8000/extract-jobs/<job_id>" | jq
```
Key fields:
- `ai_call_used`
- `openai_call_attempted_count`
- `success_count`
- `failed_count`
- `skipped_count`
- `max_concurrency_used`
- `last_error_summary`

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
# generate (overwrite same profile_id + date; current API writes profile_id=1)
curl -s -X POST "http://localhost:8000/digests/generate?date=2026-03-06" | jq

# replay by date (current API reads profile_id=1)
curl -s "http://localhost:8000/digests?date=2026-03-06" | jq

# list replayable dates
curl -s "http://localhost:8000/digests/dates" | jq

# replay by digest id
curl -s "http://localhost:8000/digests/1" | jq
```

## Generate And Replay Weekly Digest
```bash
# generate weekly digest (kind: recent_week|this_week|last_week)
curl -s -X POST "http://localhost:8000/weekly-digests/generate?kind=recent_week&date=2026-03-06" | jq

# replay by kind + anchor_date
curl -s "http://localhost:8000/weekly-digests?kind=recent_week&anchor_date=2026-03-06" | jq

# list replayable anchor dates by kind
curl -s "http://localhost:8000/weekly-digests/dates?kind=recent_week" | jq
```

## Portfolio Advice
```bash
curl -s -X POST "http://localhost:8000/portfolio/advice" \
  -H "Content-Type: application/json" \
  -d '{
    "user_goal": "控制回撤并优化调仓节奏",
    "holdings": [
      {
        "asset_id": 101,
        "symbol": "BTC",
        "holding_reason_text": "中长期看好资金持续流入",
        "sell_timing_text": "若结构破位且风险证据增多则分批减仓",
        "support_citations": [
          {
            "source_url": "https://x.com/example/post/1",
            "summary": "机构配置倾向增强"
          }
        ],
        "risk_citations": [
          {
            "source_url": "https://x.com/example/post/2",
            "summary": "短线波动率放大"
          }
        ]
      }
    ]
  }' | jq
```

## Full Validation
```bash
DATABASE_URL_TEST=postgresql+asyncpg://postgres:postgres@localhost:5433/investpulse_test make verify
```
