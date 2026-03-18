# RUNBOOK

Authoritative

## Install Dependencies
```bash
pnpm install
cd /home/zhoucookie/code/investpulse/apps/api && uv sync
```

## Start Infra
```bash
docker compose up -d db db_test redis
```

## Reset Dev DB Only
```bash
/home/zhoucookie/code/investpulse/scripts/reset_db_local.sh
```

This only targets the local dev database. Tests must still use `DATABASE_URL_TEST`.

## Run API
```bash
cd /home/zhoucookie/code/investpulse/apps/api
ENV=local DEBUG=true DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/investpulse uv run uvicorn main:app --reload
```

## Run Web
```bash
cd /home/zhoucookie/code/investpulse/apps/web
pnpm dev
```

## Background Helper

Install once:
```bash
mkdir -p ~/.local/bin
ln -sf /home/zhoucookie/code/investpulse/scripts/investpulse ~/.local/bin/investpulse
```

Common commands:
```bash
investpulse start
investpulse status
investpulse logs
investpulse logs api
investpulse logs web
investpulse restart
investpulse end
```

Current helper behavior:
- API default URL: `http://127.0.0.1:8000`
- Web default URL: `http://127.0.0.1:3000`
- logs go to `logs/api.log` and `logs/web.log`
- PID files go to `.runtime/`
- `start` brings up `db`, `db_test`, `redis` when Docker is available
- if WSL Docker is unavailable, it can fall back to Windows Docker Desktop
- API start runs `alembic upgrade head` before boot

Supported env overrides:
- `INVESTPULSE_API_PORT`
- `INVESTPULSE_WEB_PORT`
- `INVESTPULSE_DB_HOST`
- `INVESTPULSE_DB_PORT`
- `INVESTPULSE_OPEN_BROWSER=0`

## Basic Health Checks
```bash
curl -s http://127.0.0.1:8000/health | jq
curl -s http://127.0.0.1:8000/extractor-status | jq
```

## Inspect Extractions
```bash
curl -s "http://127.0.0.1:8000/extractions?limit=5" | jq
curl -s "http://127.0.0.1:8000/extractions/stats" | jq
curl -s "http://127.0.0.1:8000/extractions/1" | jq
```

Useful checks:
- confirm normalized keys are `as_of/source_url/islibrary/hasview/asset_views/library_entry`
- confirm `asset_views` rows below `confidence 70` do not survive normalization
- confirm parse-failed rows still exist as `pending` with `last_error`

## Inspect Ordered Parsed Payload
```bash
curl -s "http://127.0.0.1:8000/extractions?limit=3" | jq '.[] | {id, parsed_model_output}'
```

## Re-Extract One Row
```bash
curl -s -X POST "http://127.0.0.1:8000/extractions/123/re-extract" | jq
```

Expected behavior:
- replacement extraction is created for the same `raw_post`
- valid new AI results may remove older extraction rows for that raw post
- failed-semantics new results do not delete the old rows

## Import And Job Checks
```bash
curl -s "http://127.0.0.1:8000/ingest/x/progress" | jq
curl -s -X POST "http://127.0.0.1:8000/ingest/x/retry-failed" | jq
curl -s -X POST "http://127.0.0.1:8000/ingest/x/retry-pending-all" | jq
curl -s "http://127.0.0.1:8000/extract-jobs/<job_id>" | jq
```

Job fields to watch:
- `ai_call_used`
- `openai_call_attempted_count`
- `success_count`
- `failed_count`
- `skipped_count`
- `max_concurrency_used`
- `last_error_summary`

## Daily Digest
```bash
curl -s -X POST "http://127.0.0.1:8000/digests/generate?date=2026-03-18" | jq
curl -s "http://127.0.0.1:8000/digests?date=2026-03-18" | jq
curl -s "http://127.0.0.1:8000/digests/dates" | jq
curl -s "http://127.0.0.1:8000/digests/1" | jq
```

Current constraints:
- only the most recent 3 days are supported
- current public flow uses system `profile_id=1`

## Weekly Digest
```bash
curl -s -X POST "http://127.0.0.1:8000/weekly-digests/generate?kind=recent_week&date=2026-03-18" | jq
curl -s "http://127.0.0.1:8000/weekly-digests?kind=recent_week&anchor_date=2026-03-18" | jq
curl -s "http://127.0.0.1:8000/weekly-digests/dates?kind=recent_week" | jq
```

Current windows:
- `recent_week`: reference date and previous 6 days
- `this_week`: latest Sunday through reference date
- `last_week`: previous full Sunday-Saturday

## Portfolio Advice
```bash
curl -s -X POST "http://127.0.0.1:8000/portfolio/advice" \
  -H "Content-Type: application/json" \
  -d '{
    "user_goal": "控制回撤并优化调仓节奏",
    "holdings": [
      {
        "asset_id": 101,
        "symbol": "BTC",
        "holding_reason_text": "中长期继续观察资金流与结构",
        "sell_timing_text": "若破位并且风险证据持续增加则分批减仓",
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

## Admin Repair Endpoints
```bash
curl -s -X POST "http://127.0.0.1:8000/admin/extractions/cleanup-json?confirm=YES&days=3650&limit=5000&dry_run=true" | jq
curl -s -X POST "http://127.0.0.1:8000/admin/extractions/refresh-wrong-extracted-json?confirm=YES&days=365&limit=2000&dry_run=true" | jq
curl -s -X POST "http://127.0.0.1:8000/admin/extractions/recompute-statuses?confirm=YES&days=365&limit=5000&dry_run=true" | jq
curl -s -X POST "http://127.0.0.1:8000/admin/fix/approved-missing-views?confirm=YES&days=365&limit=2000&dry_run=true" | jq
```

## Full Validation
```bash
DATABASE_URL_TEST=postgresql+asyncpg://postgres:postgres@localhost:5433/investpulse_test make verify
```
