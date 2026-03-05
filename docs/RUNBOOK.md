# Runbook

Authoritative

TL;DR
- 本文是本地运行与回放的唯一权威操作手册。
- 开发测试与验收命令请使用 `docs/DEV_WORKFLOW.md`。

## Quickstart (Copy/Paste)

### A) Reset Local DB (DANGEROUS)

仅用于本地开发；会删除并重建本地数据库数据卷。  
严禁在生产环境执行；测试请使用 `DATABASE_URL_TEST`。  
默认开发库：`postgresql+asyncpg://postgres:postgres@localhost:5432/investpulse`。

```bash
set -euo pipefail
cd /home/zhoucookie/code/investpulse
export DATABASE_URL="${DATABASE_URL:-postgresql+asyncpg://postgres:postgres@localhost:5432/investpulse}"
[[ "$DATABASE_URL" == *"localhost:5432/investpulse"* ]] || { echo "Refusing non-local DATABASE_URL: $DATABASE_URL"; exit 1; }
/home/zhoucookie/code/investpulse/scripts/reset_db_local.sh
```

### B) Start Local Stack (DB + Redis + Migrate + API)

用于本地端到端启动：拉起 `db/redis`、执行迁移并启动 API。  
测试请使用 `DATABASE_URL_TEST`，不要复用开发库。

```bash
set -euo pipefail
cd /home/zhoucookie/code/investpulse
docker compose up -d db redis
cd /home/zhoucookie/code/investpulse/apps/api
uv sync
ENV=local DEBUG=false DATABASE_URL="${DATABASE_URL:-postgresql+asyncpg://postgres:postgres@localhost:5432/investpulse}" uv run alembic upgrade head
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Health Checks

API 启动后在另一个终端执行：

```bash
curl -fsS http://localhost:8000/health
curl -fsS http://localhost:8000/extractor-status
```

## Preconditions

- 已复制 `apps/api/.env.example` 到 `apps/api/.env`。
- Docker 可用。
- 本地端口可用：`3000`、`8000`、`5432`、`5433`、`6379`。
- 默认模型为 `OPENAI_MODEL=deepseek/deepseek-v3.2`，`OPENAI_BASE_URL=https://openrouter.ai/api/v1`。

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
- Extractor status: `http://localhost:8000/extractor-status`

5. Verify runtime model/base_url
```bash
curl -s "http://localhost:8000/extractor-status" | jq '{mode, default_model, base_url, max_output_tokens}'
```

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

4. Quick evidence-chain check
```bash
curl -s "http://localhost:8000/extractions?limit=1" | jq '.[0] | {raw_post_id, model_name, extractor_name, source_url: .extracted_json.source_url, content_kind: .extracted_json.content_kind, meta: .extracted_json.meta}'
curl -s "http://localhost:8000/digests?date=2026-02-24&profile_id=1" | jq '.per_asset_summary[0].top_views_bull[0].source_url'
```

5. Extraction semantic checks (latest-only / NoneAny / library)
```bash
# latest-only: 默认每个 raw_post 仅返回最新 extraction
curl -s "http://localhost:8000/extractions?status=all&limit=20&offset=0" | jq 'length'
curl -s "http://localhost:8000/extractions?status=all&show_history=true&limit=20&offset=0" | jq 'length'

# 观察 NoneAny/content_kind/library_entry/meta 可观测字段
curl -s "http://localhost:8000/extractions?limit=5" | jq '.[] | {id, status, last_error, assets: .extracted_json.assets, content_kind: .extracted_json.content_kind, library_entry: .extracted_json.library_entry, asset_views: .extracted_json.asset_views, summary_language: .extracted_json.meta.summary_language, summary_language_violation: .extracted_json.meta.summary_language_violation, extra_retry_budget_total: .extracted_json.meta.extra_retry_budget_total, extra_retry_budget_used: .extracted_json.meta.extra_retry_budget_used, invalid_json_retry_used: .extracted_json.meta.invalid_json_retry_used, truncated_retry_used: .extracted_json.meta.truncated_retry_used, asset_views_min_confidence_threshold: .extracted_json.meta.asset_views_min_confidence_threshold, asset_views_dropped_low_confidence_count: .extracted_json.meta.asset_views_dropped_low_confidence_count, auto_review_reason: .extracted_json.meta.auto_review_reason, auto_policy_applied: .extracted_json.meta.auto_policy_applied, content_kind_raw: .extracted_json.meta.content_kind_raw, content_kind_original: .extracted_json.meta.content_kind_original, library_entry_dropped: .extracted_json.meta.library_entry_dropped, library_entry_drop_reason: .extracted_json.meta.library_entry_drop_reason, library_downgraded: .extracted_json.meta.library_downgraded, library_downgrade_reason: .extracted_json.meta.library_downgrade_reason}'
```

示例语义（已实现）：
- NVDA 财报深度解读：`content_kind=asset` + `asset_views` 含 `NVDA`（且仅保留 `confidence>=70` 的 item）
- 纯生活贴：`content_kind=asset` + `assets=[NoneAny]` + `library_entry=null`
- 无具体标的宏观深度贴：`content_kind=library` + `assets=[NoneAny]` + `asset_views=[]` + `library_entry(confidence>=80,tags=["macro"])`
- 若模型声明 `content_kind=library` 但 `library_entry` 不满足门槛，会降级为 `content_kind=asset`，并写 `library_downgrade_reason`（`low_library_confidence/invalid_library_tags/invalid_library_shape`）。
- `content_kind=asset` 时，`assets` 会与最终 `asset_views` 同步；过滤后 `asset_views=[]` 时，`assets=[NoneAny]`。
- auto-review 阈值来源：`library` 用 `library_entry.confidence`；`asset` 用顶层 `confidence`。
- `content_kind=asset` 且 `assets=[NoneAny]` 的强拒规则保持不变（不放宽）。
- text_json `invalid_json` 会触发一次纠错重试（强制合法 JSON 字符串转义）；与截断重试/summary 中文纠正重试共享预算=1，互斥。

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

See `Quickstart (Copy/Paste)` -> `A) Reset Local DB (DANGEROUS)`.

## Not Implemented

- Event reminder runbook (scheduler/trigger lifecycle).
