# API

Authoritative

TL;DR
- 本文是 API 端点与示例请求的权威入口。
- 本地启动与回放流程以 `docs/RUNBOOK.md` 为准。
- 能力边界以 `docs/STATUS.md` 为准。

## Contract Entry

- OpenAPI JSON: `http://localhost:8000/openapi.json`
- Swagger UI: `http://localhost:8000/docs`

## Health

```bash
curl -s "http://localhost:8000/health" | jq
```

## Dashboard

```bash
curl -s "http://localhost:8000/dashboard?days=7&window=7d&limit=10&assets_window=24h&profile_id=1" | jq
```

Top-level fields include:
- `pending_extractions_count`
- `assets`
- `active_kols_7d`
- `clarity_ranking`
- `top_assets`

## Digest Endpoints

- Generate: `POST /digests/generate`
- Get latest/version by query: `GET /digests`
- List digest dates: `GET /digests/dates`
- Get by id: `GET /digests/{digest_id}`

Example (generate):
```bash
curl -s -X POST "http://localhost:8000/digests/generate?date=2026-02-24&days=7&profile_id=1" | jq
```

## Ingest Endpoints

- `POST /ingest/manual`
- `POST /ingest/x/convert`
- `POST /ingest/x/import`
- `POST /ingest/x/following/import`
- `GET /ingest/x/raw-posts/preview`
- `GET /ingest/x/progress`
- `POST /ingest/x/retry-failed`
- `POST /raw-posts`

Example (`/ingest/manual`):
```bash
curl -s -X POST "http://localhost:8000/ingest/manual" \
  -H "Content-Type: application/json" \
  --data '{
    "platform":"x",
    "author_handle":"alice",
    "url":"https://x.com/alice/status/1",
    "content_text":"BTC trend looks constructive"
  }' | jq
```

## Extraction And Review Endpoints

- `POST /raw-posts/{raw_post_id}/extract`
- `POST /raw-posts/extract-batch`
- `POST /extract-jobs`
- `GET /extract-jobs/{job_id}`
- `GET /extractions`
- `GET /extractions/{extraction_id}`
- `POST /extractions/{extraction_id}/approve`
- `POST /extractions/{extraction_id}/approve-batch`
- `POST /extractions/{extraction_id}/reject`
- `POST /extractions/{extraction_id}/re-extract`

Example (`/raw-posts/extract-batch`):
```bash
curl -s -X POST "http://localhost:8000/raw-posts/extract-batch" \
  -H "Content-Type: application/json" \
  --data '{"raw_post_ids":[1,2,3],"mode":"pending_only"}' | jq
```

## Profile / Asset / KOL Endpoints

- Profiles: `GET /profiles`, `GET /profiles/{profile_id}`, `PUT /profiles/{profile_id}/kols`, `PUT /profiles/{profile_id}/markets`
- Assets: `GET /assets`, `POST /assets`, `POST /assets/upsert`, aliases endpoints
- KOLs: `GET /kols`, `POST /kols`
- Views: `POST /kol-views`, `GET /assets/{asset_id}/views`, feed/timeline endpoints

## Admin Endpoints (Dev/Ops)

- Extraction repair/cleanup under `/admin/extractions/*`
- Hard delete endpoints under `/admin/*`

## Not Implemented

- Event reminder scheduling/triggering API.
- Prediction-market integration API.
