# API

Reference

TL;DR
- OpenAPI is available at `/openapi.json`, Swagger UI at `/docs`.
- This file provides executable curl examples for core workflows.
- For capability boundary, trust `docs/STATUS.md`.

## How To Inspect API Contract

- OpenAPI JSON: `http://localhost:8000/openapi.json`
- Swagger UI: `http://localhost:8000/docs`

## Health

Request
```bash
curl -s "http://localhost:8000/health" | jq
```

Sample response
```json
{"ok": true}
```

## Dashboard

Request
```bash
curl -s "http://localhost:8000/dashboard?days=7&window=7d&limit=10&assets_window=24h&profile_id=1" | jq
```

Response fields (top-level, abbreviated)
- `pending_extractions_count`
- `assets`
- `active_kols_7d`
- `clarity_ranking`
- `top_assets`

## Digest Generate / Replay

Generate
```bash
curl -s -X POST "http://localhost:8000/digests/generate?date=2026-02-24&days=7&profile_id=1" | jq
```

Get latest digest by date
```bash
curl -s "http://localhost:8000/digests?date=2026-02-24&profile_id=1" | jq
```

Get specific version
```bash
curl -s "http://localhost:8000/digests?date=2026-02-24&profile_id=1&version=1" | jq
```

List available digest dates
```bash
curl -s "http://localhost:8000/digests/dates?profile_id=1" | jq
```

Get by digest id
```bash
curl -s "http://localhost:8000/digests/1" | jq
```

## Ingest And Extraction

Manual ingest
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

Extract batch
```bash
curl -s -X POST "http://localhost:8000/raw-posts/extract-batch" \
  -H "Content-Type: application/json" \
  --data '{"raw_post_ids":[1,2,3],"mode":"pending_only"}' | jq
```

Create async extract job
```bash
curl -s -X POST "http://localhost:8000/extract-jobs" \
  -H "Content-Type: application/json" \
  --data '{"raw_post_ids":[1,2,3],"mode":"pending_or_failed"}' | jq
```

Get job status
```bash
curl -s "http://localhost:8000/extract-jobs/<job_id>" | jq
```

## Review Endpoints

Approve
```bash
curl -s -X POST "http://localhost:8000/extractions/1/approve" \
  -H "Content-Type: application/json" \
  --data '{
    "kol_id":1,
    "asset_id":1,
    "stance":"bull",
    "horizon":"1w",
    "confidence":70,
    "summary":"momentum improving",
    "source_url":"https://x.com/alice/status/1",
    "as_of":"2026-02-24"
  }' | jq
```

Reject
```bash
curl -s -X POST "http://localhost:8000/extractions/1/reject" \
  -H "Content-Type: application/json" \
  --data '{"reason":"noise"}' | jq
```

Re-extract
```bash
curl -s -X POST "http://localhost:8000/extractions/1/re-extract" | jq
```

## Profile Rules

List profiles
```bash
curl -s "http://localhost:8000/profiles" | jq
```

Get profile detail
```bash
curl -s "http://localhost:8000/profiles/1" | jq
```

Update profile markets
```bash
curl -s -X PUT "http://localhost:8000/profiles/1/markets" \
  -H "Content-Type: application/json" \
  --data '{"markets":["CRYPTO","STOCK"]}' | jq
```

## Non-goal Note

- Reddit-specific API pipeline is not part of current core scope.
