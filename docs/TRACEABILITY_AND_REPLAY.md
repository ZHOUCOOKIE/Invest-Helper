# Traceability And Replay

Authoritative

TL;DR
- Every output should be traceable to source evidence.
- Daily digest replay is versioned by `profile_id + digest_date + version`.
- Use this file with `docs/RUNBOOK.md` for operational checks.

## Evidence Chain

1. Raw source (`raw_posts`)
- key fields: `platform`, `external_id`, `url`, `content_text`, `posted_at`, `raw_json`

2. Extraction record (`post_extractions`)
- linkage: `raw_post_id`
- audit: `prompt_version`, `prompt_text`, `prompt_hash`, `raw_model_output`, `parsed_model_output`
- status/result: `status`, `extracted_json`, `last_error`

3. Approved view (`kol_views`)
- linkage/evidence: `source_url`, `kol_id`, `asset_id`, `horizon`, `stance`, `confidence`, `as_of`

4. Daily output (`daily_digests`)
- replay keys: `profile_id`, `digest_date`, `version`, `content`

## Replay Commands

Latest by date/profile
```bash
curl -s "http://localhost:8000/digests?date=2026-02-24&profile_id=1" | jq
```

Specific version
```bash
curl -s "http://localhost:8000/digests?date=2026-02-24&profile_id=1&version=1" | jq
```

By digest id
```bash
curl -s "http://localhost:8000/digests/1" | jq
```

Date catalog
```bash
curl -s "http://localhost:8000/digests/dates?profile_id=1" | jq
```

Evidence URLs from digest payload
```bash
curl -s "http://localhost:8000/digests?date=2026-02-24&profile_id=1" | jq '..|.source_url? // empty'
```

## Versioning Policy

- Unique key: `(profile_id, digest_date, version)`.
- `POST /digests/generate` increments version for same date/profile.
- `GET /digests` without `version` returns latest version.

## Time-Field Policy

Digest generation uses business-time fallback:
- `as_of` (preferred)
- `posted_at` fallback
- `created_at` fallback

Returned metadata includes `time_field_used`.

## Not Implemented

- Event reminder entity and replay lifecycle.
