## Offline X Import Converter

Use `scripts/x_import_converter.py` to transform CSV/JSON exports from tools such as twitter-web-exporter / TwExportly / TweetExtract into the JSON array required by `POST /ingest/x/import`.

### Supported CLI arguments

```bash
uv run python scripts/x_import_converter.py \
  --input scripts/examples/x_export_sample.csv \
  --output /tmp/x_import.json \
  --author_handle qinbafrank \
  --kol_id 123 \
  --start_date 2026-02-19 \
  --end_date 2026-02-23
```

- `--input <path>`: source export file (`.csv` or `.json`)
- `--output <path>`: output path, default `x_import.json`
- `--input_format csv|json`: optional override; otherwise inferred from extension
- `--author_handle <handle>`: optional override for all output rows (recommended for consistency)
- `--kol_id <int>`: optional kol id, copied to every output row
- `--start_date YYYY-MM-DD`: inclusive start date filter
- `--end_date YYYY-MM-DD`: inclusive end date filter
- `--no_raw_json`: skip embedding original row as `raw_json`

The converter prints a summary JSON to stdout, including skip counts and dedup info.

### Mapping behavior

- `external_id`: prefers `tweet_id/id/status_id/post_id`; otherwise extracts from URL `/status/<id>`
- `url`: uses exported URL; if missing, builds `https://x.com/{author_handle}/status/{external_id}`
- `posted_at`: parses common datetime fields into ISO8601 UTC (`...Z`)
- `content_text`: prefers `content_text/full_text/tweet_text/text/content/body`
- `author_handle`: CLI `--author_handle` first, otherwise uses exported handle fields
- `raw_json`: embeds original row by default
- `kol_id`: added when `--kol_id` is provided

### Filter and dedup strategy

- keeps rows where `start_date <= posted_at_date <= end_date` (inclusive)
- dedup key is `external_id`
- dedup strategy: `keep_first_by_external_id`

### Import to API and generate digest

```bash
curl -s -X POST "http://localhost:8000/ingest/x/import" \
  -H "Content-Type: application/json" \
  --data @/tmp/x_import.json | jq

# optional: batch extract after import-only mode
curl -s -X POST "http://localhost:8000/raw-posts/extract-batch" \
  -H "Content-Type: application/json" \
  --data '{"raw_post_ids":[1,2,3],"mode":"pending_only"}' | jq

curl -s -X POST "http://localhost:8000/digests/generate?date=2026-02-23&days=7&to_ts=2026-02-23T23:59:59Z&profile_id=1" | jq
curl -s "http://localhost:8000/digests?date=2026-02-23&profile_id=1" | jq '..|.source_url? // empty'
```

## Local DB Reset (one-time dev only)

`scripts/reset_db_local.sh` resets local DB/Redis and re-runs migrations.

```bash
./scripts/reset_db_local.sh
```

What it does:

- `docker compose down -v`
- `docker compose up -d db redis`
- `alembic upgrade head` (runs under `apps/api`, prefers `uv run` when available)

Warning: this script permanently clears local data volumes. Do not use in shared/staging/production environments.
