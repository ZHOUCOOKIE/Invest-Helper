from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, date, datetime
import io
import json
from pathlib import Path
import re
import sys
from typing import Any

STATUS_ID_RE = re.compile(r"/status/([^/?#]+)")

ID_KEYS = (
    "external_id",
    "tweet_id",
    "tweetId",
    "id",
    "status_id",
    "post_id",
)
URL_KEYS = (
    "url",
    "tweet_url",
    "status_url",
    "permalink",
    "link",
)
TEXT_KEYS = (
    "content_text",
    "full_text",
    "tweet_text",
    "text",
    "content",
    "body",
)
TITLE_KEYS = (
    "title",
    "tweet_title",
    "headline",
    "subject",
)
HANDLE_KEYS = (
    "author_handle",
    "handle",
    "username",
    "screen_name",
    "user_handle",
    "author",
)
ROW_KEYS = (
    "row",
    "tweet",
    "post",
)
TIME_KEYS = (
    "posted_at",
    "created_at",
    "createdAt",
    "tweet_created_at",
    "timestamp",
    "time",
    "date",
)

TIME_PATTERNS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y/%m/%d %H:%M",
    "%Y-%m-%d",
    "%a %b %d %H:%M:%S %z %Y",
)


@dataclass
class ConvertError:
    row_index: int
    reason: str
    external_id: str | None = None
    url: str | None = None


@dataclass
class ConvertStats:
    input_count: int = 0
    output_count: int = 0
    failed_count: int = 0
    skipped_missing_id: int = 0
    skipped_missing_time: int = 0
    skipped_missing_text: int = 0
    skipped_missing_handle: int = 0
    skipped_date_range: int = 0
    dedup_skipped: int = 0
    errors: list[ConvertError] | None = None

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []

    def add_error(
        self,
        *,
        row_index: int,
        reason: str,
        external_id: str | None = None,
        url: str | None = None,
    ) -> None:
        if self.errors is None:
            self.errors = []
        self.errors.append(
            ConvertError(
                row_index=row_index,
                reason=reason,
                external_id=(external_id or None),
                url=(url or None),
            )
        )


def _pick(row: dict[str, Any], keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _normalize_handle(value: str) -> str:
    return value.strip().lstrip("@").lower()


def _extract_status_id_from_url(url: str) -> str | None:
    matched = STATUS_ID_RE.search(url)
    if not matched:
        return None
    return matched.group(1)


def _parse_datetime_value(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        timestamp = float(value)
        if timestamp > 10_000_000_000:
            timestamp = timestamp / 1000.0
        return datetime.fromtimestamp(timestamp, tz=UTC)

    raw = str(value).strip()
    if not raw:
        return None
    if raw.isdigit():
        return _parse_datetime_value(int(raw))

    upper_raw = raw.upper()
    if upper_raw.endswith(" UTC"):
        raw = f"{raw[:-4]}+00:00"
    elif upper_raw.endswith(" GMT"):
        raw = f"{raw[:-4]}+00:00"

    iso_candidate = raw.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(iso_candidate)
    except ValueError:
        pass

    for pattern in TIME_PATTERNS:
        try:
            parsed = datetime.strptime(raw, pattern)
            return parsed
        except ValueError:
            continue
    return None


def _coerce_iso8601(value: datetime) -> tuple[datetime, str]:
    if value.tzinfo is None:
        as_utc = value.replace(tzinfo=UTC)
    else:
        as_utc = value.astimezone(UTC)
    return as_utc, as_utc.isoformat().replace("+00:00", "Z")


def _iter_json_records(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return list(payload)
    if not isinstance(payload, dict):
        return []

    for key in ("tweets", "items", "data", "posts", "results", *ROW_KEYS):
        candidate = payload.get(key)
        if isinstance(candidate, list):
            return list(candidate)

    nested = payload.get("data")
    if isinstance(nested, dict):
        for key in ("tweets", "items", "posts", *ROW_KEYS):
            candidate = nested.get(key)
            if isinstance(candidate, list):
                return list(candidate)

    return []


def load_records(input_path: Path, input_format: str | None = None) -> list[Any]:
    resolved_format = input_format
    if resolved_format is None:
        suffix = input_path.suffix.lower().lstrip(".")
        resolved_format = suffix if suffix in {"csv", "json"} else "csv"

    if resolved_format == "csv":
        with input_path.open("r", encoding="utf-8-sig", newline="") as fp:
            reader = csv.DictReader(fp)
            return [dict(row) for row in reader]

    with input_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    return _iter_json_records(payload)


def load_records_from_bytes(
    content: bytes,
    *,
    input_format: str | None = None,
    filename: str | None = None,
) -> list[Any]:
    resolved_format = input_format
    if resolved_format is None and filename:
        suffix = Path(filename).suffix.lower().lstrip(".")
        if suffix in {"csv", "json"}:
            resolved_format = suffix

    if resolved_format is None:
        stripped = content.lstrip()
        if stripped.startswith(b"{") or stripped.startswith(b"["):
            resolved_format = "json"
        else:
            resolved_format = "csv"

    if resolved_format == "csv":
        text = content.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
        return [dict(row) for row in reader]

    payload = json.loads(content.decode("utf-8"))
    return _iter_json_records(payload)


def convert_records(
    rows: list[Any],
    *,
    author_handle: str | None,
    kol_id: int | None,
    start_date: date | None,
    end_date: date | None,
    include_raw_json: bool = True,
) -> tuple[list[dict[str, Any]], ConvertStats]:
    stats = ConvertStats(input_count=len(rows))
    output: list[dict[str, Any]] = []
    seen_external_ids: set[str] = set()

    for idx, row_any in enumerate(rows, start=1):
        if not isinstance(row_any, dict):
            stats.failed_count += 1
            stats.add_error(row_index=idx, reason=f"row is not an object (type={type(row_any).__name__})")
            continue
        row = row_any

        raw_handle = author_handle or _pick(row, HANDLE_KEYS)
        if not isinstance(raw_handle, str) or not raw_handle.strip():
            stats.skipped_missing_handle += 1
            stats.failed_count += 1
            stats.add_error(row_index=idx, reason="missing author_handle/screen_name")
            continue
        final_handle = _normalize_handle(raw_handle)

        url_value = _pick(row, URL_KEYS)
        url = str(url_value).strip() if isinstance(url_value, str) else ""

        external_id_value = _pick(row, ID_KEYS)
        external_id = str(external_id_value).strip() if external_id_value is not None else ""
        if not external_id and url:
            from_url = _extract_status_id_from_url(url)
            external_id = from_url or ""
        if not external_id:
            stats.skipped_missing_id += 1
            stats.failed_count += 1
            stats.add_error(row_index=idx, reason="missing external_id and cannot infer from url", url=url or None)
            continue

        if not url:
            url = f"https://x.com/{final_handle}/status/{external_id}"

        posted_raw = _pick(row, TIME_KEYS)
        posted_dt = _parse_datetime_value(posted_raw)
        if posted_dt is None:
            stats.skipped_missing_time += 1
            stats.failed_count += 1
            stats.add_error(
                row_index=idx,
                reason=f"invalid posted_at/created_at value: {posted_raw!r}",
                external_id=external_id,
                url=url or None,
            )
            continue
        posted_utc, posted_iso = _coerce_iso8601(posted_dt)
        posted_day = posted_utc.date()
        if start_date and posted_day < start_date:
            stats.skipped_date_range += 1
            stats.failed_count += 1
            stats.add_error(
                row_index=idx,
                reason=f"posted_at out of range: {posted_day.isoformat()} < start_date {start_date.isoformat()}",
                external_id=external_id,
                url=url or None,
            )
            continue
        if end_date and posted_day > end_date:
            stats.skipped_date_range += 1
            stats.failed_count += 1
            stats.add_error(
                row_index=idx,
                reason=f"posted_at out of range: {posted_day.isoformat()} > end_date {end_date.isoformat()}",
                external_id=external_id,
                url=url or None,
            )
            continue

        text_value = _pick(row, TEXT_KEYS)
        title_value = _pick(row, TITLE_KEYS)
        text = str(text_value).strip() if text_value is not None else ""
        title = str(title_value).strip() if title_value is not None else ""
        if not text and title:
            text = title
        elif not text and not title:
            fallback_parts = []
            for key in ("title", "text"):
                value = row.get(key)
                if isinstance(value, str) and value.strip():
                    fallback_parts.append(value.strip())
            if fallback_parts:
                text = "\n".join(dict.fromkeys(fallback_parts))
        if not text:
            stats.skipped_missing_text += 1
            stats.failed_count += 1
            stats.add_error(
                row_index=idx,
                reason="missing content_text/text/full_text/title",
                external_id=external_id,
                url=url or None,
            )
            continue

        if external_id in seen_external_ids:
            stats.dedup_skipped += 1
            stats.failed_count += 1
            stats.add_error(
                row_index=idx,
                reason=f"duplicate external_id in file: {external_id}",
                external_id=external_id,
                url=url or None,
            )
            continue
        seen_external_ids.add(external_id)

        item: dict[str, Any] = {
            "external_id": external_id,
            "author_handle": final_handle,
            "resolved_author_handle": final_handle,
            "url": url,
            "posted_at": posted_iso,
            "content_text": text,
        }
        if kol_id is not None:
            item["kol_id"] = kol_id
        if include_raw_json:
            item["raw_json"] = row
        output.append(item)

    output.sort(key=lambda item: (item["posted_at"], item["external_id"]))
    stats.output_count = len(output)
    stats.failed_count = stats.input_count - stats.output_count if stats.failed_count == 0 else stats.failed_count
    return output, stats


def _parse_date_arg(value: str | None) -> date | None:
    if value is None:
        return None
    return date.fromisoformat(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert X export CSV/JSON into /ingest/x/import payload JSON.")
    parser.add_argument("--input", required=True, help="Input export file path (.csv or .json)")
    parser.add_argument("--output", default="x_import.json", help="Output JSON path")
    parser.add_argument("--input_format", choices=["csv", "json"], default=None, help="Optional input format override")
    parser.add_argument("--author_handle", default=None, help="Override author handle for all rows")
    parser.add_argument("--kol_id", type=int, default=None, help="Optional kol_id to write into each output row")
    parser.add_argument("--start_date", default=None, help="Inclusive start date in YYYY-MM-DD")
    parser.add_argument("--end_date", default=None, help="Inclusive end date in YYYY-MM-DD")
    parser.add_argument("--no_raw_json", action="store_true", help="Do not include raw_json field in output")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"error: input file not found: {input_path}", file=sys.stderr)
        return 2

    try:
        start_date = _parse_date_arg(args.start_date)
        end_date = _parse_date_arg(args.end_date)
    except ValueError as exc:
        print(f"error: invalid date argument: {exc}", file=sys.stderr)
        return 2

    if start_date and end_date and start_date > end_date:
        print("error: start_date must be <= end_date", file=sys.stderr)
        return 2

    try:
        rows = load_records(input_path, args.input_format)
    except Exception as exc:  # noqa: BLE001
        print(f"error: failed to read input file: {exc}", file=sys.stderr)
        return 2

    output_rows, stats = convert_records(
        rows,
        author_handle=args.author_handle,
        kol_id=args.kol_id,
        start_date=start_date,
        end_date=end_date,
        include_raw_json=not args.no_raw_json,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "input_count": stats.input_count,
                "output_count": stats.output_count,
                "skipped_missing_id": stats.skipped_missing_id,
                "skipped_missing_time": stats.skipped_missing_time,
                "skipped_missing_text": stats.skipped_missing_text,
                "skipped_missing_handle": stats.skipped_missing_handle,
                "skipped_date_range": stats.skipped_date_range,
                "dedup_skipped": stats.dedup_skipped,
                "output_path": str(output_path),
                "dedup_strategy": "keep_first_by_external_id",
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
