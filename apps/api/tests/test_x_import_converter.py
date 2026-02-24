from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.x_import_converter import convert_records


def test_convert_records_filters_date_extracts_id_and_dedups() -> None:
    rows = [
        {
            "url": "https://x.com/alice/status/1001",
            "created_at": "2026-02-20T10:00:00Z",
            "text": "first",
            "screen_name": "alice",
        },
        {
            "tweet_id": "1001",
            "created_at": "2026-02-20T11:00:00Z",
            "text": "duplicate",
            "screen_name": "alice",
        },
        {
            "tweet_id": "1002",
            "created_at": "2026-02-25T11:00:00Z",
            "text": "out of range",
            "screen_name": "alice",
        },
    ]

    converted, stats = convert_records(
        rows,
        author_handle=None,
        kol_id=None,
        start_date=date(2026, 2, 19),
        end_date=date(2026, 2, 23),
    )

    assert len(converted) == 1
    assert converted[0]["external_id"] == "1001"
    assert converted[0]["url"] == "https://x.com/alice/status/1001"
    assert converted[0]["author_handle"] == "alice"
    assert converted[0]["resolved_author_handle"] == "alice"
    assert converted[0]["posted_at"] == "2026-02-20T10:00:00Z"
    assert stats.dedup_skipped == 1
    assert stats.skipped_date_range == 1
    assert stats.failed_count == 2
    assert len(stats.errors or []) == 2


def test_convert_records_author_override_and_kol_id() -> None:
    rows = [
        {
            "tweet_id": "2001",
            "created_at": "2026-02-20 10:00:00",
            "full_text": "hello",
            "screen_name": "raw_name",
        }
    ]

    converted, stats = convert_records(
        rows,
        author_handle="@qinbafrank",
        kol_id=123,
        start_date=None,
        end_date=None,
    )

    assert stats.output_count == 1
    assert converted[0]["author_handle"] == "qinbafrank"
    assert converted[0]["resolved_author_handle"] == "qinbafrank"
    assert converted[0]["kol_id"] == 123
    assert converted[0]["url"] == "https://x.com/qinbafrank/status/2001"
    assert converted[0]["posted_at"].endswith("Z")
    assert stats.failed_count == 0
