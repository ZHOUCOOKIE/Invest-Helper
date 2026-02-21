from datetime import UTC, datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from enums import Horizon, Stance
from models import RawPost
from schemas import KolViewCreate
from services.extraction import DummyExtractor


def test_kol_view_create_allows_optional_fields():
    payload = KolViewCreate(
        kol_id=1,
        asset_id=1,
        stance=Stance.bull,
        horizon=Horizon.one_week,
        confidence=66,
    )

    assert payload.summary is None
    assert payload.source_url is None
    assert payload.as_of is None


def test_dummy_extractor_returns_pending_payload_shape():
    raw_post = RawPost(
        id=123,
        platform="x",
        author_handle="alice",
        external_id="post-001",
        url="https://x.com/alice/status/post-001",
        content_text="BTC looks range-bound this week.",
        posted_at=datetime(2026, 2, 21, 12, 0, tzinfo=UTC),
    )

    extractor = DummyExtractor()
    result = extractor.extract(raw_post)

    assert extractor.model_name == "dummy-v2"
    assert extractor.extractor_name == "dummy"
    assert result["assets"] == []
    assert result["stance"] == "neutral"
    assert result["horizon"] == "1w"
    assert result["confidence"] == 50
    assert result["source_url"] == "https://x.com/alice/status/post-001"
