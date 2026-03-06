from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import sys

from fastapi.testclient import TestClient
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db import get_db
from main import app, reset_runtime_counters
from models import Asset, PostExtraction, RawPost
from settings import get_settings
from test_extractor_openai_and_fallback import FakeAsyncSession


def _seed_raw_post(db: FakeAsyncSession, *, raw_post_id: int, content_text: str) -> None:
    db.seed(
        RawPost(
            id=raw_post_id,
            platform="x",
            author_handle="rule_tester",
            external_id=f"post-{raw_post_id}",
            url=f"https://x.com/rule_tester/status/{raw_post_id}",
            content_text=content_text,
            posted_at=datetime(2026, 2, 24, 9, 0, tzinfo=UTC),
            fetched_at=datetime.now(UTC),
            raw_json=None,
        )
    )


def _seed_assets(db: FakeAsyncSession, symbols: list[str]) -> None:
    for idx, symbol in enumerate(symbols, start=1):
        db.seed(
            Asset(
                id=idx,
                symbol=symbol,
                name=symbol,
                market="OTHER",
                created_at=datetime.now(UTC),
            )
        )


@pytest.fixture(autouse=True)
def _clear_runtime():
    get_settings.cache_clear()
    reset_runtime_counters()
    yield
    app.dependency_overrides.clear()
    get_settings.cache_clear()
    reset_runtime_counters()


def test_keep_parsed_asset_views_when_direct_mentions_le_3(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    _seed_assets(fake_db, ["AAPL", "TSLA", "BTC", "MSFT", "NVDA"])
    _seed_raw_post(fake_db, raw_post_id=1, content_text="AAPL TSLA BTC are discussed today.")
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fake_extract(self, raw_post):  # noqa: ANN001
        return {
            "hasview": 1,
            "horizon": "1w",
            "asset_views": [
                {"symbol": "AAPL", "stance": "bull", "horizon": "1w", "confidence": 70, "summary": "看多苹果", "reasoning": "a"},
                {"symbol": "TSLA", "stance": "bull", "horizon": "1w", "confidence": 70, "summary": "看多特斯拉", "reasoning": "b"},
                {"symbol": "BTC", "stance": "bull", "horizon": "1w", "confidence": 70, "summary": "看多比特币", "reasoning": "c"},
                {"symbol": "MSFT", "stance": "bull", "horizon": "1w", "confidence": 70, "summary": "看多微软", "reasoning": "d"},
                {"symbol": "NVDA", "stance": "bull", "horizon": "1w", "confidence": 70, "summary": "看多英伟达", "reasoning": "e"},
            ],
        }

    async def override_get_db():
        yield fake_db

    monkeypatch.setattr("services.extraction.OpenAIExtractor.extract", fake_extract)
    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post("/raw-posts/1/extract")
    assert response.status_code == 201
    body = response.json()
    views = body["extracted_json"]["asset_views"]
    symbols = {item["symbol"] for item in views}
    meta = body["extracted_json"]["meta"]
    assert len(views) == 5
    assert symbols == {"AAPL", "TSLA", "BTC", "MSFT", "NVDA"}
    assert meta["asset_views_capped"] is False
    assert meta["asset_views_original_count"] == 5
    assert meta["asset_views_final_count"] == 5
    assert meta["asset_views_cap_reason"] is None
    assert set(meta["asset_views_kept_symbols"]) == {"AAPL", "TSLA", "BTC", "MSFT", "NVDA"}
    assert "original_count" not in meta
    assert "final_count" not in meta
    assert "cap_reason" not in meta
    assert "kept_symbols" not in meta


def test_keep_parsed_asset_views_when_direct_mentions_gt_3(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    _seed_assets(fake_db, ["AAPL", "TSLA", "BTC", "USD/JPY", "EUR/USD", "AMZN"])
    _seed_raw_post(fake_db, raw_post_id=2, content_text="AAPL TSLA BTC USD/JPY EUR/USD all mentioned explicitly.")
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fake_extract(self, raw_post):  # noqa: ANN001
        return {
            "asset_views": [
                {"symbol": "AAPL", "stance": "bull", "horizon": "1w", "confidence": 75, "summary": "看多苹果", "reasoning": "a"},
                {"symbol": "TSLA", "stance": "bull", "horizon": "1w", "confidence": 75, "summary": "看多特斯拉", "reasoning": "b"},
                {"symbol": "BTC", "stance": "bull", "horizon": "1w", "confidence": 75, "summary": "看多比特币", "reasoning": "c"},
                {"symbol": "USD/JPY", "stance": "bull", "horizon": "1w", "confidence": 75, "summary": "看多美元日元", "reasoning": "d"},
                {"symbol": "EUR/USD", "stance": "bull", "horizon": "1w", "confidence": 75, "summary": "看多欧元美元", "reasoning": "e"},
                {"symbol": "AMZN", "stance": "bull", "horizon": "1w", "confidence": 60, "summary": "看多亚马逊", "reasoning": "x"},
            ],
        }

    async def override_get_db():
        yield fake_db

    monkeypatch.setattr("services.extraction.OpenAIExtractor.extract", fake_extract)
    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post("/raw-posts/2/extract")
    assert response.status_code == 201
    body = response.json()
    symbols = {item["symbol"] for item in body["extracted_json"]["asset_views"]}
    assert symbols == {"AAPL", "TSLA", "BTC", "USD/JPY", "EUR/USD"}
    assert "AMZN" not in symbols
    meta = body["extracted_json"]["meta"]
    assert body["extracted_json"]["meta"]["asset_views_capped"] is False
    assert meta["asset_views_cap_reason"] is None
    assert meta["asset_views_original_count"] == 5
    assert meta["asset_views_final_count"] == 5


def test_keep_parsed_asset_views_when_no_direct_mentions_even_in_macro_post(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = FakeAsyncSession()
    _seed_assets(fake_db, ["SPY", "GLD", "BTC", "ETH", "AAPL", "MSFT", "NVDA"])
    _seed_raw_post(fake_db, raw_post_id=3, content_text="宏观风险偏好下行，加密市场和黄金更受关注。")
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fake_extract(self, raw_post):  # noqa: ANN001
        return {
            "asset_views": [
                {"symbol": "AAPL", "stance": "bear", "horizon": "1w", "confidence": 60, "summary": "看空苹果", "reasoning": "a"},
                {"symbol": "MSFT", "stance": "bear", "horizon": "1w", "confidence": 60, "summary": "看空微软", "reasoning": "b"},
                {"symbol": "NVDA", "stance": "bear", "horizon": "1w", "confidence": 60, "summary": "看空英伟达", "reasoning": "c"},
                {"symbol": "SPY", "stance": "bear", "horizon": "1w", "confidence": 75, "summary": "看空美股", "reasoning": "d"},
                {"symbol": "GLD", "stance": "bull", "horizon": "1w", "confidence": 75, "summary": "看多黄金", "reasoning": "e"},
                {"symbol": "BTC", "stance": "bear", "horizon": "1w", "confidence": 75, "summary": "看空比特币", "reasoning": "f"},
                {"symbol": "ETH", "stance": "bear", "horizon": "1w", "confidence": 75, "summary": "看空以太坊", "reasoning": "g"},
            ],
        }

    async def override_get_db():
        yield fake_db

    monkeypatch.setattr("services.extraction.OpenAIExtractor.extract", fake_extract)
    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post("/raw-posts/3/extract")
    assert response.status_code == 201
    body = response.json()
    symbols = {item["symbol"] for item in body["extracted_json"]["asset_views"]}
    assert symbols == {"SPY", "GLD", "BTC", "ETH"}
    meta = body["extracted_json"]["meta"]
    assert meta["asset_views_capped"] is False
    assert meta["asset_views_original_count"] == 4
    assert meta["asset_views_final_count"] == 4
    assert meta["asset_views_cap_reason"] is None


def test_keep_parsed_asset_views_when_no_direct_mentions_and_no_macro(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db, raw_post_id=31, content_text="这个供应链观点后续继续跟踪。")
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fake_extract(self, raw_post):  # noqa: ANN001
        return {
            "asset_views": [
                {"symbol": "HYNIX", "stance": "bull", "horizon": "1w", "confidence": 73, "summary": "HBM需求上行", "reasoning": "cycle up"},
            ],
        }

    async def override_get_db():
        yield fake_db

    monkeypatch.setattr("services.extraction.OpenAIExtractor.extract", fake_extract)
    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post("/raw-posts/31/extract")
    assert response.status_code == 201
    body = response.json()
    views = body["extracted_json"]["asset_views"]
    assert len(views) == 1
    assert views[0]["symbol"] == "HYNIX"
    assert body["extracted_json"]["meta"]["asset_views_cap_reason"] is None


def test_precious_crypto_fx_symbols_can_persist_and_display(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    _seed_assets(fake_db, ["XAUUSD", "BTC", "USD/JPY"])
    _seed_raw_post(fake_db, raw_post_id=4, content_text="XAUUSD BTC USD/JPY 波动。")
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fake_extract(self, raw_post):  # noqa: ANN001
        return {
            "asset_views": [
                {"symbol": "XAUUSD", "stance": "bull", "horizon": "1w", "confidence": 77, "summary": "看多黄金", "reasoning": "gold"},
                {"symbol": "BTC", "stance": "bull", "horizon": "1w", "confidence": 71, "summary": "看多比特币", "reasoning": "btc"},
                {"symbol": "USD/JPY", "stance": "bull", "horizon": "1w", "confidence": 70, "summary": "看多美元日元", "reasoning": "fx"},
            ],
        }

    async def override_get_db():
        yield fake_db

    monkeypatch.setattr("services.extraction.OpenAIExtractor.extract", fake_extract)
    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post("/raw-posts/4/extract")
    assert response.status_code == 201
    extraction_id = response.json()["id"]
    assets_response = client.get("/assets")
    assert assets_response.status_code == 200
    symbols = {item["symbol"] for item in assets_response.json()}
    assert {"XAUUSD", "BTC", "USD/JPY"} <= symbols
    detail = client.get(f"/extractions/{extraction_id}")
    assert detail.status_code == 200
    detail_symbols = {item["symbol"] for item in detail.json()["extracted_json"]["asset_views"]}
    assert {"XAUUSD", "BTC", "USD/JPY"} <= detail_symbols


def test_invalid_horizon_is_dropped_instead_of_coerced(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    _seed_assets(fake_db, ["BTC"])
    _seed_raw_post(fake_db, raw_post_id=5, content_text="BTC 受事件影响。")
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fake_extract(self, raw_post):  # noqa: ANN001
        return {
            "horizon": "tonight",
            "asset_views": [
                {"symbol": "BTC", "stance": "bull", "horizon": "今晚", "confidence": 70, "summary": "看多比特币", "reasoning": "r"},
            ],
        }

    async def override_get_db():
        yield fake_db

    monkeypatch.setattr("services.extraction.OpenAIExtractor.extract", fake_extract)
    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    weekly = client.post("/raw-posts/5/extract")
    assert weekly.status_code == 201
    weekly_meta = weekly.json()["extracted_json"]["meta"]
    assert weekly.json()["extracted_json"]["asset_views"] == []
    assert "horizon_coerced" not in weekly_meta


def test_large_text_truncation_and_raw_output_truncation_without_breaking_extracted_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = FakeAsyncSession()
    _seed_assets(fake_db, ["BTC"])
    _seed_raw_post(fake_db, raw_post_id=7, content_text=("x" * 1200) + " BTC")
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("MODEL_CONTEXT_WINDOW_TOKENS", "2200")
    monkeypatch.setenv("MODEL_PROMPT_RESERVE_TOKENS", "1000")
    monkeypatch.setenv("OPENAI_MAX_OUTPUT_TOKENS", "1000")
    monkeypatch.setenv("MODEL_CHARS_PER_TOKEN", "1")
    monkeypatch.setenv("RAW_MODEL_OUTPUT_MAX_CHARS", "300")

    large_summary = "y" * 5000

    def fake_extract(self, raw_post):  # noqa: ANN001
        return {
            "horizon": "1w",
            "summary": large_summary,
            "asset_views": [
                {"symbol": "BTC", "stance": "bull", "horizon": "1w", "confidence": 80, "summary": "看多比特币", "reasoning": "r"},
            ],
        }

    async def override_get_db():
        yield fake_db

    monkeypatch.setattr("services.extraction.OpenAIExtractor.extract", fake_extract)
    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post("/raw-posts/7/extract")
    assert response.status_code == 201
    body = response.json()
    meta = body["extracted_json"]["meta"]
    assert meta["truncated"] is True
    assert meta["raw_truncated"] is True
    assert meta["raw_saved_len"] == 300
    assert len(body["raw_model_output"]) == 300
    assert body["extracted_json"]["asset_views"][0]["summary"] == "看多比特币"


def test_extract_batch_counters_include_capped_horizon_coerced_and_raw_truncated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = FakeAsyncSession()
    _seed_assets(fake_db, ["AAPL", "TSLA", "BTC", "MSFT", "NVDA"])
    _seed_raw_post(fake_db, raw_post_id=8, content_text="AAPL TSLA BTC are discussed today.")
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("RAW_MODEL_OUTPUT_MAX_CHARS", "120")

    def fake_extract(self, raw_post):  # noqa: ANN001
        return {
            "horizon": "invalid_horizon",
            "asset_views": [
                {"symbol": "AAPL", "stance": "bull", "horizon": "unknown", "confidence": 70, "summary": "看多苹果", "reasoning": "a"},
                {"symbol": "TSLA", "stance": "bull", "horizon": "unknown", "confidence": 70, "summary": "看多特斯拉", "reasoning": "b"},
                {"symbol": "BTC", "stance": "bull", "horizon": "unknown", "confidence": 70, "summary": "看多比特币", "reasoning": "c"},
                {"symbol": "MSFT", "stance": "bull", "horizon": "unknown", "confidence": 70, "summary": "看多微软", "reasoning": "d"},
            ],
        }

    async def override_get_db():
        yield fake_db

    monkeypatch.setattr("services.extraction.OpenAIExtractor.extract", fake_extract)
    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post("/raw-posts/extract-batch", json={"raw_post_ids": [8], "mode": "force"})
    assert response.status_code == 200
    body = response.json()
    assert body["success_count"] == 1
    assert body["capped_count"] in {0, 1}
    assert body["horizon_coerced_count"] == 0
    assert body["raw_truncated_count"] == 1
    extraction = next(iter(fake_db._data[PostExtraction].values()))
    meta = extraction.extracted_json["meta"]
    assert "original_count" not in meta
    assert "final_count" not in meta
    assert "cap_reason" not in meta
    assert "kept_symbols" not in meta
    assert meta["asset_views_original_count"] == 0
    assert meta["asset_views_final_count"] == 0


def test_get_extraction_detail_returns_single_top_level_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    _seed_assets(fake_db, ["BTC"])
    _seed_raw_post(fake_db, raw_post_id=9, content_text="BTC")
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fake_extract(self, raw_post):  # noqa: ANN001
        return {
            "asset_views": [
                {"symbol": "BTC", "stance": "bull", "horizon": "1w", "confidence": 70, "summary": "看多比特币", "reasoning": "btc"},
            ],
        }

    async def override_get_db():
        yield fake_db

    monkeypatch.setattr("services.extraction.OpenAIExtractor.extract", fake_extract)
    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    created = client.post("/raw-posts/9/extract")
    assert created.status_code == 201
    extraction_id = created.json()["id"]

    detail = client.get(f"/extractions/{extraction_id}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["id"] == extraction_id
    assert "extraction" not in body
