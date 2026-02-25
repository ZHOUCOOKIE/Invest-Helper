from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import sys

from fastapi.testclient import TestClient
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db import get_db
from main import _extract_directly_mentioned_symbols, app, reset_runtime_counters
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
                market="AUTO",
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


def test_cap_to_three_when_direct_mentions_le_3(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    _seed_assets(fake_db, ["AAPL", "TSLA", "BTC", "MSFT", "NVDA"])
    _seed_raw_post(fake_db, raw_post_id=1, content_text="AAPL TSLA BTC are discussed today.")
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fake_extract(self, raw_post):  # noqa: ANN001
        return {
            "assets": [{"symbol": "AAPL"}, {"symbol": "TSLA"}, {"symbol": "BTC"}, {"symbol": "MSFT"}, {"symbol": "NVDA"}],
            "horizon": "1w",
            "asset_views": [
                {"symbol": "AAPL", "stance": "bull", "horizon": "1w", "confidence": 70, "summary": "a", "reasoning": "a"},
                {"symbol": "TSLA", "stance": "bull", "horizon": "1w", "confidence": 70, "summary": "b", "reasoning": "b"},
                {"symbol": "BTC", "stance": "bull", "horizon": "1w", "confidence": 70, "summary": "c", "reasoning": "c"},
                {"symbol": "MSFT", "stance": "bull", "horizon": "1w", "confidence": 70, "summary": "d", "reasoning": "d"},
                {"symbol": "NVDA", "stance": "bull", "horizon": "1w", "confidence": 70, "summary": "e", "reasoning": "e"},
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
    assert len(views) == 3
    assert symbols == {"AAPL", "TSLA", "BTC"}
    assert meta["asset_views_capped"] is True
    assert meta["asset_views_original_count"] == 5
    assert meta["asset_views_final_count"] == 3
    assert meta["asset_views_cap_reason"] == "direct_mentions_le_3"
    assert set(meta["asset_views_kept_symbols"]) == {"AAPL", "TSLA", "BTC"}
    assert set(meta["direct_mentioned_symbols"]) == {"AAPL", "TSLA", "BTC"}
    assert "original_count" not in meta
    assert "final_count" not in meta
    assert "cap_reason" not in meta
    assert "kept_symbols" not in meta


def test_keep_all_direct_mentions_when_gt_3_without_deriving_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    _seed_assets(fake_db, ["AAPL", "TSLA", "BTC", "USD/JPY", "EUR/USD", "AMZN"])
    _seed_raw_post(fake_db, raw_post_id=2, content_text="AAPL TSLA BTC USD/JPY EUR/USD all mentioned explicitly.")
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fake_extract(self, raw_post):  # noqa: ANN001
        return {
            "asset_views": [
                {"symbol": "AAPL", "stance": "bull", "horizon": "1w", "confidence": 60, "summary": "a", "reasoning": "a"},
                {"symbol": "TSLA", "stance": "bull", "horizon": "1w", "confidence": 60, "summary": "b", "reasoning": "b"},
                {"symbol": "BTC", "stance": "bull", "horizon": "1w", "confidence": 60, "summary": "c", "reasoning": "c"},
                {"symbol": "USD/JPY", "stance": "bull", "horizon": "1w", "confidence": 60, "summary": "d", "reasoning": "d"},
                {"symbol": "EUR/USD", "stance": "bull", "horizon": "1w", "confidence": 60, "summary": "e", "reasoning": "e"},
                {"symbol": "AMZN", "stance": "bull", "horizon": "1w", "confidence": 60, "summary": "x", "reasoning": "x"},
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
    assert body["extracted_json"]["meta"]["asset_views_capped"] is True
    assert meta["asset_views_cap_reason"] == "keep_only_direct_mentions_when_gt_3"
    assert set(meta["direct_mentioned_symbols"]) == {"AAPL", "TSLA", "BTC", "USD/JPY", "EUR/USD"}
    assert set(symbols).issubset(set(meta["direct_mentioned_symbols"]))


def test_macro_post_only_keeps_representative_assets(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    _seed_assets(fake_db, ["SPY", "GLD", "BTC", "ETH", "AAPL", "MSFT", "NVDA"])
    _seed_raw_post(fake_db, raw_post_id=3, content_text="宏观风险偏好下行，加密市场和黄金更受关注。")
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fake_extract(self, raw_post):  # noqa: ANN001
        return {
            "asset_views": [
                {"symbol": "AAPL", "stance": "bear", "horizon": "1w", "confidence": 60, "summary": "a", "reasoning": "a"},
                {"symbol": "MSFT", "stance": "bear", "horizon": "1w", "confidence": 60, "summary": "b", "reasoning": "b"},
                {"symbol": "NVDA", "stance": "bear", "horizon": "1w", "confidence": 60, "summary": "c", "reasoning": "c"},
                {"symbol": "SPY", "stance": "bear", "horizon": "1w", "confidence": 60, "summary": "d", "reasoning": "d"},
                {"symbol": "GLD", "stance": "bull", "horizon": "1w", "confidence": 60, "summary": "e", "reasoning": "e"},
                {"symbol": "BTC", "stance": "bear", "horizon": "1w", "confidence": 60, "summary": "f", "reasoning": "f"},
                {"symbol": "ETH", "stance": "bear", "horizon": "1w", "confidence": 60, "summary": "g", "reasoning": "g"},
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
    assert "AAPL" not in symbols and "MSFT" not in symbols and "NVDA" not in symbols
    assert symbols & {"SPY", "GLD", "BTC", "ETH"}


def test_keep_parsed_asset_views_when_no_direct_mentions_and_no_macro(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db, raw_post_id=31, content_text="这个供应链观点后续继续跟踪。")
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fake_extract(self, raw_post):  # noqa: ANN001
        return {
            "asset_views": [
                {"symbol": "HYNIX", "stance": "bull", "horizon": "1w", "confidence": 73, "summary": "hbm up", "reasoning": "cycle up"},
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
                {"symbol": "XAUUSD", "stance": "bull", "horizon": "1w", "confidence": 77, "summary": "gold", "reasoning": "gold"},
                {"symbol": "BTC", "stance": "bull", "horizon": "1w", "confidence": 71, "summary": "btc", "reasoning": "btc"},
                {"symbol": "USD/JPY", "stance": "bull", "horizon": "1w", "confidence": 69, "summary": "fx", "reasoning": "fx"},
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


def test_horizon_invalid_value_is_coerced_to_1w_with_meta(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    _seed_assets(fake_db, ["BTC"])
    _seed_raw_post(fake_db, raw_post_id=5, content_text="BTC 受事件影响。")
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fake_extract(self, raw_post):  # noqa: ANN001
        return {
            "horizon": "tonight",
            "asset_views": [
                {"symbol": "BTC", "stance": "bull", "horizon": "今晚", "confidence": 70, "summary": "s", "reasoning": "r"},
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
    assert weekly.json()["extracted_json"]["horizon"] == "1w"
    assert weekly_meta["horizon_coerced"] is True
    assert weekly_meta["horizon_original"] == "tonight"
    assert weekly_meta["horizon_final"] == "1w"
    assert weekly_meta["horizon_coerce_reason"] == "invalid_horizon_enum"


def test_long_text_truncation_and_raw_output_truncation_without_breaking_extracted_json(
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

    long_summary = "y" * 5000

    def fake_extract(self, raw_post):  # noqa: ANN001
        return {
            "horizon": "1w",
            "summary": long_summary,
            "asset_views": [
                {"symbol": "BTC", "stance": "bull", "horizon": "1w", "confidence": 80, "summary": long_summary, "reasoning": "r"},
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
    assert len(body["extracted_json"]["summary"]) == len(long_summary)


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
        long_summary = "z" * 2000
        return {
            "horizon": "invalid_horizon",
            "summary": long_summary,
            "asset_views": [
                {"symbol": "AAPL", "stance": "bull", "horizon": "unknown", "confidence": 70, "summary": long_summary, "reasoning": "a"},
                {"symbol": "TSLA", "stance": "bull", "horizon": "unknown", "confidence": 70, "summary": long_summary, "reasoning": "b"},
                {"symbol": "BTC", "stance": "bull", "horizon": "unknown", "confidence": 70, "summary": long_summary, "reasoning": "c"},
                {"symbol": "MSFT", "stance": "bull", "horizon": "unknown", "confidence": 70, "summary": long_summary, "reasoning": "d"},
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
    assert body["capped_count"] == 1
    assert body["horizon_coerced_count"] == 1
    assert body["raw_truncated_count"] == 1
    extraction = next(iter(fake_db._data[PostExtraction].values()))
    meta = extraction.extracted_json["meta"]
    assert "original_count" not in meta
    assert "final_count" not in meta
    assert "cap_reason" not in meta
    assert "kept_symbols" not in meta
    assert meta["asset_views_original_count"] == 4
    assert meta["asset_views_final_count"] == 3


def test_direct_mentions_detection_has_no_cross_call_leakage() -> None:
    alias_to_symbol = {"特斯拉": "TSLA", "比特币": "BTC"}
    known_symbols = {"AAPL", "TSLA", "BTC"}

    first = _extract_directly_mentioned_symbols(
        content_text="今天看 AAPL 和 特斯拉",
        alias_to_symbol=alias_to_symbol,
        known_symbols=known_symbols,
    )
    second = _extract_directly_mentioned_symbols(
        content_text="比特币短线波动",
        alias_to_symbol=alias_to_symbol,
        known_symbols=known_symbols,
    )

    assert set(first) == {"AAPL", "TSLA"}
    assert set(second) == {"BTC"}
    assert "AAPL" not in second
    assert "TSLA" not in second


def test_get_extraction_detail_returns_single_top_level_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    _seed_assets(fake_db, ["BTC"])
    _seed_raw_post(fake_db, raw_post_id=9, content_text="BTC")
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fake_extract(self, raw_post):  # noqa: ANN001
        return {
            "asset_views": [
                {"symbol": "BTC", "stance": "bull", "horizon": "1w", "confidence": 70, "summary": "btc", "reasoning": "btc"},
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
