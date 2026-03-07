from __future__ import annotations

import json
from pathlib import Path
import sys

from fastapi.testclient import TestClient
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import app
import services.portfolio_advice as portfolio_advice_service
from settings import get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache() -> None:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_portfolio_advice_without_api_key_returns_rules_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    client = TestClient(app)
    response = client.post(
        "/portfolio/advice",
        json={
            "holdings": [
                {
                    "asset_id": 1,
                    "symbol": "BTC",
                    "holding_reason_text": "趋势仍在",
                    "sell_timing_text": "跌破关键位分批卖出",
                    "support_citations": [{"source_url": "https://x.com/a/1", "summary": "突破延续"}],
                    "risk_citations": [{"source_url": "https://x.com/b/2", "summary": "链上活跃下降"}],
                }
            ]
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "skipped_no_api_key"
    assert body["model"]
    assert body["advice_summary"]
    assert len(body["asset_advice"]) == 1
    assert body["asset_advice"][0]["symbol"] == "BTC"


def test_portfolio_advice_with_ai_response_returns_structured_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class FakeResponse:
        status_code = 200

        def json(self) -> dict:
            content = json.dumps(
                {
                    "advice_summary": "组合风险中性偏高，建议控制仓位并跟踪兑现节奏。",
                    "asset_advice": [
                        {
                            "asset_id": 7,
                            "symbol": "TSLA",
                            "score": 64,
                            "stance": "持有观察",
                            "suggestion": "短期不追高，回撤分批。",
                            "evaluation": "证据分歧较大，先控仓。",
                            "key_risks": ["交付不及预期"],
                            "key_triggers": ["利润率修复"],
                        }
                    ],
                },
                ensure_ascii=False,
            )
            return {"choices": [{"message": {"content": content}}]}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            pass

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
            return None

        async def post(self, *args, **kwargs) -> FakeResponse:  # noqa: ANN002, ANN003
            return FakeResponse()

    monkeypatch.setattr(portfolio_advice_service.httpx, "AsyncClient", FakeAsyncClient)

    client = TestClient(app)
    response = client.post(
        "/portfolio/advice",
        json={
            "holdings": [
                {
                    "asset_id": 7,
                    "symbol": "TSLA",
                    "holding_reason_text": "观察盈利修复",
                    "sell_timing_text": "跌破仓位纪律线减仓",
                    "support_citations": [{"source_url": "https://x.com/a/1", "summary": "利润率改善"}],
                    "risk_citations": [{"source_url": "https://x.com/b/2", "summary": "需求走弱"}],
                }
            ]
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["advice_summary"].startswith("组合风险中性偏高")
    assert len(body["asset_advice"]) == 1
    assert body["asset_advice"][0]["asset_id"] == 7
    assert body["asset_advice"][0]["score"] == 64
