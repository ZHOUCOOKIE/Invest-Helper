from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
import hashlib
import json
from pathlib import Path
import sys

from fastapi.testclient import TestClient
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db import get_db
from enums import ExtractionStatus, ReviewStatus
from main import app, postprocess_auto_review, reset_runtime_counters
from models import Asset, AssetAlias, Kol, KolView, PostExtraction, RawPost
from services.extraction import OpenAIExtractor, OpenAIRequestError, parse_text_json_object, normalize_extracted_json
from services.prompts import build_extract_prompt
from services.prompts.extract_v1 import PromptBundle
from settings import get_settings


class FakeResult:
    def __init__(self, items: list[object]):
        self._items = items

    def scalars(self) -> "FakeResult":
        return self

    def all(self) -> list[object]:
        return self._items

    def scalar_one_or_none(self) -> object | None:
        return self._items[0] if self._items else None


class FakeAsyncSession:
    def __init__(self) -> None:
        self._data: dict[type[object], dict[int, object]] = {
            RawPost: {},
            PostExtraction: {},
            Asset: {},
            AssetAlias: {},
            Kol: {},
            KolView: {},
        }
        self._new: list[object] = []

    def seed(self, obj: object) -> None:
        self._data[type(obj)][getattr(obj, "id")] = obj

    async def get(self, model: type[object], obj_id: int) -> object | None:
        return self._data.get(model, {}).get(obj_id)

    def add(self, obj: object) -> None:
        self._new.append(obj)

    async def flush(self) -> None:
        self._persist_new()

    async def commit(self) -> None:
        self._persist_new()

    async def refresh(self, obj: object) -> None:
        return None

    async def rollback(self) -> None:
        return None

    async def delete(self, obj: object) -> None:
        model = type(obj)
        obj_id = getattr(obj, "id", None)
        if obj_id is None:
            return
        self._data.get(model, {}).pop(obj_id, None)

    async def execute(self, query) -> FakeResult:  # noqa: ANN001
        sql = str(query).lower()
        if "from asset_aliases join assets" in sql:
            alias_items = list(self._data[AssetAlias].values())
            for criterion in getattr(query, "_where_criteria", ()):
                key = criterion.left.key
                value = criterion.right.value
                alias_items = [item for item in alias_items if getattr(item, key) == value]
            alias_items.sort(key=lambda item: item.id)
            assets = self._data[Asset]
            if "asset_aliases.alias, assets.symbol" in sql:
                return FakeResult([(item.alias, assets[item.asset_id].symbol) for item in alias_items])
            return FakeResult([(item, assets[item.asset_id].symbol) for item in alias_items])

        entity = query.column_descriptions[0]["entity"]
        items = list(self._data.get(entity, {}).values())

        for criterion in getattr(query, "_where_criteria", ()):
            key = criterion.left.key
            value = criterion.right.value
            items = [item for item in items if getattr(item, key) == value]

        if entity is PostExtraction:
            raw_posts = self._data[RawPost]
            for item in items:
                item.raw_post = raw_posts.get(item.raw_post_id)

        return FakeResult(items)

    def _persist_new(self) -> None:
        now = datetime.now(UTC)
        for obj in self._new:
            model = type(obj)
            bucket = self._data[model]
            if getattr(obj, "id", None) is None:
                next_id = max(bucket.keys(), default=0) + 1
                setattr(obj, "id", next_id)
            if hasattr(obj, "created_at") and getattr(obj, "created_at") is None:
                setattr(obj, "created_at", now)
            if hasattr(obj, "fetched_at") and getattr(obj, "fetched_at") is None:
                setattr(obj, "fetched_at", now)
            bucket[getattr(obj, "id")] = obj
        self._new.clear()


def _seed_raw_post(db: FakeAsyncSession) -> None:
    db.seed(
        RawPost(
            id=1,
            platform="x",
            author_handle="alice",
            external_id="post-1",
            url="https://x.com/alice/status/post-1",
            content_text="BTC looks constructive if 100k holds this week.",
            posted_at=datetime(2026, 2, 21, 9, 0, tzinfo=UTC),
            fetched_at=datetime.now(UTC),
            raw_json=None,
        )
    )


def _seed_chinese_raw_post(db: FakeAsyncSession) -> None:
    db.seed(
        RawPost(
            id=2,
            platform="x",
            author_handle="cn_view",
            external_id="post-cn-1",
            url="https://x.com/cn_view/status/post-cn-1",
            content_text=(
                "金融市场目前，避险与通胀是今晚的主题，地缘风险隐蔽风险下，黄金、原油价格走高，"
                "今晚的GDP数据与PCE影响之下，通胀预期+久期风险推高了长债收益率。\n\n"
                "目前流动性主要流向避险资产以及债市，短期虽然美股依旧上涨，但是VIX指数仅仅跌破20附近，"
                "并未有继续下跌的迹象，且风险偏好走弱，美股目前只能算是稳定不能说是乐观。\n\n"
                "当前阶段，市场流动性并不青睐新兴资产，具体说就是不青睐加密市场，"
                "所以今晚只能算是一个美股稳定保持谨慎乐观的阶段。"
            ),
            posted_at=datetime(2026, 2, 21, 10, 0, tzinfo=UTC),
            fetched_at=datetime.now(UTC),
            raw_json=None,
        )
    )


@pytest.fixture(autouse=True)
def clear_settings_cache():
    get_settings.cache_clear()
    reset_runtime_counters()
    yield
    get_settings.cache_clear()
    reset_runtime_counters()


def test_extract_endpoint_persists_mocked_openai_json_and_get_by_id(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)

    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    def fake_extract(self, raw_post: RawPost):  # noqa: ANN001
            return {
                "assets": [{"symbol": "BTC", "name": "Bitcoin", "market": "CRYPTO"}],
                "stance": "bull",
                "horizon": "1w",
                "confidence": 78,
                "summary": "支撑位有效时，短线仍有上行空间。",
                "source_url": raw_post.url,
                "as_of": "2026-02-21",
                "event_tags": ["support"],
            }

    monkeypatch.setattr("services.extraction.OpenAIExtractor.extract", fake_extract)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    create_resp = client.post("/raw-posts/1/extract")
    assert create_resp.status_code == 201
    created = create_resp.json()
    assert created["status"] == ExtractionStatus.approved.value
    assert created["reviewed_by"] == "auto"
    assert created["auto_applied_count"] == 1
    assert created["extractor_name"] == "openai_structured"
    assert created["extracted_json"]["assets"][0]["symbol"] == "BTC"

    read_resp = client.get(f"/extractions/{created['id']}")
    app.dependency_overrides.clear()

    assert read_resp.status_code == 200
    read_body = read_resp.json()
    assert read_body["id"] == created["id"]
    assert read_body["extracted_json"]["stance"] == "bull"
    assert read_body["raw_post"]["id"] == 1


def test_prompt_contract_includes_asset_view_self_check_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    monkeypatch.setenv("EXTRACTOR_MODE", "dummy")

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post("/raw-posts/1/extract")
    app.dependency_overrides.clear()

    assert response.status_code == 201
    prompt_text = response.json()["prompt_text"]
    assert "Checklist (execute in order)" in prompt_text
    assert "asset_views self-check (must pass)" in prompt_text
    assert "If confidence < 70: delete that item and do not output it." in prompt_text
    assert "asset_views must contain no item with confidence < 70." in prompt_text


def test_prompt_contract_includes_symbol_scope_and_noneany_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    monkeypatch.setenv("EXTRACTOR_MODE", "dummy")

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post("/raw-posts/1/extract")
    app.dependency_overrides.clear()

    assert response.status_code == 201
    prompt_text = response.json()["prompt_text"]
    assert "Do NOT use macro variables or theme words as symbol" in prompt_text
    assert "A-share/HK stocks: symbol must be Chinese stock name/short name." in prompt_text
    assert "US stocks/crypto/ETF/index: symbol can be ticker or English name" in prompt_text
    assert "assets must be exactly [{\"symbol\":\"NoneAny\",\"name\":null,\"market\":\"OTHER\"}]" in prompt_text


def test_prompt_forbids_top_level_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

    captured_payloads: list[dict] = []

    class _FakeResponse:
        status_code = 200

        def json(self):  # noqa: ANN001
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "content_kind": "asset",
                                    "as_of": "2026-02-21",
                                    "source_url": "https://x.com/alice/status/post-1",
                                    "assets": [{"symbol": "BTC", "name": "Bitcoin", "market": "CRYPTO"}],
                                    "asset_views": [
                                        {
                                            "symbol": "BTC",
                                            "name": "Bitcoin",
                                            "market": "CRYPTO",
                                            "stance": "bull",
                                            "horizon": "1w",
                                            "confidence": 78,
                                            "summary": "短线偏多。",
                                        }
                                    ],
                                    "library_entry": None,
                                },
                                ensure_ascii=False,
                            )
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 11, "completion_tokens": 180},
            }

    class _FakeClient:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            pass

        def __enter__(self):  # noqa: ANN001
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
            return False

        def post(self, url: str, headers: dict, json: dict):  # noqa: A002
            captured_payloads.append(json)
            return _FakeResponse()

    monkeypatch.setattr("services.extraction.httpx.Client", _FakeClient)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post("/raw-posts/1/extract")
    app.dependency_overrides.clear()

    assert response.status_code == 201
    assert len(captured_payloads) == 1
    system_prompt = captured_payloads[0]["messages"][0]["content"]
    assert "Top-level fields must be exactly: as_of, source_url, content_kind, assets, asset_views, library_entry." in system_prompt
    assert "Do not output asset_views[*].drivers or asset_views[*].reasoning." in system_prompt
    assert "stance, horizon, confidence, summary, reasoning, event_tags" not in system_prompt


def test_summary_non_zh_retries_once_and_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_CALL_BUDGET", "2")
    monkeypatch.setenv("AUTO_APPROVE_ENABLED", "false")

    call_count = {"n": 0}

    def fake_extract(self, raw_post: RawPost):  # noqa: ANN001
        call_count["n"] += 1
        if call_count["n"] == 1:
            return {
                "assets": [{"symbol": "BTC", "name": "Bitcoin", "market": "CRYPTO"}],
                "reasoning": "Risk appetite improves and BTC can continue higher this week.",
                "stance": "bull",
                "horizon": "1w",
                "confidence": 78,
                "summary": "This summary is fully English and should trigger a language retry.",
                "source_url": raw_post.url,
                "as_of": "2026-02-21",
                "asset_views": [
                    {
                        "symbol": "BTC",
                        "stance": "bull",
                        "horizon": "1w",
                        "confidence": 78,
                        "reasoning": "Momentum remains constructive.",
                        "summary": "btc up",
                    }
                ],
            }
        return {
            "assets": [{"symbol": "BTC", "name": "Bitcoin", "market": "CRYPTO"}],
            "reasoning": "风险偏好回升，BTC 本周延续上行概率更高。",
            "stance": "bull",
            "horizon": "1w",
            "confidence": 79,
            "summary": "中文总结：风险偏好回升，短线仍偏多。",
            "source_url": raw_post.url,
            "as_of": "2026-02-21",
            "asset_views": [
                {
                    "symbol": "BTC",
                    "stance": "bull",
                    "horizon": "1w",
                    "confidence": 79,
                    "reasoning": "资金回流，结构偏强。",
                    "summary": "btc up",
                }
            ],
        }

    monkeypatch.setattr("services.extraction.OpenAIExtractor.extract", fake_extract)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post("/raw-posts/1/extract")
    app.dependency_overrides.clear()

    assert response.status_code == 201
    body = response.json()
    meta = body["extracted_json"]["meta"]
    assert call_count["n"] == 2
    assert body["last_error"] is None
    assert meta["summary_language"] == "zh"
    assert meta["summary_language_violation"] is False
    assert meta["summary_language_retry_used"] is True


def test_summary_non_zh_twice_marks_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_CALL_BUDGET", "2")
    monkeypatch.setenv("AUTO_APPROVE_ENABLED", "false")

    call_count = {"n": 0}

    def fake_extract(self, raw_post: RawPost):  # noqa: ANN001
        call_count["n"] += 1
        return {
            "assets": [{"symbol": "BTC", "name": "Bitcoin", "market": "CRYPTO"}],
            "reasoning": "Market momentum remains positive and upside could continue near term.",
            "stance": "bull",
            "horizon": "1w",
            "confidence": 76,
            "summary": "This summary remains English and should still violate the language rule.",
            "source_url": raw_post.url,
            "as_of": "2026-02-21",
            "asset_views": [
                {
                    "symbol": "BTC",
                    "stance": "bull",
                    "horizon": "1w",
                    "confidence": 76,
                    "reasoning": "Follow through buying pressure remains.",
                    "summary": "btc up",
                }
            ],
        }

    monkeypatch.setattr("services.extraction.OpenAIExtractor.extract", fake_extract)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post("/raw-posts/1/extract")
    app.dependency_overrides.clear()

    assert response.status_code == 201
    body = response.json()
    meta = body["extracted_json"]["meta"]
    assert call_count["n"] == 2
    assert "summary_language_violation_after_retry" in (body["last_error"] or "")
    assert meta["summary_language"] == "non_zh"
    assert meta["summary_language_violation"] is True
    assert meta["summary_language_retry_used"] is True


def test_extract_endpoint_filters_out_empty_symbol_asset_view(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    def fake_extract(self, raw_post: RawPost):  # noqa: ANN001
        return {
            "assets": [{"symbol": "BTC", "name": "Bitcoin", "market": "CRYPTO"}],
            "stance": "bull",
            "horizon": "1w",
            "confidence": 78,
            "summary": "symbol filter",
            "source_url": raw_post.url,
            "as_of": "2026-02-21",
            "asset_views": [
                {"symbol": "", "stance": "bull", "horizon": "1w", "confidence": 70, "summary": "drop me"},
                {"symbol": "BTC", "stance": "bull", "horizon": "1w", "confidence": 78, "summary": "keep me"},
            ],
        }

    monkeypatch.setattr("services.extraction.OpenAIExtractor.extract", fake_extract)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post("/raw-posts/1/extract")
    app.dependency_overrides.clear()

    assert response.status_code == 201
    body = response.json()
    views = body["extracted_json"]["asset_views"]
    assert len(views) == 1
    assert views[0]["symbol"] == "BTC"


def test_auto_mode_without_api_key_falls_back_to_dummy_extractor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    monkeypatch.setenv("EXTRACTOR_MODE", "auto")
    monkeypatch.setenv("OPENAI_API_KEY", "")

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    response = client.post("/raw-posts/1/extract")
    app.dependency_overrides.clear()

    assert response.status_code == 201
    body = response.json()
    assert body["extractor_name"] == "dummy"
    assert body["model_name"] == "dummy-v2"
    assert body["extracted_json"]["horizon"] == "1w"


def test_reextract_rate_limit_returns_429(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    monkeypatch.setenv("EXTRACTOR_MODE", "dummy")

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    first = client.post("/raw-posts/1/extract")
    second = client.post("/raw-posts/1/extract")
    third = client.post("/raw-posts/1/extract")
    fourth = client.post("/raw-posts/1/extract")
    app.dependency_overrides.clear()

    assert first.status_code == 201
    assert second.status_code == 201
    assert third.status_code == 201
    assert fourth.status_code == 429


def test_long_content_gets_truncated_with_meta(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    long_content = "BTC " * 1000
    fake_db._data[RawPost][1].content_text = long_content

    monkeypatch.setenv("EXTRACTOR_MODE", "dummy")
    monkeypatch.setenv("EXTRACTION_MAX_CONTENT_CHARS", "120")

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post("/raw-posts/1/extract")
    app.dependency_overrides.clear()

    assert response.status_code == 201
    body = response.json()
    meta = body["extracted_json"]["meta"]
    assert meta["truncated"] is True
    assert meta["original_length"] == len(long_content)
    assert meta["max_length"] == 120


def test_openai_call_budget_falls_back_to_dummy_after_budget_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_CALL_BUDGET", "1")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    call_count = {"n": 0}

    def fake_extract(self, raw_post: RawPost):  # noqa: ANN001
        call_count["n"] += 1
        return {
            "assets": [{"symbol": "BTC", "name": "Bitcoin", "market": "CRYPTO"}],
            "stance": "bull",
            "horizon": "1w",
            "confidence": 78,
            "summary": "支撑位有效时，短线仍有上行空间。",
            "source_url": raw_post.url,
            "as_of": "2026-02-21",
            "event_tags": ["support"],
        }

    monkeypatch.setattr("services.extraction.OpenAIExtractor.extract", fake_extract)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    first = client.post("/raw-posts/1/extract")
    second = client.post("/raw-posts/1/extract")
    app.dependency_overrides.clear()

    assert first.status_code == 201
    first_body = first.json()
    assert first_body["extractor_name"] == "openai_structured"

    assert second.status_code == 201
    second_body = second.json()
    assert second_body["id"] == first_body["id"]
    assert second_body["extractor_name"] == "openai_structured"
    assert second_body["last_error"] is None
    assert call_count["n"] == 1


def test_extractor_status_includes_openrouter_and_budget_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    model_name = "deepseek/deepseek-v3.2"
    monkeypatch.setenv("EXTRACTOR_MODE", "auto")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENAI_MODEL", model_name)
    monkeypatch.setenv("OPENAI_CALL_BUDGET", "3")
    monkeypatch.setenv("OPENAI_MAX_OUTPUT_TOKENS", "888")

    client = TestClient(app)
    response = client.get("/extractor-status")

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "auto"
    assert body["has_api_key"] is True
    assert body["default_model"] == model_name
    assert body["base_url"] == "https://openrouter.ai/api/v1"
    assert body["call_budget_remaining"] == 3
    assert body["max_output_tokens"] == 888


def test_extractor_status_prefers_env_model_over_settings_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "custom/test-model")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("EXTRACTOR_MODE", "auto")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

    client = TestClient(app)
    response = client.get("/extractor-status")

    assert response.status_code == 200
    body = response.json()
    assert body["default_model"] == "custom/test-model"
    assert body["base_url"] == "https://openrouter.ai/api/v1"


def test_confidence_69_auto_rejects_and_marks_raw_post_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AUTO_APPROVE_ENABLED", "false")

    def fake_extract(self, raw_post: RawPost):  # noqa: ANN001
        return {
            "assets": [{"symbol": "BTC", "name": "Bitcoin", "market": "CRYPTO"}],
            "stance": "bull",
            "horizon": "1w",
            "confidence": 69,
            "summary": "low confidence",
            "source_url": raw_post.url,
            "as_of": "2026-02-21",
            "asset_views": [
                {
                    "symbol": "BTC",
                    "stance": "bull",
                    "horizon": "1w",
                    "confidence": 69,
                    "summary": "btc view",
                }
            ],
        }

    monkeypatch.setattr("services.extraction.OpenAIExtractor.extract", fake_extract)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post("/raw-posts/1/extract")
    app.dependency_overrides.clear()

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == ExtractionStatus.rejected.value
    assert body["reviewed_by"] == "auto"
    assert body["extracted_json"]["meta"]["auto_rejected"] is True
    assert body["extracted_json"]["meta"]["auto_review_reason"] == "no_investment_asset_noneany"
    assert body["extracted_json"]["meta"]["auto_policy_applied"] == "noneany_asset_forced_reject"
    assert fake_db._data[RawPost][1].review_status == ReviewStatus.rejected


def test_auto_review_skips_when_asset_views_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AUTO_APPROVE_ENABLED", "false")

    def fake_extract(self, raw_post: RawPost):  # noqa: ANN001
        return {
            "assets": [],
            "stance": "bull",
            "horizon": "1w",
            "confidence": 80,
            "summary": "no views",
            "source_url": raw_post.url,
            "as_of": "2026-02-21",
            "asset_views": [],
        }

    monkeypatch.setattr("services.extraction.OpenAIExtractor.extract", fake_extract)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post("/raw-posts/1/extract")
    app.dependency_overrides.clear()

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == ExtractionStatus.rejected.value
    assert body["reviewed_by"] == "auto"
    assert body["extracted_json"]["meta"]["auto_rejected"] is True
    assert body["extracted_json"]["meta"]["auto_review_reason"] == "no_investment_asset_noneany"
    assert fake_db._data[RawPost][1].review_status == ReviewStatus.rejected


def test_noneany_assets_auto_rejected_and_not_in_pending_latest(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AUTO_APPROVE_ENABLED", "false")

    def fake_extract(self, raw_post: RawPost):  # noqa: ANN001
        return {
            "assets": ["NoneAny"],
            "stance": "neutral",
            "horizon": "1w",
            "confidence": 80,
            "summary": "daily-life post",
            "source_url": raw_post.url,
            "as_of": "2026-02-21",
            "asset_views": [],
        }

    monkeypatch.setattr("services.extraction.OpenAIExtractor.extract", fake_extract)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post("/raw-posts/1/extract")
    pending = client.get("/extractions?status=pending&limit=20&offset=0")
    app.dependency_overrides.clear()

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == ExtractionStatus.rejected.value
    assert body["reviewed_by"] == "auto"
    assert body["extracted_json"]["meta"]["auto_rejected"] is True
    assert body["extracted_json"]["meta"]["auto_review_reason"] == "no_investment_asset_noneany"
    assert body["extracted_json"]["meta"]["noneany_detected"] is True
    assert body["extracted_json"]["meta"]["auto_policy_applied"] == "noneany_asset_forced_reject"
    assert body["extracted_json"]["meta"].get("auto_approved") is not True
    assert body["extracted_json"]["meta"].get("model_confidence") is None
    assert fake_db._data[RawPost][1].review_status == ReviewStatus.rejected

    assert pending.status_code == 200
    assert pending.json() == []


@pytest.mark.parametrize("confidence", [70, 80])
def test_confidence_70_or_higher_auto_approves_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    confidence: int,
) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AUTO_APPROVE_ENABLED", "false")

    call_count = {"n": 0}

    def fake_extract(self, raw_post: RawPost):  # noqa: ANN001
        call_count["n"] += 1
        return {
            "assets": [{"symbol": "BTC", "name": "Bitcoin", "market": "CRYPTO"}],
            "stance": "bull",
            "horizon": "1w",
            "confidence": confidence,
            "summary": "auto approve check",
            "source_url": raw_post.url,
            "as_of": "2026-02-21",
            "asset_views": [
                {
                    "symbol": "BTC",
                    "stance": "bull",
                    "horizon": "1w",
                    "confidence": confidence,
                    "summary": "btc view",
                }
            ],
        }

    monkeypatch.setattr("services.extraction.OpenAIExtractor.extract", fake_extract)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    first = client.post("/raw-posts/1/extract")
    second = client.post("/raw-posts/1/extract")
    app.dependency_overrides.clear()

    assert first.status_code == 201
    assert second.status_code == 201
    first_body = first.json()
    second_body = second.json()
    assert first_body["status"] == ExtractionStatus.approved.value
    assert first_body["reviewed_by"] == "auto"
    assert first_body["extracted_json"]["meta"]["auto_approved"] is True
    assert first_body["extracted_json"]["meta"]["auto_review_threshold"] == 70
    assert first_body["id"] == second_body["id"]
    assert fake_db._data[RawPost][1].review_status == ReviewStatus.approved
    assert len(fake_db._data[KolView]) == 1
    assert call_count["n"] == 1


def test_force_reextract_ignores_has_result_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AUTO_APPROVE_ENABLED", "false")

    call_count = {"n": 0}

    def fake_extract(self, raw_post: RawPost):  # noqa: ANN001
        call_count["n"] += 1
        return {
            "assets": [{"symbol": "BTC", "name": "Bitcoin", "market": "CRYPTO"}],
            "stance": "bull",
            "horizon": "1w",
            "confidence": 69,
            "summary": f"force run {call_count['n']}",
            "source_url": raw_post.url,
            "as_of": "2026-02-21",
        }

    monkeypatch.setattr("services.extraction.OpenAIExtractor.extract", fake_extract)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    first = client.post("/raw-posts/1/extract")
    normal_repeat = client.post("/raw-posts/1/extract")
    forced = client.post("/raw-posts/1/extract?force=true")
    app.dependency_overrides.clear()

    assert first.status_code == 201
    assert normal_repeat.status_code == 201
    assert forced.status_code == 201
    assert normal_repeat.json()["id"] == first.json()["id"]
    assert forced.json()["id"] != first.json()["id"]
    assert call_count["n"] == 2


def test_manual_force_reextract_skips_threshold_auto_review(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    now = datetime.now(UTC)
    fake_db.seed(
        PostExtraction(
            id=9,
            raw_post_id=1,
            status=ExtractionStatus.pending,
            extracted_json={"summary": "old pending", "asset_views": []},
            model_name="test-model",
            extractor_name="openai_structured",
            created_at=now - timedelta(minutes=1),
        )
    )
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AUTO_APPROVE_ENABLED", "false")
    monkeypatch.setenv("AUTO_APPROVE_ENABLED", "false")

    def fake_extract(self, raw_post: RawPost):  # noqa: ANN001
        return {
            "assets": [{"symbol": "BTC", "name": "Bitcoin", "market": "CRYPTO"}],
            "stance": "bull",
            "horizon": "1w",
            "confidence": 95,
            "summary": "manual force",
            "source_url": raw_post.url,
            "as_of": "2026-02-21",
            "asset_views": [
                {
                    "symbol": "BTC",
                    "stance": "bull",
                    "horizon": "1w",
                    "confidence": 95,
                    "summary": "manual review needed",
                }
            ],
        }

    monkeypatch.setattr("services.extraction.OpenAIExtractor.extract", fake_extract)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    forced = client.post("/extractions/9/re-extract")
    app.dependency_overrides.clear()

    assert forced.status_code == 201
    body = forced.json()
    assert body["status"] == ExtractionStatus.pending.value
    assert body["reviewed_by"] is None
    assert body["extracted_json"]["meta"]["force_reextract"] is True
    assert body["extracted_json"]["meta"]["force_reextract_triggered_by"] == "user"
    assert body["extracted_json"]["meta"].get("auto_approved") is not True
    assert body["extracted_json"]["meta"].get("auto_rejected") is not True
    assert body["extracted_json"]["meta"]["auto_policy_applied"] == "no_auto_review_user_trigger"


def test_text_json_unwraps_extracted_json_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fake_extract(self, raw_post: RawPost):  # noqa: ANN001
        return {
            "extracted_json": {
                "assets": [{"symbol": "BTC", "name": "Bitcoin", "market": "CRYPTO"}],
                "stance": "bull",
                "horizon": "1w",
                "confidence": 88,
                "reasoning": "中文",
                "summary": "unwrap",
                "source_url": raw_post.url,
                "as_of": "2026-02-21",
                "event_tags": [],
                "asset_views": [
                    {
                        "symbol": "BTC",
                        "stance": "bull",
                        "horizon": "1w",
                        "confidence": 88,
                        "reasoning": "中文",
                        "summary": "unwrap",
                    }
                ],
            }
        }

    monkeypatch.setattr("services.extraction.OpenAIExtractor.extract", fake_extract)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post("/raw-posts/1/extract")
    app.dependency_overrides.clear()

    assert response.status_code == 201
    body = response.json()
    extracted = body["extracted_json"]
    assert extracted["assets"][0]["symbol"] == "BTC"
    assert extracted["stance"] == "bull"
    assert extracted["source_url"] == "https://x.com/alice/status/post-1"
    assert "extracted_json" not in extracted
    assert extracted["meta"]["parse_unwrapped_extracted_json"] is True
    assert extracted["meta"]["parse_unwrapped_key"] == "extracted_json"


def test_force_reextract_reparses_and_persists_extracted_json(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    now = datetime.now(UTC)
    fake_db.seed(
        PostExtraction(
            id=9,
            raw_post_id=1,
            status=ExtractionStatus.pending,
            extracted_json={
                "assets": [],
                "reasoning": None,
                "stance": None,
                "horizon": None,
                "confidence": None,
                "summary": None,
                "source_url": None,
                "as_of": None,
                "event_tags": [],
                "asset_views": [],
            },
            model_name="dummy-v2",
            extractor_name="dummy",
            created_at=now - timedelta(minutes=1),
        )
    )
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fake_extract(self, raw_post: RawPost):  # noqa: ANN001
        return {
            "extracted_json": {
                "assets": [{"symbol": "BTC", "name": "Bitcoin", "market": "CRYPTO"}],
                "stance": "bull",
                "horizon": "1w",
                "confidence": 92,
                "reasoning": "中文",
                "summary": "fresh parse",
                "source_url": raw_post.url,
                "as_of": "2026-02-21",
                "event_tags": [],
                "asset_views": [
                    {
                        "symbol": "BTC",
                        "stance": "bull",
                        "horizon": "1w",
                        "confidence": 92,
                        "reasoning": "中文",
                        "summary": "fresh parse",
                    }
                ],
            }
        }

    monkeypatch.setattr("services.extraction.OpenAIExtractor.extract", fake_extract)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    forced = client.post("/extractions/9/re-extract")
    app.dependency_overrides.clear()

    assert forced.status_code == 201
    body = forced.json()
    assert body["id"] != 9
    assert body["extracted_json"]["summary"] == "fresh parse"
    assert body["extracted_json"]["assets"][0]["symbol"] == "BTC"
    assert "extracted_json" not in body["extracted_json"]
    assert body["extracted_json"]["meta"]["parse_unwrapped_extracted_json"] is True


def test_force_reextract_deactivates_previous_active_and_creates_new_active(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    now = datetime.now(UTC)
    fake_db.seed(
        PostExtraction(
            id=9,
            raw_post_id=1,
            status=ExtractionStatus.pending,
            extracted_json={"summary": "old active pending", "asset_views": [], "meta": {}},
            model_name="gpt-4o-mini",
            extractor_name="openai_structured",
            last_error=None,
            created_at=now - timedelta(minutes=1),
        )
    )
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fake_extract(self, raw_post: RawPost):  # noqa: ANN001
        return {
            "assets": [{"symbol": "BTC", "market": "CRYPTO"}],
            "stance": "bull",
            "horizon": "1w",
            "confidence": 88,
            "summary": "new pending extraction",
            "source_url": raw_post.url,
            "as_of": "2026-02-26",
            "event_tags": [],
            "asset_views": [],
        }

    monkeypatch.setattr("services.extraction.OpenAIExtractor.extract", fake_extract)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post("/extractions/9/re-extract")
    app.dependency_overrides.clear()

    assert response.status_code == 201
    body = response.json()
    assert body["id"] != 9

    old_extraction = fake_db._data[PostExtraction].get(9)
    new_extraction = fake_db._data[PostExtraction][body["id"]]
    if old_extraction is not None:
        assert (old_extraction.last_error or "").startswith("superseded_by_force_reextract:user")
        assert old_extraction.extracted_json["meta"]["deactivated_by_force_reextract"] is True
        assert old_extraction.extracted_json["meta"]["deactivated_reason"].startswith("superseded_by_force_reextract:user")
    assert (new_extraction.last_error or "").strip() == ""

    active_rows = [
        row
        for row in fake_db._data[PostExtraction].values()
        if row.raw_post_id == 1 and row.status == ExtractionStatus.pending and not (row.last_error or "").strip()
    ]
    assert len(active_rows) <= 1
    if active_rows:
        assert active_rows[0].id == body["id"]


def test_noneany_rejects_even_on_manual_force_reextract(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    now = datetime.now(UTC)
    fake_db.seed(
        PostExtraction(
            id=9,
            raw_post_id=1,
            status=ExtractionStatus.pending,
            extracted_json={"summary": "old pending", "asset_views": []},
            model_name="test-model",
            extractor_name="openai_structured",
            created_at=now - timedelta(minutes=1),
        )
    )
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fake_extract(self, raw_post: RawPost):  # noqa: ANN001
        return {
            "extracted_json": {
                "assets": ["NoneAny"],
                "stance": "neutral",
                "horizon": "1w",
                "confidence": 80,
                "reasoning": "中文",
                "summary": "no investment asset",
                "source_url": raw_post.url,
                "as_of": "2026-02-21",
                "event_tags": [],
                "asset_views": [],
            }
        }

    monkeypatch.setattr("services.extraction.OpenAIExtractor.extract", fake_extract)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    forced = client.post("/extractions/9/re-extract")
    app.dependency_overrides.clear()

    assert forced.status_code == 201
    body = forced.json()
    assert body["status"] == ExtractionStatus.rejected.value
    assert body["extracted_json"]["meta"]["noneany_detected"] is True
    assert body["extracted_json"]["meta"]["auto_review_reason"] == "no_investment_asset_noneany"


def test_library_content_noneany_can_be_auto_approved(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AUTO_APPROVE_ENABLED", "false")

    def fake_extract(self, raw_post: RawPost):  # noqa: ANN001
        return {
            "assets": [{"symbol": "NoneAny", "name": None, "market": "OTHER"}],
            "stance": "neutral",
            "horizon": "1w",
            "confidence": 10,
            "summary": "macro regime note",
            "reasoning": "宏观风险偏好回落。",
            "source_url": raw_post.url,
            "as_of": "2026-02-21",
            "event_tags": [],
            "asset_views": [],
            "content_kind": "library",
            "library_entry": {"confidence": 80, "tags": ["macro", "risk"], "summary": "高价值宏观框架"},
        }

    monkeypatch.setattr("services.extraction.OpenAIExtractor.extract", fake_extract)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post("/raw-posts/1/extract")
    app.dependency_overrides.clear()

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == ExtractionStatus.approved.value
    assert body["reviewed_by"] == "auto"
    assert body["extracted_json"]["content_kind"] == "library"
    assert body["extracted_json"]["assets"] == [{"symbol": "NoneAny", "name": None, "market": "OTHER"}]
    assert body["extracted_json"]["asset_views"] == []
    assert body["extracted_json"]["meta"]["auto_approved"] is True
    assert body["extracted_json"]["meta"]["auto_review_reason"] == "confidence_at_or_above_threshold"
    assert body["extracted_json"]["meta"]["auto_policy_applied"] == "threshold_library"


def test_library_content_noneany_can_be_auto_rejected_by_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AUTO_APPROVE_ENABLED", "false")

    def fake_extract(self, raw_post: RawPost):  # noqa: ANN001
        return {
            "assets": [{"symbol": "NoneAny", "name": None, "market": "OTHER"}],
            "stance": "neutral",
            "horizon": "1w",
            "confidence": 99,
            "summary": "macro watch",
            "reasoning": "主题仍在观察。",
            "source_url": raw_post.url,
            "as_of": "2026-02-21",
            "event_tags": [],
            "asset_views": [],
            "content_kind": "library",
            "library_entry": {"confidence": 69, "tags": ["macro"], "summary": "仍需跟踪"},
        }

    monkeypatch.setattr("services.extraction.OpenAIExtractor.extract", fake_extract)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post("/raw-posts/1/extract")
    app.dependency_overrides.clear()

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == ExtractionStatus.rejected.value
    assert body["reviewed_by"] == "auto"
    assert body["extracted_json"]["content_kind"] == "asset"
    assert body["extracted_json"]["library_entry"] is None
    assert body["extracted_json"]["meta"]["library_entry_dropped"] is True
    assert body["extracted_json"]["meta"]["library_entry_drop_reason"] == "low_library_confidence"
    assert body["extracted_json"]["meta"]["library_downgraded"] is True
    assert body["extracted_json"]["meta"]["library_downgrade_reason"] == "low_library_confidence"
    assert body["extracted_json"]["meta"]["auto_rejected"] is True
    assert body["extracted_json"]["meta"]["auto_review_reason"] == "no_investment_asset_noneany"
    assert body["extracted_json"]["meta"]["auto_policy_applied"] == "noneany_asset_forced_reject"


def test_library_auto_review_uses_library_entry_confidence_69() -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    raw_post = fake_db._data[RawPost][1]
    extraction = PostExtraction(
        id=77,
        raw_post_id=1,
        status=ExtractionStatus.pending,
        extracted_json={
            "assets": [{"symbol": "NoneAny", "name": None, "market": "OTHER"}],
            "asset_views": [],
            "content_kind": "library",
            "confidence": 99,
            "library_entry": {"confidence": 69, "tags": ["macro"], "summary": "macro track"},
        },
        model_name="test-model",
        extractor_name="openai_structured",
        created_at=datetime.now(UTC),
    )
    fake_db.seed(extraction)

    outcome = asyncio.run(postprocess_auto_review(db=fake_db, extraction=extraction, raw_post=raw_post, trigger="auto"))

    assert outcome == "rejected"
    assert extraction.status == ExtractionStatus.rejected
    assert extraction.extracted_json["meta"]["model_confidence"] == 69
    assert extraction.extracted_json["meta"]["auto_policy_applied"] == "threshold_library"
    assert extraction.extracted_json["meta"]["auto_review_reason"] == "confidence_below_threshold"


def test_library_auto_review_uses_library_entry_confidence_70() -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    raw_post = fake_db._data[RawPost][1]
    extraction = PostExtraction(
        id=78,
        raw_post_id=1,
        status=ExtractionStatus.pending,
        extracted_json={
            "assets": [{"symbol": "NoneAny", "name": None, "market": "OTHER"}],
            "asset_views": [],
            "content_kind": "library",
            "confidence": 1,
            "library_entry": {"confidence": 70, "tags": ["macro"], "summary": "macro framework"},
        },
        model_name="test-model",
        extractor_name="openai_structured",
        created_at=datetime.now(UTC),
    )
    fake_db.seed(extraction)

    outcome = asyncio.run(postprocess_auto_review(db=fake_db, extraction=extraction, raw_post=raw_post, trigger="auto"))

    assert outcome == "approved"
    assert extraction.status == ExtractionStatus.approved
    assert extraction.extracted_json["meta"]["model_confidence"] == 70
    assert extraction.extracted_json["meta"]["auto_policy_applied"] == "threshold_library"
    assert extraction.extracted_json["meta"]["auto_review_reason"] == "confidence_at_or_above_threshold"


def test_manual_force_reextract_library_skips_threshold_auto_review(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    now = datetime.now(UTC)
    fake_db.seed(
        PostExtraction(
            id=9,
            raw_post_id=1,
            status=ExtractionStatus.pending,
            extracted_json={"summary": "old pending", "asset_views": []},
            model_name="test-model",
            extractor_name="openai_structured",
            created_at=now - timedelta(minutes=1),
        )
    )
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AUTO_APPROVE_ENABLED", "false")

    def fake_extract(self, raw_post: RawPost):  # noqa: ANN001
        return {
            "assets": [{"symbol": "NoneAny", "name": None, "market": "OTHER"}],
            "stance": "neutral",
            "horizon": "1w",
            "confidence": 95,
            "summary": "manual library",
            "reasoning": "宏观框架更新。",
            "source_url": raw_post.url,
            "as_of": "2026-02-21",
            "event_tags": [],
            "asset_views": [],
            "content_kind": "library",
            "library_entry": {"confidence": 95, "tags": ["macro", "strategy"], "summary": "可复用方法论"},
        }

    monkeypatch.setattr("services.extraction.OpenAIExtractor.extract", fake_extract)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    forced = client.post("/extractions/9/re-extract")
    app.dependency_overrides.clear()

    assert forced.status_code == 201
    body = forced.json()
    assert body["status"] == ExtractionStatus.pending.value
    assert body["reviewed_by"] is None
    assert body["extracted_json"]["content_kind"] == "library"
    assert body["extracted_json"]["meta"]["force_reextract_triggered_by"] == "user"
    assert body["extracted_json"]["meta"].get("auto_approved") is not True
    assert body["extracted_json"]["meta"].get("auto_rejected") is not True
    assert body["extracted_json"]["meta"]["auto_policy_applied"] == "no_auto_review_user_trigger"


def test_library_without_valid_library_entry_is_downgraded_to_asset(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fake_extract(self, raw_post: RawPost):  # noqa: ANN001
        return {
            "assets": [{"symbol": "NoneAny", "name": None, "market": "OTHER"}],
            "stance": "neutral",
            "horizon": "1w",
            "confidence": 55,
            "summary": "library with empty tags",
            "reasoning": "宏观观察。",
            "source_url": raw_post.url,
            "as_of": "2026-02-21",
            "event_tags": [],
            "asset_views": [],
            "content_kind": "library",
        }

    monkeypatch.setattr("services.extraction.OpenAIExtractor.extract", fake_extract)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post("/raw-posts/1/extract")
    app.dependency_overrides.clear()

    assert response.status_code == 201
    body = response.json()
    assert body["last_error"] is None
    assert body["extracted_json"]["content_kind"] == "asset"
    assert body["extracted_json"]["assets"] == [{"symbol": "NoneAny", "name": None, "market": "OTHER"}]
    assert body["extracted_json"]["asset_views"] == []
    assert body["extracted_json"]["library_entry"] is None
    assert body["extracted_json"]["meta"]["library_downgraded"] is True
    assert body["extracted_json"]["meta"]["library_downgrade_reason"] == "invalid_library_shape"
    assert body["extracted_json"]["meta"]["content_kind_raw"] == "library"
    assert body["extracted_json"]["meta"]["content_kind_original"] == "library"
    assert body["status"] == ExtractionStatus.rejected.value


def test_library_unknown_library_tags_are_dropped_and_downgraded(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fake_extract(self, raw_post: RawPost):  # noqa: ANN001
        return {
            "assets": [{"symbol": "NoneAny", "name": None, "market": "OTHER"}],
            "stance": "neutral",
            "horizon": "1w",
            "confidence": 55,
            "summary": "library with bad tags",
            "reasoning": "行业观察。",
            "source_url": raw_post.url,
            "as_of": "2026-02-21",
            "event_tags": [],
            "asset_views": [],
            "content_kind": "library",
            "library_tags": ["macro", "foo"],
            "library_entry": {"confidence": 80, "tags": ["macro", "foo"], "summary": "bad tags"},
        }

    monkeypatch.setattr("services.extraction.OpenAIExtractor.extract", fake_extract)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post("/raw-posts/1/extract")
    app.dependency_overrides.clear()

    assert response.status_code == 201
    body = response.json()
    assert body["last_error"] is None
    assert body["extracted_json"]["content_kind"] == "asset"
    assert body["extracted_json"]["library_entry"] is None
    assert body["extracted_json"]["meta"]["library_entry_dropped"] is True
    assert body["extracted_json"]["meta"]["library_entry_drop_reason"] == "invalid_library_tags"
    assert body["extracted_json"]["meta"]["library_downgraded"] is True
    assert body["extracted_json"]["meta"]["library_downgrade_reason"] == "invalid_library_tags"
    assert body["status"] == ExtractionStatus.rejected.value


def test_library_no_longer_uses_low_value_style_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fake_extract(self, raw_post: RawPost):  # noqa: ANN001
        return {
            "assets": [{"symbol": "NoneAny", "name": None, "market": "OTHER"}],
            "stance": "neutral",
            "horizon": "1w",
            "confidence": 99,
            "summary": "顺势而为",
            "reasoning": "顺势而为，大势所趋。",
            "source_url": raw_post.url,
            "as_of": "2026-02-21",
            "event_tags": [],
            "asset_views": [],
            "content_kind": "library",
            "library_entry": {"confidence": 95, "tags": ["macro"], "summary": "顺势而为"},
        }

    monkeypatch.setattr("services.extraction.OpenAIExtractor.extract", fake_extract)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post("/raw-posts/1/extract")
    app.dependency_overrides.clear()

    assert response.status_code == 201
    body = response.json()
    assert body["extracted_json"]["content_kind"] == "library"
    assert body["extracted_json"]["library_entry"]["confidence"] == 95
    assert body["extracted_json"]["meta"].get("library_entry_quality_gate") is None
    assert body["status"] == ExtractionStatus.approved.value


def test_library_asset_views_are_cleared_with_meta(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AUTO_APPROVE_ENABLED", "false")

    def fake_extract(self, raw_post: RawPost):  # noqa: ANN001
        return {
            "assets": [{"symbol": "NoneAny", "name": None, "market": "OTHER"}],
            "stance": "neutral",
            "horizon": "1w",
            "confidence": 80,
            "summary": "library with noisy asset views",
            "reasoning": "主题观察。",
            "source_url": raw_post.url,
            "as_of": "2026-02-21",
            "event_tags": [],
            "asset_views": [
                {"symbol": "BTC", "stance": "bull", "horizon": "1w", "confidence": 80, "summary": "should clear"},
            ],
            "content_kind": "library",
            "library_entry": {"confidence": 88, "tags": ["thesis"], "summary": "高质量主题总结"},
        }

    monkeypatch.setattr("services.extraction.OpenAIExtractor.extract", fake_extract)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post("/raw-posts/1/extract")
    app.dependency_overrides.clear()

    assert response.status_code == 201
    body = response.json()
    assert body["extracted_json"]["asset_views"] == []
    assert body["extracted_json"]["meta"]["library_asset_views_cleared"] is True
    assert body["extracted_json"]["meta"]["library_asset_views_original_count"] == 1
    assert body["extracted_json"]["meta"]["library_asset_views_final_count"] == 0


def test_noneany_mixed_with_symbols_marks_last_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AUTO_APPROVE_ENABLED", "false")

    def fake_extract(self, raw_post: RawPost):  # noqa: ANN001
        return {
            "assets": ["NoneAny", "BTC", ""],
            "stance": "neutral",
            "horizon": "1w",
            "confidence": 70,
            "summary": "mixed assets",
            "source_url": raw_post.url,
            "as_of": "2026-02-21",
            "event_tags": [],
            "asset_views": [],
            "content_kind": "asset",
        }

    monkeypatch.setattr("services.extraction.OpenAIExtractor.extract", fake_extract)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post("/raw-posts/1/extract")
    app.dependency_overrides.clear()

    assert response.status_code == 201
    body = response.json()
    assert "noneany_mixed_with_symbols" in (body["last_error"] or "")
    assert body["extracted_json"]["meta"]["noneany_mixed_with_symbols"] is True


def test_top_level_library_tags_are_ignored_and_not_persisted(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fake_extract(self, raw_post: RawPost):  # noqa: ANN001
        return {
            "assets": [{"symbol": "BTC", "name": "Bitcoin", "market": "CRYPTO"}],
            "stance": "bull",
            "horizon": "1w",
            "confidence": 75,
            "summary": "asset with bad library tags",
            "reasoning": "短期偏多。",
            "source_url": raw_post.url,
            "as_of": "2026-02-21",
            "event_tags": [],
            "asset_views": [{"symbol": "BTC", "stance": "bull", "horizon": "1w", "confidence": 75, "summary": "ok"}],
            "content_kind": "asset",
            "library_tags": ["events", "foo"],
        }

    monkeypatch.setattr("services.extraction.OpenAIExtractor.extract", fake_extract)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post("/raw-posts/1/extract")
    app.dependency_overrides.clear()

    assert response.status_code == 201
    body = response.json()
    assert body["last_error"] is None
    assert body["extracted_json"]["content_kind"] == "asset"
    assert body["extracted_json"]["library_entry"] is None
    assert "library_tags" not in body["extracted_json"]
    assert body["extracted_json"]["meta"]["library_tags_stripped"] is True


def test_user_reextract_creates_new_latest_extraction_in_latest_only_list(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    now = datetime.now(UTC)
    fake_db.seed(
        PostExtraction(
            id=9,
            raw_post_id=1,
            status=ExtractionStatus.pending,
            extracted_json={"summary": "old pending", "asset_views": []},
            model_name="test-model",
            extractor_name="openai_structured",
            created_at=now - timedelta(minutes=1),
        )
    )
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AUTO_APPROVE_ENABLED", "false")

    def fake_extract(self, raw_post: RawPost):  # noqa: ANN001
        return {
            "assets": [{"symbol": "NoneAny", "name": None, "market": "OTHER"}],
            "stance": "neutral",
            "horizon": "1w",
            "confidence": 88,
            "summary": "new user retry",
            "reasoning": "更新后的抽取。",
            "source_url": raw_post.url,
            "as_of": "2026-02-21",
            "event_tags": [],
            "asset_views": [],
            "content_kind": "library",
        }

    monkeypatch.setattr("services.extraction.OpenAIExtractor.extract", fake_extract)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    forced = client.post("/extractions/9/re-extract")
    latest_only = client.get("/extractions?status=all&limit=20&offset=0")
    app.dependency_overrides.clear()

    assert forced.status_code == 201
    forced_body = forced.json()
    assert forced_body["id"] != 9
    assert forced_body["extracted_json"]["meta"]["force_reextract_triggered_by"] == "user"
    assert latest_only.status_code == 200
    rows = latest_only.json()
    assert len(rows) == 1
    assert rows[0]["id"] == forced_body["id"]


def test_legacy_extracted_json_missing_new_fields_is_readable_and_status_unchanged() -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    now = datetime.now(UTC)
    fake_db.seed(
        PostExtraction(
            id=10,
            raw_post_id=1,
            status=ExtractionStatus.approved,
            extracted_json={"summary": "legacy approved payload", "asset_views": [], "library_tags": ["macro"]},
            model_name="legacy-model",
            extractor_name="openai_structured",
            created_at=now,
        )
    )
    fake_db.seed(
        PostExtraction(
            id=11,
            raw_post_id=1,
            status=ExtractionStatus.rejected,
            extracted_json={"summary": "legacy rejected payload"},
            model_name="legacy-model",
            extractor_name="openai_structured",
            created_at=now - timedelta(minutes=1),
        )
    )

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    list_resp = client.get("/extractions?status=all&show_history=true&limit=20&offset=0")
    approved_resp = client.get("/extractions/10")
    rejected_resp = client.get("/extractions/11")
    app.dependency_overrides.clear()

    assert list_resp.status_code == 200
    rows = list_resp.json()
    by_id = {item["id"]: item for item in rows}
    assert by_id[10]["status"] == ExtractionStatus.approved.value
    assert by_id[11]["status"] == ExtractionStatus.rejected.value
    assert by_id[10]["extracted_json"]["content_kind"] == "asset"
    assert by_id[10]["extracted_json"]["library_entry"] is None
    assert "library_tags" not in by_id[10]["extracted_json"]

    assert approved_resp.status_code == 200
    approved_body = approved_resp.json()
    assert approved_body["status"] == ExtractionStatus.approved.value
    assert approved_body["extracted_json"]["content_kind"] == "asset"
    assert approved_body["extracted_json"]["library_entry"] is None
    assert "library_tags" not in approved_body["extracted_json"]

    assert rejected_resp.status_code == 200
    rejected_body = rejected_resp.json()
    assert rejected_body["status"] == ExtractionStatus.rejected.value
    assert rejected_body["extracted_json"]["content_kind"] == "asset"
    assert rejected_body["extracted_json"]["library_entry"] is None
    assert "library_tags" not in rejected_body["extracted_json"]


def test_extract_batch_auto_approved_count_increments(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AUTO_APPROVE_ENABLED", "false")

    def fake_extract(self, raw_post: RawPost):  # noqa: ANN001
        return {
            "assets": [{"symbol": "BTC", "name": "Bitcoin", "market": "CRYPTO"}],
            "stance": "bull",
            "horizon": "1w",
            "confidence": 80,
            "summary": "batch approve",
            "source_url": raw_post.url,
            "as_of": "2026-02-21",
            "asset_views": [
                {"symbol": "BTC", "stance": "bull", "horizon": "1w", "confidence": 80, "summary": "batch view"}
            ],
        }

    monkeypatch.setattr("services.extraction.OpenAIExtractor.extract", fake_extract)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post("/raw-posts/extract-batch", json={"raw_post_ids": [1], "mode": "pending_or_failed"})
    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["success_count"] == 1
    assert body["auto_approved_count"] == 1
    assert body["auto_rejected_count"] == 0


def test_openrouter_forces_text_json_mode_and_never_sends_json_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_payloads: list[dict] = []

    class _FakeResponse:
        status_code = 200

        def json(self):  # noqa: ANN001
            return {
                "choices": [{"message": {"content": "{\"assets\":[],\"asset_views\":[],\"source_url\":\"x\"}"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }

    class _FakeClient:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            pass

        def __enter__(self):  # noqa: ANN001
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
            return False

        def post(self, url: str, headers: dict, json: dict):  # noqa: A002
            captured_payloads.append(json)
            return _FakeResponse()

    monkeypatch.setattr("services.extraction.httpx.Client", _FakeClient)
    extractor = OpenAIExtractor(
        api_key="test-key",
        model_name="deepseek/deepseek-v3.2",
        base_url="https://openrouter.ai/api/v1",
        timeout_seconds=10,
        max_output_tokens=256,
    )
    raw_post = RawPost(
        id=999,
        platform="x",
        author_handle="alice",
        external_id="ext-999",
        url="https://x.com/alice/status/999",
        content_text="BTC trend",
        posted_at=datetime(2026, 2, 21, 9, 0, tzinfo=UTC),
        fetched_at=datetime.now(UTC),
        raw_json=None,
    )
    extracted = extractor.extract(raw_post)

    assert extractor.output_mode == "text_json"
    assert extracted["meta"]["provider_detected"] == "openrouter"
    assert extracted["meta"]["output_mode_used"] == "text_json"
    assert len(captured_payloads) == 1
    assert captured_payloads[0]["model"] == "deepseek/deepseek-v3.2"
    assert "response_format" not in captured_payloads[0]


def test_openrouter_wrapped_extracted_json_is_unwrapped_and_noneany_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeResponse:
        status_code = 200

        def json(self):  # noqa: ANN001
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "{\"extracted_json\":{\"assets\":[\"NoneAny\"],\"asset_views\":[],\"stance\":\"neutral\","
                                "\"horizon\":\"1w\",\"confidence\":30,\"reasoning\":\"中文\",\"summary\":\"x\",\"source_url\":\"x\","
                                "\"as_of\":\"2026-02-21\",\"event_tags\":[]}}"
                            )
                        }
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }

    class _FakeClient:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            pass

        def __enter__(self):  # noqa: ANN001
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
            return False

        def post(self, url: str, headers: dict, json: dict):  # noqa: A002
            return _FakeResponse()

    monkeypatch.setattr("services.extraction.httpx.Client", _FakeClient)
    extractor = OpenAIExtractor(
        api_key="test-key",
        model_name="deepseek/deepseek-v3.2",
        base_url="https://openrouter.ai/api/v1",
        timeout_seconds=10,
        max_output_tokens=256,
    )
    raw_post = RawPost(
        id=999,
        platform="x",
        author_handle="alice",
        external_id="ext-999",
        url="https://x.com/alice/status/999",
        content_text="BTC trend",
        posted_at=datetime(2026, 2, 21, 9, 0, tzinfo=UTC),
        fetched_at=datetime.now(UTC),
        raw_json=None,
    )
    extracted = extractor.extract(raw_post)

    assert extracted["assets"] == [{"symbol": "NoneAny", "name": None, "market": "OTHER"}]
    assert extracted["asset_views"] == []
    assert extracted["stance"] == "neutral"


def test_build_extract_prompt_hash_matches_sha256_text_and_version_extract_v1() -> None:
    bundle = build_extract_prompt(
        prompt_version="extract_v1",
        platform="x",
        author_handle="alice",
        url="https://x.com/alice/status/42",
        posted_at=datetime(2026, 2, 25, 9, 0, tzinfo=UTC),
        content_text="BTC trend remains constructive",
        assets=[{"symbol": "BTC", "name": "Bitcoin", "market": "CRYPTO"}],
        aliases=[{"alias": "大饼", "symbol": "BTC"}],
        max_assets_in_prompt=50,
    )

    expected_hash = hashlib.sha256(bundle.text.encode("utf-8")).hexdigest()
    assert bundle.version == "extract_v1"
    assert bundle.hash == expected_hash


def test_call_openai_payload_uses_prompt_bundle_and_mode_specific_system_rules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_payloads: list[dict] = []

    class _FakeResponse:
        status_code = 200

        def json(self):  # noqa: ANN001
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "{\"assets\":[],\"reasoning\":\"中文推理\",\"stance\":null,"
                                "\"horizon\":null,\"confidence\":null,\"summary\":null,"
                                "\"source_url\":null,\"as_of\":null,\"event_tags\":[],\"asset_views\":[]}"
                            )
                        }
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3},
            }

    class _FakeClient:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            pass

        def __enter__(self):  # noqa: ANN001
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
            return False

        def post(self, url: str, headers: dict, json: dict):  # noqa: A002
            captured_payloads.append(json)
            return _FakeResponse()

    monkeypatch.setattr("services.extraction.httpx.Client", _FakeClient)
    extractor = OpenAIExtractor(
        api_key="test-key",
        model_name="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        timeout_seconds=10,
        max_output_tokens=256,
    )
    extractor.set_prompt_bundle(
        PromptBundle(version="extract_v1", text="PROMPT_BUNDLE_USER_TEXT", hash="test-hash")
    )
    raw_post = RawPost(
        id=1001,
        platform="x",
        author_handle="alice",
        external_id="ext-1001",
        url="https://x.com/alice/status/1001",
        content_text="BTC trend",
        posted_at=datetime(2026, 2, 25, 9, 0, tzinfo=UTC),
        fetched_at=datetime.now(UTC),
        raw_json=None,
    )

    extractor._call_openai(raw_post, response_mode="structured")
    extractor.set_summary_language_retry_hint(True)
    extractor._call_openai(raw_post, response_mode="text_json")

    assert len(captured_payloads) == 2

    structured_payload = captured_payloads[0]
    assert structured_payload["messages"][1]["content"] == "PROMPT_BUNDLE_USER_TEXT"
    assert "response_format" in structured_payload
    assert "json_schema" in structured_payload["response_format"]
    structured_system = structured_payload["messages"][0]["content"]
    assert "Only output 1 JSON object; no markdown, no code fence, no text outside JSON." in structured_system
    assert "Top-level fields must be exactly: as_of, source_url, content_kind, assets, asset_views, library_entry." in structured_system
    assert "asset_views[*].confidence must be an integer >=70; remove any item with confidence<70." in structured_system
    assert "Correction retry: previous output had non-Chinese summary." not in structured_system

    text_json_payload = captured_payloads[1]
    assert text_json_payload["messages"][1]["content"] == "PROMPT_BUNDLE_USER_TEXT"
    assert "response_format" not in text_json_payload
    text_json_system = text_json_payload["messages"][0]["content"]
    assert "Only output 1 JSON object; no markdown, no code fence, no text outside JSON." in text_json_system
    assert "Top-level fields must be exactly: as_of, source_url, content_kind, assets, asset_views, library_entry." in text_json_system


def test_text_json_parser_handles_codeblock_and_prefix_suffix() -> None:
    text = """note before
```json
{"assets":[{"symbol":"BTC"}],"asset_views":[]}
```
note after"""
    parsed, strategy, repaired = parse_text_json_object(text)
    assert parsed["assets"][0]["symbol"] == "BTC"
    assert strategy in {"strip_codeblock", "outermost_object"}
    assert repaired is True


def test_structured_unsupported_switches_to_text_json_immediately_no_retry_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_CALL_BUDGET", "1")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    calls = {"n": 0}

    def fake_call_openai(self, raw_post: RawPost, *, response_mode: str):  # noqa: ANN001
        calls["n"] += 1
        if response_mode == "structured":
            raise OpenAIRequestError(status_code=400, body_preview="structured-outputs unsupported")
        return {
            "raw_content": "{\"assets\":[],\"asset_views\":[],\"summary\":\"ok\"}",
            "parsed_content": {"assets": [], "asset_views": [], "summary": "ok", "source_url": raw_post.url},
            "parse_strategy_used": "direct_json",
            "raw_len": 42,
            "repaired": False,
        }

    monkeypatch.setattr("services.extraction.OpenAIExtractor._call_openai", fake_call_openai)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post("/raw-posts/1/extract")
    status_resp = client.get("/extractor-status")
    app.dependency_overrides.clear()

    assert response.status_code == 201
    body = response.json()
    assert body["last_error"] is None
    assert body["extracted_json"]["meta"]["fallback_reason"] == "structured_unsupported"
    assert body["extracted_json"]["meta"]["output_mode_used"] == "text_json"
    assert calls["n"] == 2
    assert status_resp.status_code == 200
    assert status_resp.json()["call_budget_remaining"] == 0


def test_parse_failure_marks_failed_and_no_dummy_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.delenv("DUMMY_FALLBACK", raising=False)

    def fake_call_openai(self, raw_post: RawPost, *, response_mode: str):  # noqa: ANN001
        raise RuntimeError("OpenAI content is not valid JSON object text")

    monkeypatch.setattr("services.extraction.OpenAIExtractor._call_openai", fake_call_openai)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post("/raw-posts/1/extract")
    app.dependency_overrides.clear()

    assert response.status_code == 201
    body = response.json()
    assert body["extractor_name"] != "dummy"
    assert body["last_error"] is not None
    assert "json" in body["last_error"].lower()
    assert body["extracted_json"]["meta"]["output_mode_used"] == "text_json"
    assert body["extracted_json"]["meta"]["parse_error"] is True


def test_openrouter_truncated_text_json_retry_success_records_meta_and_keeps_json_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENAI_CALL_BUDGET", "4")
    monkeypatch.setenv("AUTO_APPROVE_ENABLED", "false")

    captured_payloads: list[dict] = []
    response_queue = [
        {
            "choices": [
                {
                    "message": {
                        "content": "{\"assets\":[],\"asset_views\":[],\"reasoning\":\"市场偏强\",\"stance\":\"bull"
                    },
                    "finish_reason": "length",
                }
            ],
            "usage": {"prompt_tokens": 11, "completion_tokens": 800},
        },
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "assets": [{"symbol": "BTC", "name": "Bitcoin", "market": "CRYPTO"}],
                                "reasoning": "市场风险偏好回暖，短线延续偏多。",
                                "stance": "bull",
                                "horizon": "1w",
                                "confidence": 77,
                                "summary": "短线偏多",
                                "source_url": "https://x.com/alice/status/post-1",
                                "as_of": "2026-02-21",
                                "event_tags": ["risk_on"],
                                "asset_views": [],
                            },
                            ensure_ascii=False,
                        )
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 210},
        },
    ]

    class _FakeResponse:
        status_code = 200

        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def json(self):  # noqa: ANN001
            return self._payload

    class _FakeClient:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            pass

        def __enter__(self):  # noqa: ANN001
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
            return False

        def post(self, url: str, headers: dict, json: dict):  # noqa: A002
            captured_payloads.append(json)
            return _FakeResponse(response_queue.pop(0))

    monkeypatch.setattr("services.extraction.httpx.Client", _FakeClient)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post("/raw-posts/1/extract")
    app.dependency_overrides.clear()

    assert response.status_code == 201
    body = response.json()
    meta = body["extracted_json"]["meta"]
    assert body["last_error"] is None
    assert body["extractor_name"] == "openrouter_json_mode"
    assert len(captured_payloads) == 2
    assert "response_format" not in captured_payloads[0]
    assert "response_format" not in captured_payloads[1]
    assert "Top-level fields must be exactly: as_of, source_url, content_kind, assets, asset_views, library_entry." in captured_payloads[1]["messages"][0]["content"]
    assert "asset_views[*].confidence must be an integer >=70; remove any item with confidence<70." in captured_payloads[1]["messages"][0]["content"]
    assert meta["output_mode_used"] == "text_json"
    assert meta["truncated_retry_used"] is True
    assert bool(meta.get("parse_error")) is False
    assert meta["raw_len"] > 0
    assert meta["raw_saved_len"] > 0
    assert meta["extra_retry_budget_total"] == 1
    assert meta["extra_retry_budget_used"] == 1
    assert body["raw_model_output"] is not None and len(body["raw_model_output"]) > 0


def test_invalid_json_retry_fixes_unescaped_quote_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENAI_CALL_BUDGET", "4")
    monkeypatch.setenv("AUTO_APPROVE_ENABLED", "false")

    captured_payloads: list[dict] = []
    response_queue = [
        {
            "choices": [
                {
                    "message": {
                        "content": "{\"assets\":[],\"asset_views\":[],\"summary\":\"他说\"世界上最好的生意模式\"继续扩张\",\"stance\":\"bull\"}"
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 11, "completion_tokens": 220},
        },
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "assets": [{"symbol": "SPX/标普500", "name": None, "market": "ETF"}],
                                "stance": "bull",
                                "horizon": "1w",
                                "confidence": 75,
                                "summary": "市场风险偏好回升，指数短线偏强。",
                                "source_url": "https://x.com/alice/status/post-1",
                                "as_of": "2026-02-21",
                                "asset_views": [],
                            },
                            ensure_ascii=False,
                        )
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 190},
        },
    ]

    class _FakeResponse:
        status_code = 200

        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def json(self):  # noqa: ANN001
            return self._payload

    class _FakeClient:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            pass

        def __enter__(self):  # noqa: ANN001
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
            return False

        def post(self, url: str, headers: dict, json: dict):  # noqa: A002
            captured_payloads.append(json)
            return _FakeResponse(response_queue.pop(0))

    monkeypatch.setattr("services.extraction.httpx.Client", _FakeClient)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post("/raw-posts/1/extract")
    app.dependency_overrides.clear()

    assert response.status_code == 201
    body = response.json()
    meta = body["extracted_json"]["meta"]
    assert body["last_error"] is None
    assert len(captured_payloads) == 2
    assert "Top-level fields must be exactly: as_of, source_url, content_kind, assets, asset_views, library_entry." in captured_payloads[1]["messages"][0]["content"]
    assert "Only output 1 JSON object; no markdown, no code fence, no text outside JSON." in captured_payloads[1]["messages"][0]["content"]
    assert meta["invalid_json_retry_used"] is True
    assert bool(meta.get("parse_error")) is False
    assert meta["extra_retry_budget_total"] == 1
    assert meta["extra_retry_budget_used"] == 1


def test_openrouter_invalid_json_text_json_retry_still_fails_with_after_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENAI_CALL_BUDGET", "4")
    monkeypatch.setenv("AUTO_APPROVE_ENABLED", "false")

    response_queue = [
        {
            "choices": [
                {
                    "message": {"content": "{\"assets\":[],\"summary\":\"他说\"世界上最好的生意模式\"\"}"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 11, "completion_tokens": 220},
        },
        {
            "choices": [
                {
                    "message": {"content": "{\"assets\":[],\"summary\":\"再次\"未转义\"\"}"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 190},
        },
    ]

    class _FakeResponse:
        status_code = 200

        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def json(self):  # noqa: ANN001
            return self._payload

    class _FakeClient:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            pass

        def __enter__(self):  # noqa: ANN001
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
            return False

        def post(self, url: str, headers: dict, json: dict):  # noqa: A002
            return _FakeResponse(response_queue.pop(0))

    monkeypatch.setattr("services.extraction.httpx.Client", _FakeClient)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post("/raw-posts/1/extract")
    app.dependency_overrides.clear()

    assert response.status_code == 201
    body = response.json()
    meta = body["extracted_json"]["meta"]
    assert "parse_error_invalid_json_after_retry" in (body["last_error"] or "")
    assert meta["parse_error_reason"] == "invalid_json"
    assert meta["invalid_json_retry_used"] is True
    assert meta["extra_retry_budget_total"] == 1
    assert meta["extra_retry_budget_used"] == 1


def test_openrouter_truncated_text_json_retry_still_fails_with_observable_meta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENAI_CALL_BUDGET", "4")
    monkeypatch.setenv("AUTO_APPROVE_ENABLED", "false")

    response_queue = [
        {
            "choices": [
                {
                    "message": {"content": "{\"assets\":[],\"asset_views\":[],\"reasoning\":\"市场偏强\",\"stance\":\"bull"},
                    "finish_reason": "length",
                }
            ],
            "usage": {"prompt_tokens": 11, "completion_tokens": 800},
        },
        {
            "choices": [
                {
                    "message": {"content": "{\"assets\":[],\"asset_views\":[],\"reasoning\":\"继续截断\",\"stance\":\"bull"},
                    "finish_reason": "length",
                }
            ],
            "usage": {"prompt_tokens": 11, "completion_tokens": 800},
        },
    ]

    class _FakeResponse:
        status_code = 200

        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def json(self):  # noqa: ANN001
            return self._payload

    class _FakeClient:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            pass

        def __enter__(self):  # noqa: ANN001
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
            return False

        def post(self, url: str, headers: dict, json: dict):  # noqa: A002
            return _FakeResponse(response_queue.pop(0))

    monkeypatch.setattr("services.extraction.httpx.Client", _FakeClient)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post("/raw-posts/1/extract")
    app.dependency_overrides.clear()

    assert response.status_code == 201
    body = response.json()
    meta = body["extracted_json"]["meta"]
    assert body["extractor_name"] == "openrouter_json_mode"
    assert "parse_error_truncated_output_after_retry" in (body["last_error"] or "")
    assert meta["parse_error"] is True
    assert meta["parse_error_reason"] == "truncated_output"
    assert meta["truncated_retry_used"] is True
    assert meta["parse_strategy_used"] == "text_json_repair_failed_truncated"
    assert meta["raw_len"] > 0
    assert meta["raw_saved_len"] > 0
    assert meta["finish_reason"] == "length"
    assert meta["finish_reason_best_effort"] is True
    assert body["raw_model_output"] is not None and len(body["raw_model_output"]) > 0


def test_first_output_truncated_and_english_reasoning_uses_single_extra_retry_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENAI_CALL_BUDGET", "5")
    monkeypatch.setenv("AUTO_APPROVE_ENABLED", "false")

    captured_payloads: list[dict] = []
    response_queue = [
        {
            "choices": [
                {
                    "message": {"content": "{\"assets\":[],\"asset_views\":[],\"reasoning\":\"English reasoning starts but gets cut"},
                    "finish_reason": "length",
                }
            ],
            "usage": {"prompt_tokens": 11, "completion_tokens": 800},
        },
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "assets": [{"symbol": "BTC", "name": "Bitcoin", "market": "CRYPTO"}],
                                "reasoning": "Momentum remains positive and upside can continue.",
                                "stance": "bull",
                                "horizon": "1w",
                                "confidence": 74,
                                "summary": "This summary stays fully English after truncated retry and should fail language check.",
                                "source_url": "https://x.com/alice/status/post-1",
                                "as_of": "2026-02-21",
                                "event_tags": ["risk_on"],
                                "asset_views": [],
                            }
                        )
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 210},
        },
    ]

    class _FakeResponse:
        status_code = 200

        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def json(self):  # noqa: ANN001
            return self._payload

    class _FakeClient:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            pass

        def __enter__(self):  # noqa: ANN001
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
            return False

        def post(self, url: str, headers: dict, json: dict):  # noqa: A002
            captured_payloads.append(json)
            return _FakeResponse(response_queue.pop(0))

    monkeypatch.setattr("services.extraction.httpx.Client", _FakeClient)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post("/raw-posts/1/extract")
    app.dependency_overrides.clear()

    assert response.status_code == 201
    body = response.json()
    meta = body["extracted_json"]["meta"]
    assert len(captured_payloads) == 2
    assert "summary_language_violation_no_retry_budget_after_parse_retry" in (body["last_error"] or "")
    assert meta["truncated_retry_used"] is True
    assert meta["summary_language"] == "non_zh"
    assert meta["summary_language_violation"] is True
    assert meta["summary_language_retry_used"] is False
    assert meta["extra_retry_budget_total"] == 1
    assert meta["extra_retry_budget_used"] == 1


def test_openrouter_finish_reason_best_effort_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.delenv("DUMMY_FALLBACK", raising=False)
    monkeypatch.setenv("AUTO_APPROVE_ENABLED", "false")

    class _FakeResponse:
        status_code = 200

        def json(self):  # noqa: ANN001
            return {
                "choices": [
                    {
                        "message": {"content": "{\"assets\":[],\"asset_views\":[],\"reasoning\":\"bad truncate\",\"stance\":\"bull"},
                    }
                ],
                "usage": {"prompt_tokens": 9, "completion_tokens": 120},
            }

    class _FakeClient:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            pass

        def __enter__(self):  # noqa: ANN001
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
            return False

        def post(self, url: str, headers: dict, json: dict):  # noqa: A002
            return _FakeResponse()

    monkeypatch.setattr("services.extraction.httpx.Client", _FakeClient)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post("/raw-posts/1/extract")
    app.dependency_overrides.clear()

    assert response.status_code == 201
    body = response.json()
    meta = body["extracted_json"]["meta"]
    assert meta["parse_error"] is True
    assert meta["parse_error_reason"] == "truncated_output"
    assert meta["finish_reason"] is None
    assert meta["finish_reason_best_effort"] is True


def test_raw_model_output_cap_sets_len_meta_without_breaking_extracted_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("RAW_MODEL_OUTPUT_MAX_CHARS", "80")
    monkeypatch.setenv("AUTO_APPROVE_ENABLED", "false")

    long_summary = "x" * 500

    def fake_call_openai(self, raw_post: RawPost, *, response_mode: str):  # noqa: ANN001
        assert response_mode == "text_json"
        raw_content = json.dumps(
            {
                "assets": [{"symbol": "BTC", "name": "Bitcoin", "market": "CRYPTO"}],
                "reasoning": "中文推理",
                "stance": "bull",
                "horizon": "1w",
                "confidence": 70,
                "summary": long_summary,
                "source_url": raw_post.url,
                "as_of": "2026-02-21",
                "event_tags": [],
                "asset_views": [],
            },
            ensure_ascii=False,
        )
        return {
            "raw_content": raw_content,
            "parsed_content": json.loads(raw_content),
            "latency_ms": 10,
            "input_tokens": 10,
            "output_tokens": 300,
            "parse_strategy_used": "direct_json",
            "raw_len": len(raw_content),
            "repaired": False,
            "parse_error_reason": None,
            "finish_reason": "stop",
        }

    monkeypatch.setattr("services.extraction.OpenAIExtractor._call_openai", fake_call_openai)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post("/raw-posts/1/extract")
    app.dependency_overrides.clear()

    assert response.status_code == 201
    body = response.json()
    meta = body["extracted_json"]["meta"]
    assert meta["raw_len"] > meta["raw_saved_len"]
    assert meta["raw_truncated"] is True
    assert body["raw_model_output"] is not None
    assert len(body["raw_model_output"]) == 80
    assert body["extracted_json"]["summary"] == long_summary
    assert body["extracted_json"]["assets"][0]["symbol"] == "BTC"


def test_structured_unsupported_retries_once_with_json_mode_and_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_CALL_BUDGET", "2")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    calls = {"n": 0}

    def fake_call_openai(self, raw_post: RawPost, *, response_mode: str):  # noqa: ANN001
        calls["n"] += 1
        if response_mode == "structured":
            raise OpenAIRequestError(
                status_code=400,
                body_preview="model does not support feature: structured-outputs",
            )
        return {
            "assets": [{"symbol": "BTC", "name": "Bitcoin", "market": "CRYPTO"}],
            "summary": "strict json mode fallback",
            "source_url": raw_post.url,
        }

    monkeypatch.setattr("services.extraction.OpenAIExtractor._call_openai", fake_call_openai)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    response = client.post("/raw-posts/1/extract")
    status_resp = client.get("/extractor-status")
    app.dependency_overrides.clear()

    assert response.status_code == 201
    body = response.json()
    assert body["extractor_name"] == "openai_structured"
    assert body["last_error"] is None
    assert body["extracted_json"]["meta"]["extraction_mode"] == "text_json"
    assert body["extracted_json"]["meta"]["fallback_reason"] == "structured_unsupported"
    assert calls["n"] == 2
    assert status_resp.status_code == 200
    assert status_resp.json()["call_budget_remaining"] == 1


def test_structured_unsupported_and_json_mode_fail_falls_back_to_dummy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_CALL_BUDGET", "2")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("DUMMY_FALLBACK", "true")

    calls = {"n": 0}

    def fake_call_openai(self, raw_post: RawPost, *, response_mode: str):  # noqa: ANN001
        calls["n"] += 1
        if response_mode == "structured":
            raise OpenAIRequestError(
                status_code=400,
                body_preview="qwen does not support feature: structured-outputs",
            )
        raise RuntimeError("json mode failed")

    monkeypatch.setattr("services.extraction.OpenAIExtractor._call_openai", fake_call_openai)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    response = client.post("/raw-posts/1/extract")
    app.dependency_overrides.clear()

    assert response.status_code == 201
    body = response.json()
    assert body["extractor_name"] == "dummy"
    assert body["last_error"] is not None
    assert "structured unsupported" in body["last_error"].lower()
    assert body["extracted_json"]["meta"]["extraction_mode"] == "dummy"
    assert body["extracted_json"]["meta"]["fallback_reason"] == "structured_unsupported"
    assert calls["n"] == 2


def test_chinese_post_json_mode_normalization(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    _seed_chinese_raw_post(fake_db)
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_CALL_BUDGET", "2")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    calls = {"n": 0}

    def fake_call_openai(self, raw_post: RawPost, *, response_mode: str):  # noqa: ANN001
        calls["n"] += 1
        if response_mode == "structured":
            raise OpenAIRequestError(
                status_code=400,
                body_preview="qwen structured-outputs unsupported for this model",
            )
        return {
            "assets": {"symbol": "SPY", "name": "SPDR S&P 500 ETF", "market": "ETF", "unknown": "drop-me"},
            "stance": "谨慎乐观",
            "horizon": "今晚",
            "confidence": 0.78,
            "summary": "市场偏避险，短期美股稳定但风险偏好走弱。",
            "event_tags": None,
            "source_url": raw_post.url,
            "as_of": "2026-02-21",
            "platform": raw_post.platform,
            "url": raw_post.url,
            "posted_at": raw_post.posted_at.isoformat(),
        }

    monkeypatch.setattr("services.extraction.OpenAIExtractor._call_openai", fake_call_openai)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    response = client.post("/raw-posts/2/extract")
    app.dependency_overrides.clear()

    assert response.status_code == 201
    body = response.json()
    extracted = body["extracted_json"]
    assert body["extractor_name"] == "openai_structured"
    assert body["last_error"] is None
    assert extracted["meta"]["extraction_mode"] == "text_json"
    assert extracted["meta"]["fallback_reason"] == "structured_unsupported"
    assert extracted["horizon"] == "1w"
    assert extracted["stance"] == "neutral"
    assert extracted["confidence"] == 78
    assert extracted["assets"] == [{"symbol": "SPY", "name": "SPDR S&P 500 ETF", "market": "ETF"}]
    assert "platform" not in extracted
    assert "url" not in extracted
    assert "posted_at" not in extracted
    assert extracted["meta"]["extraction_mode"] != "dummy"
    assert calls["n"] == 2


def test_json_mode_assets_auto_create_missing_asset(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_CALL_BUDGET", "2")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    def fake_call_openai(self, raw_post: RawPost, *, response_mode: str):  # noqa: ANN001
        if response_mode == "structured":
            raise OpenAIRequestError(
                status_code=400,
                body_preview="structured-outputs unsupported",
            )
        return {
            "assets": ["VIX"],
            "stance": "看空",
            "horizon": "今晚",
            "confidence": "88",
            "reasoning": "风险偏好下降，波动率上行。",
            "summary": "短线偏防御",
            "event_tags": ["vix", "risk_off"],
            "platform": raw_post.platform,
            "author_handle": raw_post.author_handle,
            "url": raw_post.url,
            "posted_at": raw_post.posted_at.isoformat(),
        }

    monkeypatch.setattr("services.extraction.OpenAIExtractor._call_openai", fake_call_openai)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    response = client.post("/raw-posts/1/extract")
    app.dependency_overrides.clear()

    assert response.status_code == 201
    body = response.json()
    assert body["extractor_name"] == "openai_structured"
    assert body["extracted_json"]["assets"] == [{"symbol": "VIX", "name": None, "market": "AUTO"}]
    assert body["extracted_json"]["as_of"] == "2026-02-21"
    assert body["extracted_json"]["meta"]["extraction_mode"] == "text_json"
    assert body["last_error"] is None
    created_assets = list(fake_db._data[Asset].values())
    assert len(created_assets) == 1
    assert created_assets[0].symbol == "VIX"
    assert created_assets[0].market == "AUTO"


def test_json_mode_allows_auto_market_after_structured_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    fake_db.seed(
        Asset(
            id=99,
            symbol="VIX",
            name="CBOE Volatility Index",
            market="AUTO",
            created_at=datetime.now(UTC),
        )
    )
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_CALL_BUDGET", "2")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    def fake_call_openai(self, raw_post: RawPost, *, response_mode: str):  # noqa: ANN001
        if response_mode == "structured":
            raise OpenAIRequestError(
                status_code=400,
                body_preview="structured-outputs unsupported",
            )
        return {
            "assets": [{"symbol": "VIX", "name": "CBOE Volatility Index", "market": "AUTO"}],
            "summary": "volatility risk remains",
            "source_url": raw_post.url,
        }

    monkeypatch.setattr("services.extraction.OpenAIExtractor._call_openai", fake_call_openai)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    response = client.post("/raw-posts/1/extract")
    app.dependency_overrides.clear()

    assert response.status_code == 201
    body = response.json()
    assert body["extractor_name"] == "openai_structured"
    assert body["last_error"] is None
    assert body["extracted_json"]["meta"]["extraction_mode"] == "text_json"
    assert body["extracted_json"]["assets"][0]["market"] == "OTHER"


def test_normalize_as_of_datetime_to_date_only() -> None:
    normalized = normalize_extracted_json(
        {
            "assets": [{"symbol": "GC=F", "market": "AUTO"}],
            "as_of": "2026-02-21T10:30:45+00:00",
            "posted_at": "2026-02-21T10:00:00+00:00",
        }
    )

    assert normalized["as_of"] == "2026-02-21"


def test_dummy_extractor_persists_prompt_and_raw_output(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    monkeypatch.setenv("EXTRACTOR_MODE", "dummy")
    monkeypatch.setenv("PROMPT_VERSION", "extract_v1")

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    response = client.post("/raw-posts/1/extract")
    app.dependency_overrides.clear()

    assert response.status_code == 201
    body = response.json()
    assert body["extractor_name"] == "dummy"
    assert body["prompt_version"] == "extract_v1"
    assert isinstance(body["prompt_text"], str) and "Post Content:" in body["prompt_text"]
    assert isinstance(body["prompt_hash"], str) and len(body["prompt_hash"]) == 64
    assert isinstance(body["raw_model_output"], str) and "\"horizon\": \"1w\"" in body["raw_model_output"]


def test_asset_alias_upsert_normalizes_case_and_spaces() -> None:
    fake_db = FakeAsyncSession()
    now = datetime.now(UTC)
    fake_db.seed(Asset(id=1, symbol="XAUUSD", name="Gold", market="FOREX", created_at=now))
    fake_db.seed(Asset(id=2, symbol="BTC", name="Bitcoin", market="CRYPTO", created_at=now))

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    first = client.post("/assets/1/aliases", json={"alias": "  Gold  "})
    second = client.post("/assets/1/aliases", json={"alias": "gold"})
    third = client.post("/assets/2/aliases", json={"alias": " GOLD "})
    app.dependency_overrides.clear()

    assert first.status_code == 201
    assert second.status_code == 201
    assert third.status_code == 201
    assert len(first.json()) == 1
    assert len(second.json()) == 1
    assert len(third.json()) == 1
    assert third.json()[0]["asset_id"] == 2
    assert third.json()[0]["alias"] == "GOLD"


def test_prompt_includes_alias_map_from_db(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    now = datetime.now(UTC)
    fake_db.seed(Asset(id=1, symbol="XAUUSD", name="Gold", market="FOREX", created_at=now))
    fake_db.seed(AssetAlias(id=1, asset_id=1, alias="黄金", created_at=now))
    monkeypatch.setenv("EXTRACTOR_MODE", "dummy")

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post("/raw-posts/1/extract")
    app.dependency_overrides.clear()

    assert response.status_code == 201
    body = response.json()
    assert "Alias -> Symbol Map" in body["prompt_text"]
    assert "- 黄金 -> XAUUSD" in body["prompt_text"]


def test_normalize_corrects_symbol_by_alias_text_match() -> None:
    normalized = normalize_extracted_json(
        {
            "assets": [],
            "asset_views": [
                {
                    "symbol": "UNKNOWN",
                    "stance": "bull",
                    "horizon": "1w",
                    "confidence": 82,
                    "summary": "黄金走强，风险偏好下降。",
                    "reasoning": "避险资金回流黄金。",
                    "drivers": ["黄金突破关键阻力"],
                }
            ],
        },
        posted_at="2026-02-21T08:00:00+00:00",
        include_meta=True,
        alias_to_symbol={"黄金": "XAUUSD"},
        known_symbols={"BTC", "SPX", "XAUUSD"},
    )

    assert normalized["asset_views"][0]["symbol"] == "XAUUSD"
    assert normalized["meta"]["alias_corrections"] == [
        {"from": "UNKNOWN", "to": "XAUUSD", "reason": "alias_match"}
    ]


def test_manual_ingest_alias_corrections_hit_expected_symbols_and_auto_approve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = FakeAsyncSession()
    now = datetime.now(UTC)
    for asset_id, symbol, name, market in [
        (1, "XAUUSD", "Gold", "FOREX"),
        (2, "CL=F", "WTI Crude", "OTHER"),
        (3, "US10Y", "US 10Y Yield", "OTHER"),
        (4, "VIX", "CBOE VIX", "OTHER"),
        (5, "BTC", "Bitcoin", "CRYPTO"),
        (6, "SPX", "S&P 500", "INDEX"),
    ]:
        fake_db.seed(Asset(id=asset_id, symbol=symbol, name=name, market=market, created_at=now))
    for alias_id, asset_id, alias in [
        (1, 1, "黄金"),
        (2, 2, "原油"),
        (3, 3, "长债收益率"),
        (4, 4, "波动率"),
        (5, 4, "VIX"),
        (6, 5, "加密"),
        (7, 6, "美股"),
    ]:
        fake_db.seed(AssetAlias(id=alias_id, asset_id=asset_id, alias=alias, created_at=now))

    monkeypatch.setenv("EXTRACTOR_MODE", "dummy")
    monkeypatch.setenv("AUTO_APPROVE_ENABLED", "true")
    monkeypatch.setenv("AUTO_APPROVE_CONFIDENCE_THRESHOLD", "70")
    monkeypatch.setenv("AUTO_APPROVE_MIN_DISPLAY_CONFIDENCE", "50")

    def fake_dummy_extract(self, raw_post: RawPost):  # noqa: ANN001
        return {
            "source_url": raw_post.url,
            "as_of": "2026-02-21",
            "asset_views": [
                {"symbol": "黄金", "stance": "bull", "horizon": "1w", "confidence": 88, "summary": "黄金偏强"},
                {"symbol": "原油", "stance": "bull", "horizon": "1w", "confidence": 82, "summary": "原油受供给冲击"},
                {
                    "symbol": "US10YEAR",
                    "stance": "bull",
                    "horizon": "1w",
                    "confidence": 79,
                    "summary": "长债收益率继续上行",
                },
                {"symbol": "VIX", "stance": "bear", "horizon": "1w", "confidence": 76, "summary": "波动率回落"},
                {"symbol": "", "stance": "bull", "horizon": "1w", "confidence": 77, "summary": "加密资金回流"},
                {"symbol": "UNKNOWN", "stance": "neutral", "horizon": "1w", "confidence": 74, "summary": "美股震荡"},
            ],
        }

    monkeypatch.setattr("services.extraction.DummyExtractor.extract", fake_dummy_extract)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post(
        "/ingest/manual",
        json={
            "platform": "x",
            "author_handle": "alias_tester",
            "url": "https://x.com/alias_tester/status/1",
            "content_text": "黄金、原油、长债收益率、VIX、加密和美股都在波动。",
        },
    )
    app.dependency_overrides.clear()

    assert response.status_code == 201
    extraction = response.json()["extraction"]
    symbols = {item["symbol"] for item in extraction["extracted_json"]["asset_views"]}
    assert {"XAUUSD", "US10Y", "VIX", "BTC", "SPX"}.issubset(symbols)
    assert "CL=F" not in symbols
    assert extraction["auto_applied_count"] >= 5
    assert extraction["status"] == ExtractionStatus.approved.value


def test_normalize_derives_asset_views_from_global_payload() -> None:
    normalized = normalize_extracted_json(
        {
            "assets": ["BTC", "VIX"],
            "stance": "看空",
            "horizon": "今晚",
            "confidence": 0.82,
            "summary": "风险偏好回落",
            "reasoning": "避险情绪升温",
        },
        posted_at="2026-02-21T08:00:00+00:00",
        include_meta=True,
    )

    assert len(normalized["asset_views"]) == 2
    assert normalized["asset_views"][0]["stance"] == "bear"
    assert normalized["asset_views"][0]["horizon"] == "1w"
    assert normalized["asset_views"][0]["confidence"] == 82
    assert normalized["meta"]["derived_from_global"] is True


def test_normalize_assets_string_array_to_object_array_with_meta() -> None:
    normalized = normalize_extracted_json(
        {
            "assets": ["NVDA", "CRCL"],
            "content_kind": "asset",
        },
        posted_at="2026-02-21T08:00:00+00:00",
        include_meta=True,
    )

    assert normalized["assets"] == [{"symbol": "NoneAny", "name": None, "market": "OTHER"}]
    assert normalized["meta"]["assets_normalized_from_strings"] is True
    assert normalized["meta"]["assets_default_market"] == "AUTO"


def test_library_confidence_gate_downgrades() -> None:
    normalized = normalize_extracted_json(
        {
            "content_kind": "library",
            "assets": [{"symbol": "NoneAny", "market": "OTHER"}],
            "asset_views": [],
            "library_entry": {"confidence": 75, "tags": ["macro"], "summary": "too weak"},
        },
        posted_at="2026-02-21T08:00:00+00:00",
        include_meta=True,
    )

    assert normalized["content_kind"] == "asset"
    assert normalized["library_entry"] is None
    assert normalized["meta"]["library_downgraded"] is True
    assert normalized["meta"]["library_downgrade_reason"] == "low_library_confidence"
    assert normalized["meta"]["library_entry_drop_reason"] == "low_library_confidence"


def test_library_tags_gate_downgrades() -> None:
    normalized_empty_tags = normalize_extracted_json(
        {
            "content_kind": "library",
            "assets": [{"symbol": "NoneAny", "market": "OTHER"}],
            "asset_views": [],
            "library_entry": {"confidence": 88, "tags": [], "summary": "invalid tags"},
        },
        posted_at="2026-02-21T08:00:00+00:00",
        include_meta=True,
    )
    assert normalized_empty_tags["content_kind"] == "asset"
    assert normalized_empty_tags["library_entry"] is None
    assert normalized_empty_tags["meta"]["library_downgraded"] is True
    assert normalized_empty_tags["meta"]["library_downgrade_reason"] == "invalid_library_tags"

    normalized_unknown_tag = normalize_extracted_json(
        {
            "content_kind": "library",
            "assets": [{"symbol": "NoneAny", "market": "OTHER"}],
            "asset_views": [],
            "library_entry": {"confidence": 88, "tags": ["unknown"], "summary": "invalid tags"},
        },
        posted_at="2026-02-21T08:00:00+00:00",
        include_meta=True,
    )
    assert normalized_unknown_tag["content_kind"] == "asset"
    assert normalized_unknown_tag["library_entry"] is None
    assert normalized_unknown_tag["meta"]["library_downgraded"] is True
    assert normalized_unknown_tag["meta"]["library_downgrade_reason"] == "invalid_library_tags"


def test_asset_views_threshold_filters_and_syncs_assets() -> None:
    normalized = normalize_extracted_json(
        {
            "content_kind": "asset",
            "assets": [
                {"symbol": "BTC", "name": "Bitcoin", "market": "CRYPTO"},
                {"symbol": "ETH", "name": "Ethereum", "market": "CRYPTO"},
            ],
            "asset_views": [
                {"symbol": "BTC", "stance": "bull", "horizon": "1w", "confidence": 70, "summary": "keep"},
                {"symbol": "ETH", "stance": "bull", "horizon": "1w", "confidence": 40, "summary": "drop"},
            ],
        },
        posted_at="2026-02-21T08:00:00+00:00",
        include_meta=True,
    )

    assert len(normalized["asset_views"]) == 1
    assert normalized["asset_views"][0]["symbol"] == "BTC"
    assert all(item["confidence"] >= 70 for item in normalized["asset_views"])
    assert normalized["assets"] == [{"symbol": "BTC", "name": "Bitcoin", "market": "CRYPTO"}]

    normalized_empty = normalize_extracted_json(
        {
            "content_kind": "asset",
            "assets": [{"symbol": "ETH", "name": "Ethereum", "market": "CRYPTO"}],
            "asset_views": [
                {"symbol": "ETH", "stance": "bull", "horizon": "1w", "confidence": 69, "summary": "drop"}
            ],
        },
        posted_at="2026-02-21T08:00:00+00:00",
        include_meta=True,
    )
    assert normalized_empty["asset_views"] == []
    assert all(item["confidence"] >= 70 for item in normalized_empty["asset_views"])
    assert normalized_empty["assets"] == [{"symbol": "NoneAny", "name": None, "market": "OTHER"}]


def test_normalize_library_without_valid_library_entry_is_downgraded() -> None:
    normalized = normalize_extracted_json(
        {
            "content_kind": "library",
            "assets": [{"symbol": "NoneAny", "market": "OTHER"}],
            "asset_views": [{"symbol": "BTC", "stance": "bull", "horizon": "1w", "confidence": 88, "summary": "noise"}],
            "library_entry": {"confidence": 55, "tags": ["macro"], "summary": "too low"},
        },
        posted_at="2026-02-21T08:00:00+00:00",
        include_meta=True,
    )

    assert normalized["content_kind"] == "asset"
    assert normalized["library_entry"] is None
    assert normalized["meta"]["library_entry_dropped"] is True
    assert normalized["meta"]["library_entry_drop_reason"] == "low_library_confidence"
    assert normalized["meta"]["library_downgraded"] is True
    assert normalized["meta"]["library_downgrade_reason"] == "low_library_confidence"
    assert normalized["meta"]["content_kind_original"] == "library"


def test_normalize_keeps_chinese_symbol_and_drops_invalid_symbols_with_meta() -> None:
    normalized = normalize_extracted_json(
        {
            "assets": [
                {"symbol": "茅台", "market": "AUTO"},
                {"symbol": "BAD\nSYM", "market": "AUTO"},
                {"symbol": "{BTC}", "market": "AUTO"},
                {"symbol": "A" * 31, "market": "AUTO"},
            ],
            "asset_views": [
                {"symbol": "茅台", "stance": "bull", "horizon": "1w", "confidence": 80, "summary": "s"},
                {"symbol": "", "stance": "bull", "horizon": "1w", "confidence": 70, "summary": "drop"},
                {"symbol": "BAD\tSYM", "stance": "bull", "horizon": "1w", "confidence": 70, "summary": "drop2"},
            ],
            "content_kind": "asset",
        },
        posted_at="2026-02-21T08:00:00+00:00",
        include_meta=True,
    )

    assert normalized["assets"][0]["symbol"] == "茅台"
    assert any(item["symbol"] == "茅台" for item in normalized["asset_views"])
    assert normalized["meta"]["dropped_invalid_symbol_count"] >= 3
    assert normalized["meta"]["asset_views_dropped_empty_symbol_count"] >= 1


def test_normalize_drops_asset_view_with_empty_symbol() -> None:
    normalized = normalize_extracted_json(
        {
            "assets": [{"symbol": "BTC", "market": "CRYPTO"}],
            "asset_views": [
                {"symbol": "", "stance": "bull", "horizon": "1w", "confidence": 80, "summary": "drop"},
                {"symbol": "BTC", "stance": "bull", "horizon": "1w", "confidence": 80, "summary": "keep"},
            ],
            "content_kind": "asset",
        },
        posted_at="2026-02-21T08:00:00+00:00",
        include_meta=True,
    )

    assert len(normalized["asset_views"]) == 1
    assert normalized["asset_views"][0]["symbol"] == "BTC"
    assert normalized["meta"]["asset_views_dropped_empty_symbol_count"] == 1


def test_auto_approve_applies_threshold_views(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AUTO_APPROVE_ENABLED", "true")
    monkeypatch.setenv("AUTO_APPROVE_CONFIDENCE_THRESHOLD", "70")
    monkeypatch.setenv("AUTO_APPROVE_MAX_VIEWS", "10")

    def fake_extract(self, raw_post: RawPost):  # noqa: ANN001
        return {
            "assets": [{"symbol": "BTC", "market": "CRYPTO"}, {"symbol": "VIX", "market": "AUTO"}],
            "confidence": 82,
            "source_url": raw_post.url,
            "as_of": "2026-02-21",
            "asset_views": [
                {
                    "symbol": "BTC",
                    "stance": "bull",
                    "horizon": "1w",
                    "confidence": 82,
                    "reasoning": "risk-on momentum",
                    "summary": "BTC stronger",
                },
                {
                    "symbol": "VIX",
                    "stance": "bear",
                    "horizon": "1w",
                    "confidence": 77,
                    "reasoning": "volatility easing",
                    "summary": "VIX lower",
                },
                {
                    "symbol": "US10Y",
                    "stance": "bull",
                    "horizon": "1w",
                    "confidence": 60,
                    "reasoning": "rates sticky",
                    "summary": "yields up",
                },
                {
                    "symbol": "SPX",
                    "stance": "bear",
                    "horizon": "1w",
                    "confidence": 49,
                    "reasoning": "weak breadth",
                    "summary": "suspicious rally",
                },
            ],
        }

    monkeypatch.setattr("services.extraction.OpenAIExtractor.extract", fake_extract)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post("/raw-posts/1/extract")
    extraction_id = response.json()["id"] if response.status_code == 201 else 0
    detail = client.get(f"/extractions/{extraction_id}") if extraction_id else None
    app.dependency_overrides.clear()

    assert response.status_code == 201
    body = response.json()
    assert body["auto_applied_count"] == 2
    assert body["auto_policy"] == "threshold"
    assert body["status"] == ExtractionStatus.approved.value
    assert body["reviewed_by"] == "auto"
    assert isinstance(body["auto_applied_kol_view_ids"], list)
    assert len(body["auto_applied_kol_view_ids"]) == 2
    assert len(fake_db._data[KolView]) == 2
    assert all(view.confidence >= 70 for view in fake_db._data[KolView].values())
    assert detail is not None and detail.status_code == 200
    detail_body = detail.json()
    assert isinstance(detail_body["auto_applied_asset_view_keys"], list)
    assert len(detail_body["auto_applied_asset_view_keys"]) == 2
    assert isinstance(detail_body["auto_applied_views"], list)
    assert len(detail_body["auto_applied_views"]) == 2


def test_auto_approve_repeat_trigger_skips_same_key(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AUTO_APPROVE_ENABLED", "true")
    monkeypatch.setenv("AUTO_APPROVE_CONFIDENCE_THRESHOLD", "70")
    monkeypatch.setenv("AUTO_APPROVE_MAX_VIEWS", "10")

    call = {"n": 0}

    def fake_extract(self, raw_post: RawPost):  # noqa: ANN001
        call["n"] += 1
        return {
            "assets": [{"symbol": "BTC", "market": "CRYPTO"}],
            "confidence": 82,
            "source_url": raw_post.url,
            "as_of": "2026-02-21",
            "asset_views": [
                {
                    "symbol": "BTC",
                    "stance": "bull",
                    "horizon": "1w",
                    "confidence": 82,
                    "reasoning": "risk-on",
                    "summary": f"BTC view v{call['n']}",
                }
            ],
        }

    monkeypatch.setattr("services.extraction.OpenAIExtractor.extract", fake_extract)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    first = client.post("/raw-posts/1/extract")
    second = client.post("/raw-posts/1/extract")
    app.dependency_overrides.clear()

    assert first.status_code == 201
    first_body = first.json()
    assert first_body["status"] == ExtractionStatus.approved.value
    assert second.status_code == 201
    second_body = second.json()
    assert second_body["id"] == first_body["id"]
    assert second_body["auto_applied_count"] == first_body["auto_applied_count"]
    assert second_body["status"] == ExtractionStatus.approved.value
    assert len(fake_db._data[KolView]) == 1
    only_view = list(fake_db._data[KolView].values())[0]
    assert only_view.summary == "BTC view v1"


def test_approve_batch_on_auto_applied_same_key_does_not_duplicate(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AUTO_APPROVE_ENABLED", "true")
    monkeypatch.setenv("AUTO_APPROVE_CONFIDENCE_THRESHOLD", "70")
    monkeypatch.setenv("AUTO_APPROVE_MAX_VIEWS", "10")

    def fake_extract(self, raw_post: RawPost):  # noqa: ANN001
        return {
            "assets": [{"symbol": "BTC", "market": "CRYPTO"}],
            "confidence": 82,
            "source_url": raw_post.url,
            "as_of": "2026-02-21",
            "asset_views": [
                {
                    "symbol": "BTC",
                    "stance": "bull",
                    "horizon": "1w",
                    "confidence": 82,
                    "reasoning": "risk-on",
                    "summary": "auto summary",
                }
            ],
        }

    monkeypatch.setattr("services.extraction.OpenAIExtractor.extract", fake_extract)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    extract_resp = client.post("/raw-posts/1/extract")
    assert extract_resp.status_code == 201
    extraction = extract_resp.json()
    assert extraction["status"] == ExtractionStatus.approved.value

    pending_extraction = PostExtraction(
        id=2,
        raw_post_id=1,
        status=ExtractionStatus.pending,
        extracted_json={"summary": "pending review"},
        model_name="dummy-v1",
        extractor_name="dummy",
        created_at=datetime.now(UTC),
    )
    fake_db.seed(pending_extraction)

    approve_resp = client.post(
        "/extractions/2/approve-batch",
        json={
            "kol_id": 1,
            "views": [
                {
                    "asset_id": 1,
                    "stance": "bull",
                    "horizon": "1w",
                    "confidence": 91,
                    "summary": "manual summary",
                    "source_url": "https://x.com/alice/status/post-1",
                    "as_of": "2026-02-21",
                }
            ],
        },
    )
    app.dependency_overrides.clear()

    assert approve_resp.status_code == 200
    approve_body = approve_resp.json()
    assert approve_body["status"] == ExtractionStatus.approved.value
    assert approve_body["approve_inserted_count"] == 0
    assert approve_body["approve_skipped_count"] == 1
    assert len(fake_db._data[KolView]) == 1
    only_view = list(fake_db._data[KolView].values())[0]
    assert only_view.confidence == 82
    assert only_view.summary == "auto summary"


def test_confidence_below_70_rejects_even_if_asset_view_candidates_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AUTO_APPROVE_ENABLED", "true")
    monkeypatch.setenv("AUTO_APPROVE_CONFIDENCE_THRESHOLD", "70")
    monkeypatch.setenv("AUTO_APPROVE_MIN_DISPLAY_CONFIDENCE", "50")
    monkeypatch.setenv("AUTO_APPROVE_MAX_VIEWS", "10")

    def fake_extract(self, raw_post: RawPost):  # noqa: ANN001
        return {
            "assets": [{"symbol": "BTC", "market": "CRYPTO"}, {"symbol": "VIX", "market": "AUTO"}],
            "confidence": 68,
            "source_url": raw_post.url,
            "as_of": "2026-02-21",
            "asset_views": [
                {"symbol": "BTC", "stance": "bull", "horizon": "1w", "confidence": 68, "summary": "btc"},
                {"symbol": "VIX", "stance": "bear", "horizon": "1w", "confidence": 55, "summary": "vix"},
                {"symbol": "US10Y", "stance": "bull", "horizon": "1w", "confidence": 49, "summary": "us10y"},
            ],
        }

    monkeypatch.setattr("services.extraction.OpenAIExtractor.extract", fake_extract)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post("/raw-posts/1/extract")
    app.dependency_overrides.clear()

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == ExtractionStatus.rejected.value
    assert body["auto_policy"] is None
    assert body["auto_applied_count"] is None
    assert body["extracted_json"]["meta"]["auto_rejected"] is True
    assert len(fake_db._data[KolView]) == 0
