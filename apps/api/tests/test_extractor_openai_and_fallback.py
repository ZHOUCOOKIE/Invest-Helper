from __future__ import annotations

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
from main import app, reset_runtime_counters
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
            "summary": "If support holds, short-term upside remains.",
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


def test_prompt_contract_keeps_reasoning_zh_and_symbol_non_empty_rules(monkeypatch: pytest.MonkeyPatch) -> None:
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
    assert "extracted_json.reasoning must be Chinese" in prompt_text
    assert "sentence body must be Chinese" in prompt_text
    assert "asset_views[].symbol must be non-empty" in prompt_text


def test_reasoning_non_zh_retries_once_and_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
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
                "summary": "first pass english reasoning",
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
            "summary": "retry chinese reasoning",
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
    assert meta["reasoning_language"] == "zh"
    assert meta["reasoning_language_violation"] is False
    assert meta["reasoning_language_retry_used"] is True


def test_reasoning_non_zh_twice_marks_failed(monkeypatch: pytest.MonkeyPatch) -> None:
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
            "summary": "still english reasoning",
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
    assert "reasoning_language_violation_after_retry" in (body["last_error"] or "")
    assert meta["reasoning_language"] == "non_zh"
    assert meta["reasoning_language_violation"] is True
    assert meta["reasoning_language_retry_used"] is True


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
            "summary": "If support holds, short-term upside remains.",
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
    model_name = "minimax/minimax-m2.5"
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
    assert body["extracted_json"]["meta"]["auto_review_reason"] == "confidence_below_threshold"
    assert body["extracted_json"]["meta"]["auto_review_threshold"] == 70
    assert body["extracted_json"]["meta"]["model_confidence"] == 69
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
    assert body["status"] == ExtractionStatus.pending.value
    assert body["reviewed_by"] is None
    assert body["extracted_json"]["meta"].get("auto_approved") is not True
    assert body["extracted_json"]["meta"].get("auto_rejected") is not True
    assert fake_db._data[RawPost][1].review_status in {None, ReviewStatus.unreviewed}


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
        model_name="minimax/minimax-m2.5",
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
        model_name="minimax/minimax-m2.5",
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

    assert extracted["assets"] == ["NoneAny"]
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
    extractor.set_reasoning_language_retry_hint(True)
    extractor._call_openai(raw_post, response_mode="text_json")

    assert len(captured_payloads) == 2

    structured_payload = captured_payloads[0]
    assert structured_payload["messages"][1]["content"] == "PROMPT_BUNDLE_USER_TEXT"
    assert "response_format" in structured_payload
    assert "json_schema" in structured_payload["response_format"]
    structured_system = structured_payload["messages"][0]["content"]
    assert "extracted_json.reasoning must be written in Chinese." in structured_system
    assert "Return exactly one JSON object." in structured_system
    assert "Correction retry: previous output had non-Chinese reasoning." not in structured_system
    assert "JSON mode hard rules (中英都适用):" not in structured_system

    text_json_payload = captured_payloads[1]
    assert text_json_payload["messages"][1]["content"] == "PROMPT_BUNDLE_USER_TEXT"
    assert "response_format" not in text_json_payload
    text_json_system = text_json_payload["messages"][0]["content"]
    assert "Correction retry: previous output had non-Chinese reasoning." in text_json_system
    assert "JSON mode hard rules (中英都适用):" in text_json_system


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
    assert "previous output was truncated JSON" in captured_payloads[1]["messages"][0]["content"]
    assert "extracted_json.reasoning must be written in Chinese." in captured_payloads[1]["messages"][0]["content"]
    assert meta["output_mode_used"] == "text_json"
    assert meta["truncated_retry_used"] is True
    assert bool(meta.get("parse_error")) is False
    assert meta["raw_len"] > 0
    assert meta["raw_saved_len"] > 0
    assert meta["extra_retry_budget_total"] == 1
    assert meta["extra_retry_budget_used"] == 1
    assert body["raw_model_output"] is not None and len(body["raw_model_output"]) > 0


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
                                "summary": "english reasoning",
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
    assert "reasoning_language_violation_no_retry_budget_after_truncated_retry" in (body["last_error"] or "")
    assert meta["truncated_retry_used"] is True
    assert meta["reasoning_language"] == "non_zh"
    assert meta["reasoning_language_violation"] is True
    assert meta["reasoning_language_retry_used"] is False
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
    assert extracted["horizon"] == "intraday"
    assert extracted["stance"] == "neutral"
    assert extracted["confidence"] == 78
    assert extracted["event_tags"] == []
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
    assert body["extracted_json"]["assets"][0]["market"] == "AUTO"


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
    assert {"XAUUSD", "CL=F", "US10Y", "VIX", "BTC", "SPX"}.issubset(symbols)
    assert extraction["auto_applied_count"] >= 6
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
    assert normalized["asset_views"][0]["horizon"] == "intraday"
    assert normalized["asset_views"][0]["confidence"] == 82
    assert normalized["meta"]["derived_from_global"] is True


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
