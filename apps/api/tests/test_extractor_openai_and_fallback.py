from __future__ import annotations

from datetime import UTC, datetime, timedelta
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
    extraction_id = response.json()["id"] if response.status_code == 201 else 0
    detail = client.get(f"/extractions/{extraction_id}") if extraction_id else None
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
    monkeypatch.setenv("EXTRACTOR_MODE", "auto")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENAI_MODEL", "qwen/qwen-2.5-72b-instruct")
    monkeypatch.setenv("OPENAI_CALL_BUDGET", "3")
    monkeypatch.setenv("OPENAI_MAX_OUTPUT_TOKENS", "888")

    client = TestClient(app)
    response = client.get("/extractor-status")

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "auto"
    assert body["has_api_key"] is True
    assert body["default_model"] == "qwen/qwen-2.5-72b-instruct"
    assert body["base_url"] == "https://openrouter.ai/api/v1"
    assert body["call_budget_remaining"] == 3
    assert body["max_output_tokens"] == 888


def test_runtime_settings_budget_get_and_post(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXTRACTOR_MODE", "auto")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_CALL_BUDGET", "5")
    monkeypatch.setenv("OPENAI_MODEL", "qwen/qwen-2.5-72b-instruct")

    client = TestClient(app)

    initial = client.get("/settings/runtime")
    assert initial.status_code == 200
    initial_body = initial.json()
    assert initial_body["extractor_mode"] == "auto"
    assert initial_body["model"] == "qwen/qwen-2.5-72b-instruct"
    assert initial_body["has_api_key"] is True
    assert initial_body["budget_total"] == 5
    assert initial_body["budget_remaining"] == 5
    assert initial_body["auto_reject_confidence_threshold"] == 50
    assert "runtime_overrides" in initial_body

    updated = client.post("/settings/runtime/call-budget", json={"call_budget": 12})
    assert updated.status_code == 200
    updated_body = updated.json()
    assert updated_body["budget_total"] == 12
    assert updated_body["budget_remaining"] == 12
    assert updated_body["runtime_overrides"]["call_budget"] is True


def test_runtime_default_budget_is_30_per_hour(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_CALL_BUDGET", raising=False)
    monkeypatch.setenv("CALL_BUDGET_PER_HOUR", "30")

    client = TestClient(app)
    response = client.get("/settings/runtime")
    assert response.status_code == 200
    body = response.json()
    assert body["default_budget_total"] == 30
    assert body["budget_total"] == 30


def test_runtime_budget_resets_on_new_window(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_CALL_BUDGET", raising=False)
    monkeypatch.setenv("CALL_BUDGET_PER_HOUR", "2")
    monkeypatch.setenv("CALL_BUDGET_WINDOW_MINUTES", "60")
    now_ref = {"value": datetime(2026, 2, 23, 10, 0, tzinfo=UTC)}

    def fake_now():
        return now_ref["value"]

    def fake_extract(self, raw_post: RawPost):  # noqa: ANN001
        return {
            "assets": [{"symbol": "BTC", "name": "Bitcoin", "market": "CRYPTO"}],
            "stance": "bull",
            "horizon": "1w",
            "confidence": 78,
            "summary": "window test",
            "source_url": raw_post.url,
            "as_of": "2026-02-23",
            "event_tags": [],
        }

    monkeypatch.setattr("main._utc_now", fake_now)
    monkeypatch.setattr("services.extraction.OpenAIExtractor.extract", fake_extract)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    initial = client.get("/settings/runtime")
    assert initial.status_code == 200
    assert initial.json()["budget_remaining"] == 2

    extracted = client.post("/raw-posts/1/extract")
    assert extracted.status_code == 201
    after_extract = client.get("/settings/runtime")
    assert after_extract.status_code == 200
    assert after_extract.json()["budget_remaining"] == 1

    now_ref["value"] = now_ref["value"] + timedelta(minutes=61)
    after_window = client.get("/settings/runtime")
    app.dependency_overrides.clear()

    assert after_window.status_code == 200
    body = after_window.json()
    assert body["budget_total"] == 2
    assert body["budget_remaining"] == 2


def test_runtime_burst_enable_and_expire_restores_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_CALL_BUDGET", raising=False)
    monkeypatch.setenv("CALL_BUDGET_PER_HOUR", "5")
    now_ref = {"value": datetime(2026, 2, 23, 12, 0, tzinfo=UTC)}

    def fake_now():
        return now_ref["value"]

    monkeypatch.setattr("main._utc_now", fake_now)
    client = TestClient(app)

    burst_on = client.post(
        "/settings/runtime/burst",
        json={"enabled": True, "call_budget": 500, "duration_minutes": 10},
    )
    assert burst_on.status_code == 200
    on_body = burst_on.json()
    assert on_body["burst"]["enabled"] is True
    assert on_body["budget_total"] == 500

    now_ref["value"] = now_ref["value"] + timedelta(minutes=11)
    after_expire = client.get("/settings/runtime")
    assert after_expire.status_code == 200
    expired_body = after_expire.json()
    assert expired_body["burst"]["enabled"] is False
    assert expired_body["budget_total"] == 5

    burst_off = client.post(
        "/settings/runtime/burst",
        json={"enabled": False, "call_budget": 0, "duration_minutes": 1},
    )
    assert burst_off.status_code == 200
    assert burst_off.json()["burst"]["enabled"] is False


def test_runtime_burst_unlimited_safe_sets_large_budget_and_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_CALL_BUDGET", raising=False)
    monkeypatch.setenv("CALL_BUDGET_PER_HOUR", "30")
    client = TestClient(app)

    response = client.post(
        "/settings/runtime/burst",
        json={"enabled": True, "mode": "unlimited_safe", "call_budget": 1, "duration_minutes": 30},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["budget_total"] == 100000
    assert body["burst"]["enabled"] is True
    assert body["burst"]["mode"] == "unlimited_safe"


def test_runtime_throttle_update_clamps_to_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXTRACT_MAX_CONCURRENCY_MAX", "4")
    monkeypatch.setenv("EXTRACT_MAX_RPM_MAX", "60")
    monkeypatch.setenv("EXTRACT_BATCH_SIZE_MAX", "50")
    monkeypatch.setenv("EXTRACT_BATCH_SLEEP_MS_MIN", "100")

    client = TestClient(app)
    response = client.post(
        "/settings/runtime/throttle",
        json={
            "max_concurrency": 999,
            "max_rpm": 9999,
            "batch_size": 999,
            "batch_sleep_ms": 1,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["throttle"]["max_concurrency"] == 4
    assert body["throttle"]["max_rpm"] == 60
    assert body["throttle"]["batch_size"] == 50
    assert body["throttle"]["batch_sleep_ms"] == 100


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


def test_runtime_call_budget_override_visibility_and_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_CALL_BUDGET", raising=False)
    monkeypatch.setenv("CALL_BUDGET_PER_HOUR", "50")

    client = TestClient(app)
    set_override = client.post("/settings/runtime/call-budget", json={"call_budget": 3})
    assert set_override.status_code == 200
    set_body = set_override.json()
    assert set_body["budget_total"] == 3
    assert set_body["default_budget_total"] == 50
    assert set_body["call_budget_override_enabled"] is True
    assert set_body["call_budget_override_value"] == 3

    clear_resp = client.post("/settings/runtime/call-budget/clear")
    assert clear_resp.status_code == 200
    cleared = clear_resp.json()
    assert cleared["call_budget_override_enabled"] is False
    assert cleared["call_budget_override_value"] is None
    assert cleared["default_budget_total"] == 50
    assert cleared["budget_total"] == 50


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
        model_name="qwen/qwen-2.5-72b-instruct",
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
    assert extracted["meta"]["output_mode_used"] == "text_json"
    assert len(captured_payloads) == 1
    assert "response_format" not in captured_payloads[0]


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
