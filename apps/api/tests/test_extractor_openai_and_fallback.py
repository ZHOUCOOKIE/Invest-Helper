from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import inspect
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from pydantic import ValidationError

from enums import ExtractionStatus
from main import _detect_extracted_summary_language, postprocess_auto_review
from models import Asset, AssetAlias, Kol, KolView, PostExtraction, RawPost
from services import extraction as extraction_service
from services.extraction import normalize_extracted_json
from services.prompts.extraction_prompt import render_prompt_bundle


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


def _raw_post() -> RawPost:
    return RawPost(
        id=1,
        platform="x",
        author_handle="alice",
        external_id="post-1",
        url="https://x.com/alice/status/post-1",
        content_text="NVDA and BTC update",
        posted_at=datetime(2026, 2, 21, 9, 0, tzinfo=UTC),
        fetched_at=datetime.now(UTC),
        raw_json=None,
    )


def _extraction(payload: dict) -> PostExtraction:
    return PostExtraction(
        id=1,
        raw_post_id=1,
        status=ExtractionStatus.pending,
        extracted_json=payload,
        model_name="test",
        extractor_name="openai_structured",
        created_at=datetime.now(UTC),
    )


def test_prompt_bundle_has_new_contract_phrases() -> None:
    bundle = render_prompt_bundle(
        platform="x",
        author_handle="alice",
        url="https://x.com/alice/status/post-1",
        posted_at=datetime(2026, 2, 21, 9, 0, tzinfo=UTC),
        content_text="NVDA BTC",
    )
    prompt_text = bundle.text
    assert "Top-level fields MUST be exactly" in prompt_text
    assert "as_of, source_url, islibrary, hasview, asset_views, library_entry" in prompt_text
    assert "Now output the final JSON object only." in prompt_text
    lowered = prompt_text.lower()
    assert ("content" + "_kind") not in lowered
    assert ("never output " + "name") not in lowered
    assert ("if islibrary=1, " + "assets must be") not in lowered
    assert ("do not output " + "drivers/reasoning") not in lowered


def test_render_prompt_bundle_no_unreplaced_placeholders_and_url_once() -> None:
    url = "https://x.com/alice/status/post-1"
    bundle = render_prompt_bundle(
        platform="x",
        author_handle="alice",
        url=url,
        posted_at=datetime(2026, 2, 21, 9, 0, tzinfo=UTC),
        content_text="NVDA and BTC update",
    )
    assert "{platform}" not in bundle.text
    assert "{author_handle}" not in bundle.text
    assert "{url}" not in bundle.text
    assert "{posted_at}" not in bundle.text
    assert "{content_text}" not in bundle.text
    assert bundle.text.count(url) == 1
    assert f"url: {url}" in bundle.text


def test_render_prompt_bundle_has_no_lang_field() -> None:
    bundle = render_prompt_bundle(
        platform="x",
        author_handle="alice",
        url="https://x.com/alice/status/post-1",
        posted_at=datetime(2026, 2, 21, 9, 0, tzinfo=UTC),
        content_text="NVDA BTC",
    )
    assert "lang:" not in bundle.text
    assert "{lang}" not in bundle.text


def test_openai_request_sends_single_user_message(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    payload = {
        "as_of": "2026-02-21",
        "source_url": "https://x.com/alice/status/post-1",
        "islibrary": 0,
        "hasview": 0,
        "asset_views": [],
        "library_entry": None,
    }

    class FakeResponse:
        status_code = 200
        text = ""
        headers: dict[str, str] = {}

        def json(self) -> dict[str, object]:
            return {
                "choices": [{"message": {"content": json.dumps(payload)}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10},
            }

    class FakeClient:
        def __init__(self, *, timeout: float) -> None:
            del timeout

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
            del exc_type, exc, tb
            return None

        def post(self, url: str, *, headers: dict[str, str], json: dict[str, object]) -> FakeResponse:
            captured["url"] = url
            captured["headers"] = headers
            captured["request_payload"] = json
            return FakeResponse()

    monkeypatch.setattr(extraction_service.httpx, "Client", FakeClient)

    extractor = extraction_service.OpenAIExtractor(
        api_key="k",
        model_name="gpt-test",
        base_url="https://api.openai.com/v1",
        timeout_seconds=3.0,
        max_output_tokens=256,
    )
    extractor._call_openai(_raw_post(), response_mode=extraction_service.EXTRACTION_OUTPUT_STRUCTURED)

    request_payload = captured["request_payload"]
    assert isinstance(request_payload, dict)
    messages = request_payload.get("messages")
    assert isinstance(messages, list)
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert "source_url" in str(messages[0]["content"])


def test_extraction_service_has_no_alias_map() -> None:
    source = inspect.getsource(extraction_service)
    alias_marker = "_ALIAS" + "_MAP"
    assert alias_marker not in source


def test_asset_view_market_enum_rejects_auto() -> None:
    removed_market_value = "AU" + "TO"
    with pytest.raises(ValidationError):
        extraction_service.AssetView.model_validate(
            {
                "symbol": "BTC",
                "market": removed_market_value,
                "stance": "bull",
                "horizon": "1w",
                "confidence": 80,
                "summary": "看多",
            }
        )


def test_normalize_drops_extra_top_level_fields() -> None:
    normalized = normalize_extracted_json(
        {
            "as_of": "2026-02-21",
            "source_url": "https://x.com/alice/status/post-1",
            "islibrary": 0,
            "hasview": 1,
            "asset_views": [
                {
                    "symbol": "BTC",
                    "market": "CRYPTO",
                    "stance": "bull",
                    "horizon": "1w",
                    "confidence": 80,
                    "summary": "看多",
                }
            ],
            "library_entry": None,
            "summary": "legacy",
            "foo": "bar",
        },
        posted_at="2026-02-21T08:00:00+00:00",
        include_meta=False,
    )
    assert "summary" not in normalized
    assert "foo" not in normalized


def test_normalize_drops_asset_view_below_70() -> None:
    normalized = normalize_extracted_json(
        {
            "as_of": "2026-02-21",
            "source_url": "https://x.com/alice/status/post-1",
            "islibrary": 0,
            "hasview": 1,
            "asset_views": [
                {"symbol": "BTC", "market": "CRYPTO", "stance": "bull", "horizon": "1w", "confidence": 80, "summary": "看多"},
                {"symbol": "ETH", "market": "CRYPTO", "stance": "bull", "horizon": "1w", "confidence": 60, "summary": "看多"},
            ],
            "library_entry": None,
        },
        posted_at="2026-02-21T08:00:00+00:00",
        include_meta=True,
    )
    assert len(normalized["asset_views"]) == 1
    assert normalized["asset_views"][0]["symbol"] == "BTC"
    assert normalized["meta"]["asset_views_dropped_low_confidence_count"] == 1


def test_normalize_empty_assets_filled_noneany() -> None:
    normalized = normalize_extracted_json(
        {
            "as_of": "2026-02-21",
            "source_url": "https://x.com/alice/status/post-1",
            "islibrary": 0,
            "hasview": 1,
            "asset_views": [],
            "library_entry": None,
        },
        posted_at="2026-02-21T08:00:00+00:00",
        include_meta=True,
    )
    assert normalized["hasview"] == 0
    assert "assets" not in normalized


def test_normalize_islibrary_zero_forces_library_entry_null() -> None:
    normalized = normalize_extracted_json(
        {
            "as_of": "2026-02-21",
            "source_url": "https://x.com/alice/status/post-1",
            "islibrary": 0,
            "hasview": 1,
            "asset_views": [
                {"symbol": "BTC", "market": "CRYPTO", "stance": "bull", "horizon": "1w", "confidence": 80, "summary": "看多"}
            ],
            "library_entry": {"tag": "macro", "summary": "中文"},
        },
        posted_at="2026-02-21T08:00:00+00:00",
        include_meta=True,
    )
    assert normalized["islibrary"] == 0
    assert normalized["library_entry"] is None


def test_normalize_islibrary_one_invalid_tag_downgrades() -> None:
    normalized = normalize_extracted_json(
        {
            "as_of": "2026-02-21",
            "source_url": "https://x.com/alice/status/post-1",
            "islibrary": 1,
            "hasview": 0,
            "asset_views": [],
            "library_entry": {"tag": "bad", "summary": "中文"},
        },
        posted_at="2026-02-21T08:00:00+00:00",
        include_meta=True,
    )
    assert normalized["islibrary"] == 0
    assert normalized["library_entry"] is None
    assert normalized["meta"]["library_downgraded"] is True


def test_normalize_islibrary_one_missing_tag_downgrades() -> None:
    normalized = normalize_extracted_json(
        {
            "as_of": "2026-02-21",
            "source_url": "https://x.com/alice/status/post-1",
            "islibrary": 1,
            "hasview": 0,
            "asset_views": [],
            "library_entry": {"summary": "中文"},
        },
        posted_at="2026-02-21T08:00:00+00:00",
        include_meta=True,
    )
    assert normalized["islibrary"] == 0
    assert normalized["library_entry"] is None
    assert normalized["meta"]["library_downgraded"] is True


def test_normalize_library_entry_keeps_tag_shape() -> None:
    normalized = normalize_extracted_json(
        {
            "as_of": "2026-02-21",
            "source_url": "https://x.com/alice/status/post-1",
            "islibrary": 1,
            "hasview": 0,
            "asset_views": [],
            "library_entry": {"tag": "macro", "summary": "测试"},
        },
        posted_at="2026-02-21T08:00:00+00:00",
        include_meta=True,
    )
    assert normalized["islibrary"] == 1
    assert normalized["library_entry"]["tag"] == "macro"


def test_summary_language_check_only_asset_views_and_library_entry() -> None:
    payload = {
        "summary": "This is English but should be ignored",
        "asset_views": [
            {
                "symbol": "BTC",
                "market": "CRYPTO",
                "stance": "bull",
                "horizon": "1w",
                "confidence": 80,
                "summary": "看多",
            }
        ],
        "library_entry": None,
    }
    assert _detect_extracted_summary_language(payload) == "zh"


def test_library_without_hasview_auto_rejected() -> None:
    raw_post = _raw_post()
    extraction = _extraction(
        {
            "as_of": "2026-02-21",
            "source_url": raw_post.url,
            "islibrary": 1,
            "hasview": 0,
            "asset_views": [],
            "library_entry": {"tag": "macro", "summary": "测试"},
        }
    )

    outcome = asyncio.run(
        postprocess_auto_review(
            db=FakeAsyncSession(),
            extraction=extraction,
            raw_post=raw_post,
            trigger="auto",
        )
    )
    assert outcome == "rejected"
    assert extraction.status == ExtractionStatus.rejected


def test_non_library_without_asset_views_auto_rejected() -> None:
    raw_post = _raw_post()
    extraction = _extraction(
        {
            "as_of": "2026-02-21",
            "source_url": raw_post.url,
            "islibrary": 0,
            "hasview": 0,
            "asset_views": [],
            "library_entry": None,
        }
    )

    outcome = asyncio.run(
        postprocess_auto_review(
            db=FakeAsyncSession(),
            extraction=extraction,
            raw_post=raw_post,
            trigger="auto",
        )
    )

    assert outcome == "rejected"
    assert extraction.status == ExtractionStatus.rejected
    meta = extraction.extracted_json.get("meta")
    assert isinstance(meta, dict)
    assert meta.get("auto_rejected") is True
    assert meta.get("auto_review_reason") == "hasview_zero"


def test_non_library_hasview_zero_auto_rejected_even_if_asset_views_present() -> None:
    raw_post = _raw_post()
    extraction = _extraction(
        {
            "as_of": "2026-02-21",
            "source_url": raw_post.url,
            "islibrary": 0,
            "hasview": 0,
            "asset_views": [
                {
                    "symbol": "BTC",
                    "market": "CRYPTO",
                    "stance": "bull",
                    "horizon": "1w",
                    "confidence": 95,
                    "summary": "看多",
                }
            ],
            "library_entry": None,
        }
    )

    outcome = asyncio.run(
        postprocess_auto_review(
            db=FakeAsyncSession(),
            extraction=extraction,
            raw_post=raw_post,
            trigger="auto",
        )
    )

    assert outcome == "rejected"
    assert extraction.status == ExtractionStatus.rejected
    meta = extraction.extracted_json.get("meta")
    assert isinstance(meta, dict)
    assert meta.get("auto_review_reason") == "hasview_zero"


def test_hasview_one_with_asset_views_and_confidence_ge_80_auto_approved() -> None:
    raw_post = _raw_post()
    extraction = _extraction(
        {
            "as_of": "2026-02-21",
            "source_url": raw_post.url,
            "islibrary": 0,
            "hasview": 1,
            "asset_views": [
                {
                    "symbol": "BTC",
                    "market": "CRYPTO",
                    "stance": "bull",
                    "horizon": "1w",
                    "confidence": 80,
                    "summary": "看多",
                }
            ],
            "library_entry": None,
        }
    )

    outcome = asyncio.run(
        postprocess_auto_review(
            db=FakeAsyncSession(),
            extraction=extraction,
            raw_post=raw_post,
            trigger="auto",
        )
    )

    assert outcome == "approved"
    assert extraction.status == ExtractionStatus.approved


def test_islibrary_one_hasview_zero_with_asset_views_is_not_auto_approved() -> None:
    raw_post = _raw_post()
    extraction = _extraction(
        {
            "as_of": "2026-02-21",
            "source_url": raw_post.url,
            "islibrary": 1,
            "hasview": 0,
            "asset_views": [
                {
                    "symbol": "BTC",
                    "market": "CRYPTO",
                    "stance": "bull",
                    "horizon": "1w",
                    "confidence": 80,
                    "summary": "看多",
                }
            ],
            "library_entry": {"tag": "thesis", "summary": "测试"},
        }
    )

    outcome = asyncio.run(
        postprocess_auto_review(
            db=FakeAsyncSession(),
            extraction=extraction,
            raw_post=raw_post,
            trigger="auto",
        )
    )

    assert outcome == "rejected"
    assert extraction.status == ExtractionStatus.rejected


def test_islibrary_one_user_trigger_is_not_auto_reviewed() -> None:
    raw_post = _raw_post()
    extraction = _extraction(
        {
            "as_of": "2026-02-21",
            "source_url": raw_post.url,
            "islibrary": 1,
            "hasview": 0,
            "asset_views": [],
            "library_entry": {"tag": "macro", "summary": "测试"},
        }
    )

    outcome = asyncio.run(
        postprocess_auto_review(
            db=FakeAsyncSession(),
            extraction=extraction,
            raw_post=raw_post,
            trigger="user",
        )
    )

    assert outcome is None
    assert extraction.status == ExtractionStatus.pending
