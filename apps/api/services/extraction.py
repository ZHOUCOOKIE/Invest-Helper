from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
import json
from threading import Lock
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

from models import RawPost
from settings import Settings

MARKET_VALUES = ["CRYPTO", "STOCK", "ETF", "FOREX", "OTHER"]
STANCE_VALUES = ["bull", "bear", "neutral"]
HORIZON_VALUES = ["intraday", "1w", "1m", "3m", "1y"]
_OPENAI_CALLS_MADE = 0
_OPENAI_CALLS_LOCK = Lock()


class ExtractedAsset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    name: str | None = None
    market: Literal["CRYPTO", "STOCK", "ETF", "FOREX", "OTHER"] | None = None


class ExtractionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assets: list[ExtractedAsset] = Field(default_factory=list)
    stance: Literal["bull", "bear", "neutral"] | None = None
    horizon: Literal["intraday", "1w", "1m", "3m", "1y"] | None = None
    confidence: int | None = Field(default=None, ge=0, le=100)
    summary: str | None = None
    source_url: str | None = None
    as_of: date | None = None
    event_tags: list[str] = Field(default_factory=list)


EXTRACTION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "assets",
        "stance",
        "horizon",
        "confidence",
        "summary",
        "source_url",
        "as_of",
        "event_tags",
    ],
    "properties": {
        "assets": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["symbol", "name", "market"],
                "properties": {
                    "symbol": {"type": "string", "minLength": 1, "maxLength": 32},
                    "name": {"type": ["string", "null"], "maxLength": 255},
                    "market": {"type": ["string", "null"], "enum": MARKET_VALUES + [None]},
                },
            },
        },
        "stance": {"type": ["string", "null"], "enum": STANCE_VALUES + [None]},
        "horizon": {"type": ["string", "null"], "enum": HORIZON_VALUES + [None]},
        "confidence": {"type": ["integer", "null"], "minimum": 0, "maximum": 100},
        "summary": {"type": ["string", "null"], "maxLength": 1024},
        "source_url": {"type": ["string", "null"], "maxLength": 1024},
        "as_of": {"type": ["string", "null"], "format": "date"},
        "event_tags": {
            "type": "array",
            "items": {"type": "string", "minLength": 1, "maxLength": 64},
        },
    },
}


class Extractor(ABC):
    model_name: str
    extractor_name: str

    @abstractmethod
    def extract(self, raw_post: RawPost) -> dict[str, Any]:
        """Return structured candidates inferred from a raw post."""


class DummyExtractor(Extractor):
    model_name = "dummy-v2"
    extractor_name = "dummy"

    def extract(self, raw_post: RawPost) -> dict[str, Any]:
        content = raw_post.content_text.strip()
        summary = content[:280] if content else None

        payload = ExtractionPayload(
            assets=[],
            stance="neutral",
            horizon="1w",
            confidence=50,
            summary=summary,
            source_url=raw_post.url.strip() or None,
            as_of=raw_post.posted_at.date(),
            event_tags=[],
        )
        return payload.model_dump(mode="json")


class OpenAIExtractor(Extractor):
    extractor_name = "openai_structured"

    def __init__(
        self,
        *,
        api_key: str,
        model_name: str,
        base_url: str,
        timeout_seconds: float,
        max_output_tokens: int,
        openrouter_site_url: str = "",
        openrouter_app_name: str = "",
    ) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_output_tokens = max(1, max_output_tokens)
        self.openrouter_site_url = openrouter_site_url.strip()
        self.openrouter_app_name = openrouter_app_name.strip()

    def extract(self, raw_post: RawPost) -> dict[str, Any]:
        response_json = self._call_openai(raw_post)
        payload = ExtractionPayload.model_validate(response_json)
        return payload.model_dump(mode="json")

    def _call_openai(self, raw_post: RawPost) -> dict[str, Any]:
        system_prompt = (
            "You extract structured investment-view signals from a single post. "
            "Only output fields required by schema. "
            "If stance/horizon/confidence cannot be inferred, return null. "
            "Do not guess symbols; keep assets empty when unknown."
        )
        user_prompt = (
            f"platform: {raw_post.platform}\n"
            f"author_handle: {raw_post.author_handle}\n"
            f"url: {raw_post.url}\n"
            f"posted_at: {raw_post.posted_at.isoformat()}\n"
            f"content_text:\n{raw_post.content_text}"
        )

        request_payload = {
            "model": self.model_name,
            "temperature": 0,
            "max_tokens": self.max_output_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "post_extraction",
                    "strict": True,
                    "schema": EXTRACTION_JSON_SCHEMA,
                },
            },
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.openrouter_site_url:
            headers["HTTP-Referer"] = self.openrouter_site_url
        if self.openrouter_app_name:
            headers["X-Title"] = self.openrouter_app_name

        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=request_payload,
            )

        if response.status_code >= 400:
            body_preview = response.text[:500]
            raise RuntimeError(f"OpenAI request failed: status={response.status_code}, body={body_preview}")

        body = response.json()
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("OpenAI response missing choices")

        message = choices[0].get("message", {})
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            refusal = message.get("refusal")
            raise RuntimeError(f"OpenAI returned empty content. refusal={refusal}")

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"OpenAI content is not valid JSON: {exc}") from exc

        if not isinstance(parsed, dict):
            raise RuntimeError("OpenAI content is not a JSON object")
        return parsed


def default_extracted_json(raw_post: RawPost) -> dict[str, Any]:
    payload = ExtractionPayload(
        assets=[],
        stance=None,
        horizon=None,
        confidence=None,
        summary=None,
        source_url=raw_post.url.strip() or None,
        as_of=raw_post.posted_at.date(),
        event_tags=[],
    )
    return payload.model_dump(mode="json")


def reset_openai_call_budget_counter() -> None:
    global _OPENAI_CALLS_MADE
    with _OPENAI_CALLS_LOCK:
        _OPENAI_CALLS_MADE = 0


def try_consume_openai_call_budget(settings: Settings) -> bool:
    budget = max(0, settings.openai_call_budget)
    if budget == 0:
        return True

    global _OPENAI_CALLS_MADE
    with _OPENAI_CALLS_LOCK:
        if _OPENAI_CALLS_MADE >= budget:
            return False
        _OPENAI_CALLS_MADE += 1
        return True


def get_openai_call_budget_remaining(settings: Settings) -> int | None:
    budget = max(0, settings.openai_call_budget)
    if budget == 0:
        return None

    with _OPENAI_CALLS_LOCK:
        remaining = budget - _OPENAI_CALLS_MADE
    return max(0, remaining)


def select_extractor(settings: Settings) -> Extractor:
    mode = settings.extractor_mode.strip().lower()
    api_key = settings.openai_api_key.strip()

    if mode == "dummy":
        return DummyExtractor()

    if mode == "openai":
        if api_key:
            return OpenAIExtractor(
                api_key=api_key,
                model_name=settings.openai_model,
                base_url=settings.openai_base_url,
                timeout_seconds=settings.openai_timeout_seconds,
                max_output_tokens=settings.openai_max_output_tokens,
                openrouter_site_url=settings.openrouter_site_url,
                openrouter_app_name=settings.openrouter_app_name,
            )
        return DummyExtractor()

    if mode == "auto" and api_key:
        return OpenAIExtractor(
            api_key=api_key,
            model_name=settings.openai_model,
            base_url=settings.openai_base_url,
            timeout_seconds=settings.openai_timeout_seconds,
            max_output_tokens=settings.openai_max_output_tokens,
            openrouter_site_url=settings.openrouter_site_url,
            openrouter_app_name=settings.openrouter_app_name,
        )

    return DummyExtractor()
