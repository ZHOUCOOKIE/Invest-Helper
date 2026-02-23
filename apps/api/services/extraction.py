from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime
import json
from threading import Lock
from time import perf_counter
from typing import Any, Literal, Mapping

import httpx
from pydantic import BaseModel, ConfigDict, Field

from models import RawPost
from services.prompts.extract_v1 import PromptBundle
from settings import Settings

MARKET_VALUES = ["CRYPTO", "STOCK", "ETF", "FOREX", "OTHER", "AUTO"]
STANCE_VALUES = ["bull", "bear", "neutral"]
HORIZON_VALUES = ["intraday", "1w", "1m", "3m", "1y"]
_OPENAI_CALLS_MADE = 0
_OPENAI_CALLS_LOCK = Lock()


class OpenAIRequestError(RuntimeError):
    def __init__(self, *, status_code: int, body_preview: str):
        self.status_code = status_code
        self.body_preview = body_preview
        super().__init__(f"OpenAI request failed: status={status_code}, body={body_preview}")


class OpenAIFallbackError(RuntimeError):
    def __init__(self, message: str, *, fallback_reason: str):
        self.fallback_reason = fallback_reason
        super().__init__(message)


@dataclass
class ExtractionAudit:
    prompt_version: str = "extract_v1"
    prompt_text: str | None = None
    prompt_hash: str | None = None
    raw_model_output: str | None = None
    parsed_model_output: dict[str, Any] | None = None
    model_latency_ms: int | None = None
    model_input_tokens: int | None = None
    model_output_tokens: int | None = None


class ExtractedAsset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    name: str | None = None
    market: Literal["CRYPTO", "STOCK", "ETF", "FOREX", "OTHER", "AUTO"] | None = None


class ExtractionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assets: list[ExtractedAsset] = Field(default_factory=list)
    reasoning: str | None = Field(default=None, max_length=1024)
    stance: Literal["bull", "bear", "neutral"] | None = None
    horizon: Literal["intraday", "1w", "1m", "3m", "1y"] | None = None
    confidence: int | None = Field(default=None, ge=0, le=100)
    summary: str | None = None
    source_url: str | None = None
    as_of: date | None = None
    event_tags: list[str] = Field(default_factory=list)
    asset_views: list["AssetView"] = Field(default_factory=list)


class AssetView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    stance: Literal["bull", "bear", "neutral"]
    horizon: Literal["intraday", "1w", "1m", "3m", "1y"]
    confidence: int = Field(ge=0, le=100)
    reasoning: str | None = Field(default=None, max_length=1024)
    summary: str | None = Field(default=None, max_length=1024)
    drivers: list[str] = Field(default_factory=list)


ExtractionPayload.model_rebuild()


_EXTRACTION_TOP_LEVEL_KEYS = set(ExtractionPayload.model_fields.keys())
_ASSET_KEYS = set(ExtractedAsset.model_fields.keys())
_ASSET_VIEW_KEYS = set(AssetView.model_fields.keys())
_STANCE_ALIAS_MAP: dict[str, str] = {
    "bull": "bull",
    "long": "bull",
    "看多": "bull",
    "多头": "bull",
    "上涨": "bull",
    "乐观": "bull",
    "偏多": "bull",
    "做多": "bull",
    "bear": "bear",
    "short": "bear",
    "看空": "bear",
    "空头": "bear",
    "下跌": "bear",
    "悲观": "bear",
    "偏空": "bear",
    "做空": "bear",
    "neutral": "neutral",
    "中性": "neutral",
    "观望": "neutral",
    "不确定": "neutral",
    "谨慎乐观": "neutral",
    "稳定": "neutral",
}
_HORIZON_ALIAS_MAP: dict[str, str] = {
    "intraday": "intraday",
    "today": "intraday",
    "tonight": "intraday",
    "今晚": "intraday",
    "日内": "intraday",
    "短线": "intraday",
    "短期": "intraday",
    "short": "intraday",
    "1w": "1w",
    "week": "1w",
    "一周": "1w",
    "本周": "1w",
    "1m": "1m",
    "month": "1m",
    "一个月": "1m",
    "一月": "1m",
    "本月": "1m",
    "短中期": "1m",
    "3m": "3m",
    "quarter": "3m",
    "一季": "3m",
    "三月": "3m",
    "三个月": "3m",
    "1y": "1y",
    "year": "1y",
    "一年": "1y",
    "长期": "1y",
}


EXTRACTION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "assets",
        "reasoning",
        "stance",
        "horizon",
        "confidence",
        "summary",
        "source_url",
        "as_of",
        "event_tags",
        "asset_views",
    ],
    "properties": {
        "assets": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["symbol"],
                "properties": {
                    "symbol": {"type": "string", "minLength": 1, "maxLength": 32},
                    "name": {"type": ["string", "null"], "maxLength": 255},
                    "market": {"type": ["string", "null"], "enum": MARKET_VALUES + [None]},
                },
            },
        },
        "reasoning": {"type": ["string", "null"], "maxLength": 1024},
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
        "asset_views": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["symbol", "stance", "horizon", "confidence", "reasoning", "summary"],
                "properties": {
                    "symbol": {"type": "string", "minLength": 1, "maxLength": 32},
                    "stance": {"type": "string", "enum": STANCE_VALUES},
                    "horizon": {"type": "string", "enum": HORIZON_VALUES},
                    "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
                    "reasoning": {"type": ["string", "null"], "maxLength": 1024},
                    "summary": {"type": ["string", "null"], "maxLength": 1024},
                    "drivers": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1, "maxLength": 64},
                    },
                },
            },
        },
    },
}


class Extractor(ABC):
    model_name: str
    extractor_name: str
    _prompt_bundle: PromptBundle | None = None
    _audit: ExtractionAudit = ExtractionAudit()

    def set_prompt_bundle(self, bundle: PromptBundle) -> None:
        self._prompt_bundle = bundle
        self._audit = ExtractionAudit(
            prompt_version=bundle.version,
            prompt_text=bundle.text,
            prompt_hash=bundle.hash,
        )

    def get_audit(self) -> ExtractionAudit:
        return self._audit

    def _record_model_output(
        self,
        *,
        raw_model_output: str | None = None,
        parsed_model_output: dict[str, Any] | None = None,
        model_latency_ms: int | None = None,
        model_input_tokens: int | None = None,
        model_output_tokens: int | None = None,
    ) -> None:
        if raw_model_output is not None:
            self._audit.raw_model_output = raw_model_output
        if parsed_model_output is not None:
            self._audit.parsed_model_output = parsed_model_output
        if model_latency_ms is not None:
            self._audit.model_latency_ms = model_latency_ms
        if model_input_tokens is not None:
            self._audit.model_input_tokens = model_input_tokens
        if model_output_tokens is not None:
            self._audit.model_output_tokens = model_output_tokens

    @abstractmethod
    def extract(self, raw_post: RawPost) -> dict[str, Any]:
        """Return structured candidates inferred from a raw post."""


def normalize_extracted_json(
    extracted_json: dict[str, Any],
    *,
    posted_at: datetime | date | str | None = None,
    include_meta: bool = False,
    alias_to_symbol: Mapping[str, str] | None = None,
    known_symbols: set[str] | None = None,
) -> dict[str, Any]:
    normalized = {k: v for k, v in extracted_json.items() if k in _EXTRACTION_TOP_LEVEL_KEYS}
    normalized_meta: dict[str, Any] = {}
    if isinstance(extracted_json.get("meta"), dict):
        normalized_meta = dict(extracted_json["meta"])

    reasoning_raw = normalized.get("reasoning")
    normalized["reasoning"] = _normalize_reasoning(reasoning_raw)

    stance_raw = normalized.get("stance")
    normalized["stance"] = _normalize_stance(stance_raw)

    horizon_raw = normalized.get("horizon")
    normalized["horizon"] = _normalize_horizon(horizon_raw)

    confidence_raw = normalized.get("confidence")
    normalized["confidence"] = _normalize_confidence(confidence_raw)

    assets_raw = normalized.get("assets")
    normalized["assets"] = _normalize_assets(assets_raw)

    alias_map = _normalize_alias_symbol_mapping(alias_to_symbol or {})
    known_symbol_set = _normalize_known_symbols(known_symbols)

    asset_views_raw = normalized.get("asset_views")
    normalized["asset_views"] = _normalize_asset_views(
        asset_views_raw,
        allow_missing_symbol=bool(alias_map),
    )
    if alias_map and normalized["asset_views"]:
        corrections = _apply_alias_symbol_corrections(
            normalized["asset_views"],
            alias_to_symbol=alias_map,
            known_symbols=known_symbol_set,
        )
        if corrections:
            normalized_meta["alias_corrections"] = corrections
    normalized["asset_views"] = [item for item in normalized["asset_views"] if item.get("symbol")]
    if not normalized["asset_views"] and normalized["assets"]:
        normalized["asset_views"] = _derive_asset_views_from_global(normalized)
        if normalized["asset_views"]:
            normalized_meta["derived_from_global"] = True

    event_tags_raw = normalized.get("event_tags")
    normalized["event_tags"] = _normalize_event_tags(event_tags_raw)

    as_of_raw = normalized.get("as_of")
    normalized["as_of"] = _normalize_as_of(
        as_of_raw,
        posted_at=posted_at if posted_at is not None else extracted_json.get("posted_at"),
    )
    if include_meta and normalized_meta:
        normalized["meta"] = normalized_meta

    return normalized


def _coerce_call_result(
    model_json: dict[str, Any],
) -> tuple[dict[str, Any], str | None, int | None, int | None, int | None]:
    if "parsed_content" in model_json:
        parsed_content = model_json.get("parsed_content")
        raw_content = model_json.get("raw_content")
        latency_ms = model_json.get("latency_ms")
        input_tokens = model_json.get("input_tokens")
        output_tokens = model_json.get("output_tokens")
        if isinstance(parsed_content, dict):
            return (
                parsed_content,
                raw_content if isinstance(raw_content, str) else None,
                latency_ms if isinstance(latency_ms, int) else None,
                input_tokens if isinstance(input_tokens, int) else None,
                output_tokens if isinstance(output_tokens, int) else None,
            )
    return (model_json, json.dumps(model_json, ensure_ascii=False), None, None, None)


def _normalize_stance(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return "neutral"
    normalized = value.strip()
    mapped = _STANCE_ALIAS_MAP.get(normalized.lower()) or _STANCE_ALIAS_MAP.get(normalized)
    if mapped:
        return mapped
    lowered = normalized.lower()
    if "bull" in lowered or "long" in lowered:
        return "bull"
    if "bear" in lowered or "short" in lowered:
        return "bear"
    if any(token in normalized for token in ("看多", "多头", "上涨", "乐观")):
        return "bull"
    if any(token in normalized for token in ("看空", "空头", "下跌", "悲观")):
        return "bear"
    if any(token in normalized for token in ("中性", "观望", "不确定", "谨慎")):
        return "neutral"
    return "neutral"


def _normalize_horizon(value: Any) -> str:
    if value is None:
        return "1w"
    if not isinstance(value, str):
        return "1w"
    normalized = value.strip()
    if not normalized:
        return "1w"
    mapped = _HORIZON_ALIAS_MAP.get(normalized.lower()) or _HORIZON_ALIAS_MAP.get(normalized)
    if mapped:
        return mapped
    lowered = normalized.lower()
    if "intraday" in lowered or "today" in lowered or "tonight" in lowered or "short" in lowered:
        return "intraday"
    if "week" in lowered:
        return "1w"
    if "month" in lowered:
        return "1m"
    if "quarter" in lowered:
        return "3m"
    if "year" in lowered or "long" in lowered:
        return "1y"
    if any(token in normalized for token in ("今晚", "日内", "短线")):
        return "intraday"
    if any(token in normalized for token in ("一周", "本周")):
        return "1w"
    if any(token in normalized for token in ("一月", "一个月", "本月")):
        return "1m"
    if any(token in normalized for token in ("一季", "三月", "三个月")):
        return "3m"
    if any(token in normalized for token in ("一年", "长期")):
        return "1y"
    return mapped or "1w"


def _normalize_confidence(value: Any) -> int:
    if value is None:
        return 50
    parsed: float
    if isinstance(value, bool):
        return 50
    if isinstance(value, (int, float)):
        parsed = float(value)
    elif isinstance(value, str):
        try:
            parsed = float(value.strip())
        except ValueError:
            return 50
    else:
        return 50
    if 0 <= parsed <= 1:
        parsed *= 100
    rounded = int(round(parsed))
    if rounded < 0 or rounded > 100:
        return 50
    return rounded


def _normalize_assets(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []

    if isinstance(value, dict) or isinstance(value, str):
        candidates = [value]
    elif isinstance(value, list):
        candidates = value
    else:
        return []

    normalized_assets: list[dict[str, Any]] = []
    for candidate in candidates:
        parsed_asset = _normalize_asset_item(candidate)
        if parsed_asset is not None:
            normalized_assets.append(parsed_asset)
    return normalized_assets


def _normalize_asset_views(value: Any, *, allow_missing_symbol: bool = False) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, dict):
        candidates = [value]
    elif isinstance(value, list):
        candidates = value
    else:
        return []

    normalized_views: list[dict[str, Any]] = []
    for candidate in candidates:
        normalized_view = _normalize_asset_view_item(candidate, allow_missing_symbol=allow_missing_symbol)
        if normalized_view is not None:
            normalized_views.append(normalized_view)
    return normalized_views


def _normalize_asset_view_item(value: Any, *, allow_missing_symbol: bool = False) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    normalized = {k: v for k, v in value.items() if k in _ASSET_VIEW_KEYS}
    symbol_raw = normalized.get("symbol")
    if not isinstance(symbol_raw, str) or not symbol_raw.strip():
        if not allow_missing_symbol:
            return None
        normalized["symbol"] = ""
    else:
        normalized["symbol"] = symbol_raw.strip().upper()

    if not allow_missing_symbol and not normalized["symbol"]:
        return None

    drivers_raw = normalized.get("drivers")
    normalized["drivers"] = _normalize_drivers(drivers_raw)
    normalized["stance"] = _normalize_stance(normalized.get("stance")) or "neutral"
    normalized["horizon"] = _normalize_horizon(normalized.get("horizon"))
    normalized["confidence"] = _normalize_confidence(normalized.get("confidence"))
    normalized["reasoning"] = _normalize_reasoning(normalized.get("reasoning"))
    normalized["summary"] = _normalize_reasoning(normalized.get("summary"))
    return normalized


def _normalize_alias_symbol_mapping(value: Mapping[str, str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for alias, symbol in value.items():
        normalized_alias = alias.strip().lower()
        normalized_symbol = symbol.strip().upper()
        if normalized_alias and normalized_symbol:
            mapping[normalized_alias] = normalized_symbol
    return mapping


def _normalize_known_symbols(value: set[str] | None) -> set[str]:
    if not value:
        return set()
    return {item.strip().upper() for item in value if isinstance(item, str) and item.strip()}


def _apply_alias_symbol_corrections(
    asset_views: list[dict[str, Any]],
    *,
    alias_to_symbol: dict[str, str],
    known_symbols: set[str],
) -> list[dict[str, str]]:
    corrections: list[dict[str, str]] = []
    alias_items = sorted(alias_to_symbol.items(), key=lambda item: len(item[0]), reverse=True)

    for view in asset_views:
        symbol_raw = view.get("symbol")
        current_symbol = symbol_raw.strip().upper() if isinstance(symbol_raw, str) else ""
        corrected_symbol: str | None = None

        direct_alias_target = alias_to_symbol.get(current_symbol.lower()) if current_symbol else None
        if direct_alias_target:
            corrected_symbol = direct_alias_target

        needs_text_match = not current_symbol or (bool(known_symbols) and current_symbol not in known_symbols)
        if corrected_symbol is None and needs_text_match:
            matched = _find_alias_match_for_view(view, alias_items)
            if matched is not None:
                _, matched_symbol = matched
                corrected_symbol = matched_symbol

        if corrected_symbol and corrected_symbol != current_symbol:
            view["symbol"] = corrected_symbol
            corrections.append({"from": current_symbol, "to": corrected_symbol, "reason": "alias_match"})

    return corrections


def _find_alias_match_for_view(
    view: dict[str, Any],
    alias_items: list[tuple[str, str]],
) -> tuple[str, str] | None:
    text_fields: list[str] = []
    summary = view.get("summary")
    reasoning = view.get("reasoning")
    drivers = view.get("drivers")
    if isinstance(summary, str) and summary.strip():
        text_fields.append(summary.strip())
    if isinstance(reasoning, str) and reasoning.strip():
        text_fields.append(reasoning.strip())
    if isinstance(drivers, list):
        for item in drivers:
            if isinstance(item, str) and item.strip():
                text_fields.append(item.strip())

    for text in text_fields:
        text_norm = text.lower()
        for alias, symbol in alias_items:
            if alias in text_norm:
                return alias, symbol
    return None


def _derive_asset_views_from_global(normalized: dict[str, Any]) -> list[dict[str, Any]]:
    views: list[dict[str, Any]] = []
    stance = _normalize_stance(normalized.get("stance")) or "neutral"
    horizon = _normalize_horizon(normalized.get("horizon"))
    confidence = _normalize_confidence(normalized.get("confidence"))
    reasoning = _normalize_reasoning(normalized.get("reasoning"))
    summary = _normalize_reasoning(normalized.get("summary"))
    for asset in normalized.get("assets", []):
        symbol = asset.get("symbol")
        if not isinstance(symbol, str) or not symbol.strip():
            continue
        views.append(
            {
                "symbol": symbol.strip().upper(),
                "stance": stance,
                "horizon": horizon,
                "confidence": confidence,
                "reasoning": reasoning,
                "summary": summary,
                "drivers": [],
            }
        )
    return views


def _normalize_asset_item(value: Any) -> dict[str, Any] | None:
    if isinstance(value, str):
        symbol = value.strip()
        if not symbol:
            return None
        return {"symbol": symbol.upper(), "market": "AUTO"}

    if not isinstance(value, dict):
        return None

    normalized = {k: v for k, v in value.items() if k in _ASSET_KEYS}
    symbol = normalized.get("symbol")
    if not isinstance(symbol, str) or not symbol.strip():
        return None
    normalized["symbol"] = symbol.strip().upper()
    normalized["market"] = _normalize_market(normalized.get("market"))
    return normalized


def _normalize_market(value: Any) -> str:
    if isinstance(value, str):
        normalized = value.strip().upper()
        if normalized in MARKET_VALUES:
            return normalized
    return "AUTO"


def _normalize_as_of(value: Any, *, posted_at: datetime | date | str | None) -> str | None:
    normalized_value = _to_iso_date(value)
    if normalized_value is not None:
        return normalized_value
    return _to_iso_date(posted_at)


def _to_iso_date(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if len(text) >= 10:
            date_part = text[:10]
            try:
                return date.fromisoformat(date_part).isoformat()
            except ValueError:
                pass
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            return parsed.date().isoformat()
        except ValueError:
            return None
    return None


def _normalize_reasoning(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _normalize_event_tags(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized_tags: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        tag = item.strip().lower()
        if tag:
            normalized_tags.append(tag[:64])
    return normalized_tags


def _normalize_drivers(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    drivers: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        cleaned = item.strip()
        if cleaned:
            drivers.append(cleaned[:64])
    return drivers


class DummyExtractor(Extractor):
    model_name = "dummy-v2"
    extractor_name = "dummy"

    def __init__(self) -> None:
        self._audit = ExtractionAudit()

    def extract(self, raw_post: RawPost) -> dict[str, Any]:
        content = raw_post.content_text.strip()
        summary = content[:280] if content else None

        payload = ExtractionPayload(
            assets=[],
            reasoning=None,
            stance="neutral",
            horizon="1w",
            confidence=50,
            summary=summary,
            source_url=raw_post.url.strip() or None,
            as_of=raw_post.posted_at.date(),
            event_tags=[],
        )
        parsed = payload.model_dump(mode="json")
        self._record_model_output(
            raw_model_output=json.dumps(parsed, ensure_ascii=False),
            parsed_model_output=parsed,
            model_latency_ms=0,
        )
        return parsed


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
        self._audit = ExtractionAudit()

    def extract(self, raw_post: RawPost) -> dict[str, Any]:
        try:
            model_json = self._call_openai(raw_post, response_mode="structured")
            parsed_content, raw_content, latency_ms, input_tokens, output_tokens = _coerce_call_result(model_json)
            self._record_model_output(
                raw_model_output=raw_content,
                parsed_model_output=parsed_content,
                model_latency_ms=latency_ms,
                model_input_tokens=input_tokens,
                model_output_tokens=output_tokens,
            )
            payload = ExtractionPayload.model_validate(parsed_content)
            parsed = payload.model_dump(mode="json")
            parsed["meta"] = {"extraction_mode": "structured"}
            return parsed
        except OpenAIRequestError as exc:
            self._record_model_output(raw_model_output=exc.body_preview)
            if not self._is_structured_outputs_unsupported(exc):
                raise

            from settings import get_settings

            settings = get_settings()
            if not try_consume_openai_call_budget(settings):
                raise OpenAIFallbackError(
                    "OpenAI structured-outputs unsupported, but retry budget exhausted",
                    fallback_reason="structured_unsupported",
                ) from exc

            try:
                model_json = self._call_openai(raw_post, response_mode="json_mode")
                parsed_content, raw_content, latency_ms, input_tokens, output_tokens = _coerce_call_result(model_json)
                self._record_model_output(
                    raw_model_output=raw_content,
                    parsed_model_output=parsed_content,
                    model_latency_ms=latency_ms,
                    model_input_tokens=input_tokens,
                    model_output_tokens=output_tokens,
                )
                normalized_json = normalize_extracted_json(parsed_content, posted_at=raw_post.posted_at)
                payload = ExtractionPayload.model_validate(normalized_json)
                parsed = payload.model_dump(mode="json")
                parsed["meta"] = {
                    "extraction_mode": "json_mode",
                    "fallback_reason": "structured_unsupported",
                }
                return parsed
            except Exception as retry_exc:
                raise OpenAIFallbackError(
                    f"OpenAI json_mode retry failed after structured unsupported: {retry_exc}",
                    fallback_reason="structured_unsupported",
                ) from retry_exc

    def _call_openai(
        self, raw_post: RawPost, *, response_mode: Literal["structured", "json_mode"]
    ) -> dict[str, Any]:
        strict_json_hint = (
            "Return exactly one JSON object. "
            "Do not include markdown fences, explanations, or any extra text."
        )
        json_mode_hard_rules = (
            "JSON mode hard rules (中英都适用): "
            "only return one JSON object. "
            "stance must be one of bull/bear/neutral; if uncertain use neutral. "
            "horizon must be one of intraday/1w/1m/3m/1y; if uncertain use 1w. "
            "confidence should be integer 0-100. "
            "include reasoning (1-3 short sentences). "
            "include event_tags as string array. "
            "assets should be array of objects with symbol/name/market, symbol is required. "
            "assets[*].market must be one of CRYPTO/STOCK/ETF/FOREX/OTHER/AUTO. "
            "asset_views must be array of per-asset views, each contains "
            "symbol/stance/horizon/confidence/reasoning/summary and optional drivers. "
            "as_of must be date-only string in YYYY-MM-DD (no time)."
        )
        system_prompt = (
            "You extract structured investment-view signals from a single post. "
            "Only output fields required by schema. "
            "If stance/horizon/confidence cannot be inferred, return null. "
            "Do not guess symbols; keep assets empty when unknown. "
            f"{strict_json_hint} "
            f"{json_mode_hard_rules if response_mode == 'json_mode' else ''}"
        )
        user_prompt = self._build_user_prompt(raw_post)

        request_payload: dict[str, Any] = {
            "model": self.model_name,
            "temperature": 0,
            "max_tokens": self.max_output_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if response_mode == "structured":
            request_payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "post_extraction",
                    "strict": True,
                    "schema": EXTRACTION_JSON_SCHEMA,
                },
            }
        else:
            request_payload["response_format"] = {"type": "json_object"}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.openrouter_site_url:
            headers["HTTP-Referer"] = self.openrouter_site_url
        if self.openrouter_app_name:
            headers["X-Title"] = self.openrouter_app_name

        started = perf_counter()
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=request_payload,
            )
        latency_ms = int((perf_counter() - started) * 1000)

        if response.status_code >= 400:
            body_preview = response.text[:500]
            self._record_model_output(raw_model_output=body_preview, model_latency_ms=latency_ms)
            raise OpenAIRequestError(status_code=response.status_code, body_preview=body_preview)

        body = response.json()
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("OpenAI response missing choices")

        message = choices[0].get("message", {})
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            refusal = message.get("refusal")
            raise RuntimeError(f"OpenAI returned empty content. refusal={refusal}")

        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            self._record_model_output(raw_model_output=content, model_latency_ms=latency_ms)
            raise RuntimeError(f"OpenAI content is not valid JSON: {exc}") from exc

        if not isinstance(parsed, dict):
            raise RuntimeError("OpenAI content is not a JSON object")
        return {
            "raw_content": content,
            "parsed_content": parsed,
            "latency_ms": latency_ms,
            "input_tokens": int(prompt_tokens) if isinstance(prompt_tokens, int) else None,
            "output_tokens": int(completion_tokens) if isinstance(completion_tokens, int) else None,
        }

    def _build_user_prompt(self, raw_post: RawPost) -> str:
        bundle = self._prompt_bundle
        if bundle and bundle.text.strip():
            return bundle.text
        return (
            f"platform: {raw_post.platform}\n"
            f"author_handle: {raw_post.author_handle}\n"
            f"url: {raw_post.url}\n"
            f"posted_at: {raw_post.posted_at.isoformat()}\n"
            f"content_text:\n{raw_post.content_text}"
        )

    def _is_structured_outputs_unsupported(self, exc: OpenAIRequestError) -> bool:
        if exc.status_code != 400:
            return False

        text = exc.body_preview.lower()
        has_structured_token = "structured-outputs" in text or "structured_outputs" in text
        has_unsupported_signal = "does not support" in text or "unsupported" in text
        return has_structured_token and has_unsupported_signal


def default_extracted_json(raw_post: RawPost) -> dict[str, Any]:
    payload = ExtractionPayload(
        assets=[],
        reasoning=None,
        stance=None,
        horizon=None,
        confidence=None,
        summary=None,
        source_url=raw_post.url.strip() or None,
        as_of=raw_post.posted_at.date(),
        event_tags=[],
        asset_views=[],
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
