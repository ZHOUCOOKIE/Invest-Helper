from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime
import json
import logging
import re
from threading import Lock
from time import perf_counter
from typing import Any, Literal, Mapping

import httpx
from pydantic import BaseModel, ConfigDict, Field

from models import RawPost
from services.prompts.extraction_prompt import PromptBundle, render_prompt_bundle
from settings import Settings

MARKET_VALUES = ["CRYPTO", "STOCK", "ETF", "FOREX", "OTHER"]
STANCE_VALUES = ["bull", "bear", "neutral"]
HORIZON_VALUES = ["intraday", "1w", "1m", "3m", "1y"]
LIBRARY_TAG_VALUES = ["macro", "industry", "thesis", "strategy", "risk", "events"]
LIBRARY_RULESET_VERSION = "A3_islibrary_strict_v1"
ASSET_VIEW_MIN_CONFIDENCE = 70
_OPENAI_CALLS_MADE = 0
_OPENAI_CALLS_LOCK = Lock()
OPENAI_PROVIDER_OPENROUTER = "openrouter"
OPENAI_PROVIDER_COMPATIBLE = "openai_compatible"
EXTRACTION_OUTPUT_STRUCTURED = "structured"
EXTRACTION_OUTPUT_TEXT_JSON = "text_json"
logger = logging.getLogger("uvicorn.error")


class OpenAIRequestError(RuntimeError):
    def __init__(self, *, status_code: int, body_preview: str, retry_after_seconds: float | None = None):
        self.status_code = status_code
        self.body_preview = body_preview
        self.retry_after_seconds = retry_after_seconds
        super().__init__(f"OpenAI request failed: status={status_code}, body={body_preview}")


class OpenAIFallbackError(RuntimeError):
    def __init__(self, message: str, *, fallback_reason: str):
        self.fallback_reason = fallback_reason
        super().__init__(message)


def _parse_retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        seconds = float(value.strip())
    except ValueError:
        return None
    return max(0.0, seconds)


class TextJsonParseError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        raw_content: str,
        parse_strategy_used: str,
        repaired: bool,
        parse_error_reason: str,
        suspected_truncated: bool,
        finish_reason: str | None = None,
    ) -> None:
        self.raw_content = raw_content
        self.parse_strategy_used = parse_strategy_used
        self.repaired = repaired
        self.parse_error_reason = parse_error_reason
        self.suspected_truncated = suspected_truncated
        self.finish_reason = finish_reason
        super().__init__(message)

    def to_meta(self, *, truncated_retry_used: bool, invalid_json_retry_used: bool = False) -> dict[str, Any]:
        return {
            "parse_error": True,
            "parse_error_reason": self.parse_error_reason,
            "parse_strategy_used": self.parse_strategy_used,
            "repaired": self.repaired,
            "truncated_retry_used": truncated_retry_used,
            "invalid_json_retry_used": invalid_json_retry_used,
            "finish_reason": self.finish_reason,
            "finish_reason_best_effort": True,
        }


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


class ExtractionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_url: str
    as_of: date
    islibrary: Literal[0, 1] = 0
    hasview: Literal[0, 1] = 0
    asset_views: list["AssetView"] = Field(default_factory=list)
    library_entry: "LibraryEntry | None" = None


class AssetView(BaseModel):
    model_config = ConfigDict(extra="ignore")

    symbol: str
    market: Literal["CRYPTO", "STOCK", "ETF", "FOREX", "OTHER"] = "OTHER"
    stance: Literal["bull", "bear", "neutral"]
    horizon: Literal["intraday", "1w", "1m", "3m", "1y"]
    confidence: int = Field(ge=ASSET_VIEW_MIN_CONFIDENCE, le=100)
    summary: str = Field(max_length=1024)


class LibraryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tag: Literal["macro", "industry", "thesis", "strategy", "risk", "events"]
    summary: str = Field(max_length=512)


ExtractionPayload.model_rebuild()


_ASSET_VIEW_KEYS = set(AssetView.model_fields.keys())
EXTRACTION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "as_of",
        "source_url",
        "islibrary",
        "hasview",
        "asset_views",
        "library_entry",
    ],
    "properties": {
        "as_of": {"type": "string", "format": "date"},
        "source_url": {"type": "string", "maxLength": 1024},
        "islibrary": {"type": "integer", "enum": [0, 1]},
        "hasview": {"type": "integer", "enum": [0, 1]},
        "asset_views": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["symbol", "market", "stance", "horizon", "confidence", "summary"],
                "properties": {
                    "symbol": {"type": "string", "minLength": 1, "maxLength": 32},
                    "market": {"type": "string", "enum": MARKET_VALUES},
                    "stance": {"type": "string", "enum": STANCE_VALUES},
                    "horizon": {"type": "string", "enum": HORIZON_VALUES},
                    "confidence": {"type": "integer", "minimum": ASSET_VIEW_MIN_CONFIDENCE, "maximum": 100},
                    "summary": {"type": "string", "maxLength": 1024},
                },
            },
        },
        "library_entry": {
            "type": ["object", "null"],
            "additionalProperties": False,
            "required": ["tag", "summary"],
            "properties": {
                "tag": {"type": "string", "enum": LIBRARY_TAG_VALUES},
                "summary": {"type": "string", "maxLength": 512},
            },
        },
    },
    "allOf": [
        {
            "if": {"properties": {"islibrary": {"const": 1}}},
            "then": {
                "properties": {
                    "library_entry": {
                        "type": "object",
                        "required": ["tag", "summary"],
                    },
                },
            },
        },
        {
            "if": {"properties": {"islibrary": {"const": 0}}},
            "then": {"properties": {"library_entry": {"type": "null"}}},
        }
    ],
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
    normalized: dict[str, Any] = {
        key: extracted_json.get(key)
        for key in ("as_of", "source_url", "islibrary", "hasview", "asset_views", "library_entry")
        if key in extracted_json
    }
    normalized_meta: dict[str, Any] = {}
    if isinstance(extracted_json.get("meta"), dict):
        normalized_meta = dict(extracted_json["meta"])
    alias_map = _normalize_alias_symbol_mapping(alias_to_symbol or {})
    known_symbol_set = _normalize_known_symbols(known_symbols)

    asset_views_raw = normalized.get("asset_views")
    normalized_asset_views = _normalize_asset_views(
        asset_views_raw,
        allow_missing_symbol=bool(alias_map),
    )
    normalized["asset_views"] = normalized_asset_views
    if alias_map and normalized["asset_views"]:
        _apply_alias_symbol_corrections(
            normalized["asset_views"],
            alias_to_symbol=alias_map,
            known_symbols=known_symbol_set,
        )
    valid_asset_views, _asset_views_meta = _filter_asset_views_by_symbol_quality(normalized["asset_views"])
    normalized["asset_views"] = valid_asset_views

    as_of_raw = normalized.get("as_of")
    normalized_as_of = _normalize_as_of(
        as_of_raw,
        posted_at=posted_at if posted_at is not None else extracted_json.get("posted_at"),
    )
    normalized["as_of"] = normalized_as_of or date.today().isoformat()
    normalized["source_url"] = (
        _normalize_text(normalized.get("source_url"))
        or _normalize_text(extracted_json.get("source_url"))
        or _normalize_text(extracted_json.get("url"))
        or ""
    )

    raw_islibrary = extracted_json.get("islibrary")
    islibrary = _normalize_islibrary(raw_islibrary)
    normalized["islibrary"] = islibrary

    raw_library_entry = normalized.get("library_entry")
    normalized_library_entry, library_entry_drop_reason = _normalize_library_entry(raw_library_entry)
    normalized["library_entry"] = normalized_library_entry

    if islibrary == 1 and normalized_library_entry is None:
        normalized["islibrary"] = 0
        normalized_meta["library_downgraded"] = True
        normalized_meta["library_downgrade_reason"] = library_entry_drop_reason or "invalid_library_shape"
        normalized["library_entry"] = None
        islibrary = 0

    if islibrary == 0:
        normalized["library_entry"] = None

    normalized_asset_views, _threshold_meta = _filter_asset_views_by_confidence(
        normalized["asset_views"],
        min_confidence=ASSET_VIEW_MIN_CONFIDENCE,
    )
    normalized["asset_views"] = normalized_asset_views
    normalized_asset_views, _language_meta = _filter_asset_views_by_summary_language(normalized["asset_views"])
    normalized["asset_views"] = normalized_asset_views
    normalized["hasview"] = 1 if normalized["asset_views"] else 0

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


def _unwrap_extracted_json_envelope(payload: dict[str, Any]) -> dict[str, Any]:
    # Some upstreams may wrap model content as {"extracted_json": {...}}.
    inner = payload.get("extracted_json")
    if isinstance(inner, dict):
        return inner
    return payload


def _normalize_stance(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized if normalized in STANCE_VALUES else None


def _normalize_horizon(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    return normalized if normalized in HORIZON_VALUES else None


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


def _count_empty_symbol_asset_views_from_raw(value: Any) -> int:
    if isinstance(value, dict):
        candidates = [value]
    elif isinstance(value, list):
        candidates = value
    else:
        return 0
    count = 0
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        symbol_raw = candidate.get("symbol")
        if not isinstance(symbol_raw, str) or not symbol_raw.strip():
            count += 1
    return count


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

    normalized_stance = _normalize_stance(normalized.get("stance"))
    normalized_horizon = _normalize_horizon(normalized.get("horizon"))
    if normalized_stance is None or normalized_horizon is None:
        return None
    normalized["stance"] = normalized_stance
    normalized["horizon"] = normalized_horizon
    normalized["confidence"] = _normalize_confidence(normalized.get("confidence"))
    normalized["market"] = _normalize_market(normalized.get("market"))
    normalized["summary"] = _normalize_text(normalized.get("summary")) or ""
    return normalized


def _filter_asset_views_by_symbol_quality(asset_views: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    valid_views: list[dict[str, Any]] = []
    dropped_empty_symbol_count = 0
    dropped_invalid_symbol_count = 0
    dropped_invalid_symbols_sample: list[str] = []
    for item in asset_views:
        symbol_raw = item.get("symbol")
        if not isinstance(symbol_raw, str) or not symbol_raw.strip():
            dropped_empty_symbol_count += 1
            continue
        normalized_symbol = _normalize_symbol_for_storage(symbol_raw)
        if normalized_symbol is None:
            dropped_invalid_symbol_count += 1
            if len(dropped_invalid_symbols_sample) < 3:
                dropped_invalid_symbols_sample.append(symbol_raw)
            continue
        item["symbol"] = normalized_symbol
        valid_views.append(item)
    meta: dict[str, Any] = {
        "asset_views_dropped_empty_symbol_count": dropped_empty_symbol_count,
    }
    if dropped_invalid_symbol_count:
        meta["dropped_invalid_symbol_count"] = (
            int(meta.get("dropped_invalid_symbol_count", 0)) + dropped_invalid_symbol_count
        )
        meta["dropped_invalid_symbols_sample"] = dropped_invalid_symbols_sample
    return valid_views, meta


def _filter_asset_views_by_confidence(
    asset_views: list[dict[str, Any]],
    *,
    min_confidence: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    valid_views: list[dict[str, Any]] = []
    dropped_count = 0
    for item in asset_views:
        confidence_value = item.get("confidence")
        confidence = int(confidence_value) if isinstance(confidence_value, int) else _normalize_confidence(confidence_value)
        if confidence < min_confidence:
            dropped_count += 1
            continue
        item["confidence"] = confidence
        valid_views.append(item)
    return (
        valid_views,
        {
            "asset_views_min_confidence_threshold": min_confidence,
            "asset_views_dropped_low_confidence_count": dropped_count,
        },
    )


def _filter_asset_views_by_summary_language(
    asset_views: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    valid_views: list[dict[str, Any]] = []
    dropped_count = 0
    for item in asset_views:
        summary = item.get("summary")
        if not isinstance(summary, str) or not _is_chinese_summary(summary):
            dropped_count += 1
            continue
        valid_views.append(item)
    return valid_views, {"asset_views_dropped_non_zh_summary_count": dropped_count}


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
    if isinstance(summary, str) and summary.strip():
        text_fields.append(summary.strip())

    for text in text_fields:
        text_norm = text.lower()
        for alias, symbol in alias_items:
            if alias in text_norm:
                return alias, symbol
    return None


def _merge_symbol_drop_meta(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if key in {"dropped_invalid_symbol_count", "asset_views_dropped_empty_symbol_count"}:
            target[key] = int(target.get(key, 0) or 0) + int(value or 0)
            continue
        if key == "dropped_invalid_symbols_sample":
            existing = list(target.get(key) or [])
            for item in list(value or []):
                if len(existing) >= 3:
                    break
                if item not in existing:
                    existing.append(item)
            if existing:
                target[key] = existing
            continue
        target[key] = value


def _normalize_market(value: Any) -> str:
    if isinstance(value, str):
        normalized = value.strip().upper()
        if normalized in MARKET_VALUES:
            return normalized
    return "OTHER"


def _normalize_islibrary(value: Any) -> int:
    return 1 if isinstance(value, int) and not isinstance(value, bool) and value == 1 else 0


def _normalize_library_entry(value: Any) -> tuple[dict[str, Any] | None, str | None]:
    if value is None:
        return None, None
    if not isinstance(value, dict):
        return None, "invalid_library_shape"
    tag = value.get("tag")
    if not isinstance(tag, str):
        return None, "invalid_library_tags"
    normalized_tag = tag.strip().lower()
    if normalized_tag not in LIBRARY_TAG_VALUES:
        return None, "invalid_library_tags"
    summary = _normalize_text(value.get("summary"))
    if summary is None:
        return None, "invalid_library_summary"
    if not _is_chinese_summary(summary):
        return None, "invalid_library_summary_language"
    if summary != "测试":
        return None, "invalid_library_summary_exact"
    return {
        "tag": normalized_tag,
        "summary": summary,
    }, None


_ASCII_SYMBOL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-/]{0,29}$")
_CJK_PATTERN = re.compile(r"[\u3400-\u9fff]")
_FORBIDDEN_STRUCT_CHARS = set("{}[]")
_ZERO_WIDTH_CHARS = {"\u200b", "\u200c", "\u200d", "\ufeff"}


def _normalize_symbol_for_storage(value: str) -> str | None:
    cleaned = re.sub(r"\s+", " ", value.strip())
    if not cleaned:
        return None
    if len(cleaned) < 1 or len(cleaned) > 30:
        return None
    if any(ch in _ZERO_WIDTH_CHARS for ch in cleaned):
        return None
    if any(ch in _FORBIDDEN_STRUCT_CHARS for ch in cleaned):
        return None
    for ch in cleaned:
        if ch in {"\n", "\r", "\t"}:
            return None
        if ord(ch) < 32 or ord(ch) == 127:
            return None
    has_cjk = bool(_CJK_PATTERN.search(cleaned))
    if has_cjk:
        return cleaned
    if _ASCII_SYMBOL_PATTERN.fullmatch(cleaned) is None:
        return None
    return cleaned.upper()


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


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _is_chinese_summary(value: str) -> bool:
    return bool(_CJK_PATTERN.search(value))

class DummyExtractor(Extractor):
    model_name = "dummy-v2"
    extractor_name = "dummy"

    def __init__(self) -> None:
        self._audit = ExtractionAudit()

    def extract(self, raw_post: RawPost) -> dict[str, Any]:
        payload = ExtractionPayload(
            source_url=raw_post.url.strip() or "",
            as_of=raw_post.posted_at.date(),
            islibrary=0,
            hasview=0,
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
        self.provider_detected = detect_provider_from_base_url(self.base_url)
        self.output_mode = resolve_extraction_output_mode(self.base_url)
        self.truncated_retry_hint = False
        self.invalid_json_retry_hint = False
        self._last_parse_error_meta: dict[str, Any] = {}
        self._truncated_retry_used = False
        self._invalid_json_retry_used = False
        self._audit = ExtractionAudit()

    def _reset_parse_error_context(self) -> None:
        self._last_parse_error_meta = {}
        self._truncated_retry_used = False
        self._invalid_json_retry_used = False

    def get_last_parse_error_meta(self) -> dict[str, Any]:
        return dict(self._last_parse_error_meta)

    def extract(self, raw_post: RawPost) -> dict[str, Any]:
        self._reset_parse_error_context()
        if self.output_mode == EXTRACTION_OUTPUT_TEXT_JSON:
            try:
                model_json = self._call_openai(raw_post, response_mode=EXTRACTION_OUTPUT_TEXT_JSON)
            except TextJsonParseError as exc:
                self._last_parse_error_meta = exc.to_meta(truncated_retry_used=False, invalid_json_retry_used=False)
                if exc.parse_error_reason not in {"truncated_output", "invalid_json"}:
                    raise
                retry_reason = exc.parse_error_reason
                if retry_reason == "truncated_output":
                    self._truncated_retry_used = True
                    self.truncated_retry_hint = True
                else:
                    self._invalid_json_retry_used = True
                    self.invalid_json_retry_hint = True
                try:
                    model_json = self._call_openai(raw_post, response_mode=EXTRACTION_OUTPUT_TEXT_JSON)
                except TextJsonParseError as retry_exc:
                    self._last_parse_error_meta = retry_exc.to_meta(
                        truncated_retry_used=self._truncated_retry_used,
                        invalid_json_retry_used=self._invalid_json_retry_used,
                    )
                    if retry_reason == "invalid_json":
                        raise RuntimeError("parse_error_invalid_json_after_retry") from retry_exc
                    raise RuntimeError("parse_error_truncated_output_after_retry") from retry_exc
                finally:
                    self.truncated_retry_hint = False
                    self.invalid_json_retry_hint = False
            if self._truncated_retry_used or self._invalid_json_retry_used:
                self._last_parse_error_meta = {
                    "parse_error": False,
                    "parse_error_reason": None,
                    "truncated_retry_used": self._truncated_retry_used,
                    "invalid_json_retry_used": self._invalid_json_retry_used,
                    "finish_reason_best_effort": True,
                }
            parsed_content, raw_content, latency_ms, input_tokens, output_tokens = _coerce_call_result(model_json)
            parsed_content = _unwrap_extracted_json_envelope(parsed_content)
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
                "extraction_mode": EXTRACTION_OUTPUT_TEXT_JSON,
                "output_mode_used": EXTRACTION_OUTPUT_TEXT_JSON,
                "provider_detected": self.provider_detected,
                "parse_strategy_used": model_json.get("parse_strategy_used"),
                "raw_len": model_json.get("raw_len"),
                "repaired": bool(model_json.get("repaired")),
                "parse_error_reason": model_json.get("parse_error_reason"),
                "truncated_retry_used": bool(self._truncated_retry_used),
                "invalid_json_retry_used": bool(self._invalid_json_retry_used),
            }
            return parsed

        try:
            model_json = self._call_openai(raw_post, response_mode=EXTRACTION_OUTPUT_STRUCTURED)
            parsed_content, raw_content, latency_ms, input_tokens, output_tokens = _coerce_call_result(model_json)
            parsed_content = _unwrap_extracted_json_envelope(parsed_content)
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
                "extraction_mode": EXTRACTION_OUTPUT_STRUCTURED,
                "output_mode_used": EXTRACTION_OUTPUT_STRUCTURED,
                "provider_detected": self.provider_detected,
                "parse_strategy_used": "native_json",
                "raw_len": model_json.get("raw_len"),
                "repaired": False,
            }
            return parsed
        except OpenAIRequestError as exc:
            self._record_model_output(raw_model_output=exc.body_preview)
            if not self._is_structured_outputs_unsupported(exc):
                raise

            try:
                model_json = self._call_openai(raw_post, response_mode=EXTRACTION_OUTPUT_TEXT_JSON)
                parsed_content, raw_content, latency_ms, input_tokens, output_tokens = _coerce_call_result(model_json)
                parsed_content = _unwrap_extracted_json_envelope(parsed_content)
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
                    "extraction_mode": EXTRACTION_OUTPUT_TEXT_JSON,
                    "output_mode_used": EXTRACTION_OUTPUT_TEXT_JSON,
                    "provider_detected": self.provider_detected,
                    "parse_strategy_used": model_json.get("parse_strategy_used"),
                    "raw_len": model_json.get("raw_len"),
                    "repaired": bool(model_json.get("repaired")),
                    "fallback_reason": "structured_unsupported",
                }
                return parsed
            except Exception as retry_exc:
                raise OpenAIFallbackError(
                    f"OpenAI text_json retry failed after structured unsupported: {retry_exc}",
                    fallback_reason="structured_unsupported",
                ) from retry_exc

    def _call_openai(
        self, raw_post: RawPost, *, response_mode: Literal["structured", "text_json"]
    ) -> dict[str, Any]:
        prompt = self._build_user_prompt(raw_post)

        request_payload: dict[str, Any] = {
            "model": self.model_name,
            "temperature": 0,
            "max_tokens": self.max_output_tokens,
            "messages": [
                {"role": "user", "content": prompt},
            ],
        }
        if response_mode == EXTRACTION_OUTPUT_STRUCTURED:
            request_payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "post_extraction",
                    "strict": True,
                    "schema": EXTRACTION_JSON_SCHEMA,
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

        started = perf_counter()
        logger.info(
            "openai_request_start provider=%s output_mode=%s model=%s url=%s",
            self.provider_detected,
            response_mode,
            self.model_name,
            f"{self.base_url}/chat/completions",
        )
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=request_payload,
            )
        latency_ms = int((perf_counter() - started) * 1000)
        logger.info(
            "openai_request_end provider=%s output_mode=%s model=%s status_code=%s latency_ms=%s",
            self.provider_detected,
            response_mode,
            self.model_name,
            response.status_code,
            latency_ms,
        )

        if response.status_code >= 400:
            body_preview = response.text[:500]
            self._record_model_output(raw_model_output=body_preview, model_latency_ms=latency_ms)
            raise OpenAIRequestError(
                status_code=response.status_code,
                body_preview=body_preview,
                retry_after_seconds=_parse_retry_after_seconds(response.headers.get("Retry-After")),
            )

        body = response.json()
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("OpenAI response missing choices")

        message = choices[0].get("message", {})
        # OpenRouter may omit finish_reason on some upstreams; treat it as best-effort telemetry only.
        finish_reason = choices[0].get("finish_reason")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            refusal = message.get("refusal")
            raise RuntimeError(f"OpenAI returned empty content. refusal={refusal}")

        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")

        parse_strategy_used = "native_json"
        repaired = False
        if response_mode == EXTRACTION_OUTPUT_STRUCTURED:
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as exc:
                self._record_model_output(raw_model_output=content, model_latency_ms=latency_ms)
                raise RuntimeError(f"OpenAI content is not valid JSON: {exc}") from exc
        else:
            try:
                parsed, parse_strategy_used, repaired = parse_text_json_object(content, finish_reason=finish_reason)
            except TextJsonParseError as exc:
                self._record_model_output(raw_model_output=content, model_latency_ms=latency_ms)
                token_budget_hit = isinstance(completion_tokens, int) and completion_tokens >= self.max_output_tokens
                suspected_truncated = (
                    exc.suspected_truncated or token_budget_hit or (str(finish_reason).lower() == "length")
                )
                parse_error_reason = "truncated_output" if suspected_truncated else exc.parse_error_reason
                parse_strategy = (
                    "text_json_repair_failed_truncated" if parse_error_reason == "truncated_output" else exc.parse_strategy_used
                )
                raise TextJsonParseError(
                    "OpenAI content is not valid JSON object text",
                    raw_content=content,
                    parse_strategy_used=parse_strategy,
                    repaired=exc.repaired,
                    parse_error_reason=parse_error_reason,
                    suspected_truncated=suspected_truncated,
                    finish_reason=str(finish_reason) if finish_reason is not None else None,
                ) from exc
            except RuntimeError:
                self._record_model_output(raw_model_output=content, model_latency_ms=latency_ms)
                raise

        if not isinstance(parsed, dict):
            raise RuntimeError("OpenAI content is not a JSON object")
        parsed = _unwrap_extracted_json_envelope(parsed)
        return {
            "raw_content": content,
            "parsed_content": parsed,
            "latency_ms": latency_ms,
            "input_tokens": int(prompt_tokens) if isinstance(prompt_tokens, int) else None,
            "output_tokens": int(completion_tokens) if isinstance(completion_tokens, int) else None,
            "parse_strategy_used": parse_strategy_used,
            "raw_len": len(content),
            "repaired": repaired,
            "parse_error_reason": None,
            "finish_reason": finish_reason,
        }

    def _build_user_prompt(self, raw_post: RawPost) -> str:
        bundle = self._prompt_bundle
        if bundle and bundle.text.strip():
            return bundle.text
        return render_prompt_bundle(
            platform=raw_post.platform,
            author_handle=raw_post.author_handle,
            url=raw_post.url,
            posted_at=raw_post.posted_at,
            content_text=raw_post.content_text,
        ).text

    def _is_structured_outputs_unsupported(self, exc: OpenAIRequestError) -> bool:
        if exc.status_code != 400:
            return False

        text = exc.body_preview.lower()
        has_structured_token = "structured-outputs" in text or "structured_outputs" in text
        has_unsupported_signal = "does not support" in text or "unsupported" in text
        return has_structured_token and has_unsupported_signal


def detect_provider_from_base_url(base_url: str) -> str:
    normalized = (base_url or "").strip().lower()
    if "openrouter.ai" in normalized:
        return OPENAI_PROVIDER_OPENROUTER
    return OPENAI_PROVIDER_COMPATIBLE


def resolve_extraction_output_mode(base_url: str) -> str:
    provider = detect_provider_from_base_url(base_url)
    if provider == OPENAI_PROVIDER_OPENROUTER:
        return EXTRACTION_OUTPUT_TEXT_JSON
    return EXTRACTION_OUTPUT_STRUCTURED


def parse_text_json_object(content: str, *, finish_reason: str | None = None) -> tuple[dict[str, Any], str, bool]:
    if not isinstance(content, str):
        raise RuntimeError("OpenAI content is not text")

    cleaned = content.lstrip("\ufeff").strip()
    if not cleaned:
        raise RuntimeError("OpenAI content is empty")

    parse_candidates: list[tuple[str, str, bool]] = [(cleaned, "direct_json", False)]
    fenced = _extract_json_codeblock(cleaned)
    if fenced:
        parse_candidates.append((fenced, "strip_codeblock", True))
    outermost = _extract_outermost_json_object(cleaned)
    if outermost and outermost != cleaned:
        parse_candidates.append((outermost, "outermost_object", True))

    for candidate, strategy, repaired in parse_candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed, strategy, repaired
        raise RuntimeError("OpenAI content is not a JSON object")
    suspected_truncated = _looks_like_truncated_json_text(cleaned) or str(finish_reason).lower() == "length"
    parse_error_reason = "truncated_output" if suspected_truncated else "invalid_json"
    parse_strategy_used = "text_json_repair_failed_truncated" if suspected_truncated else "text_json_repair_failed_invalid"
    raise TextJsonParseError(
        "OpenAI content is not valid JSON object text",
        raw_content=cleaned,
        parse_strategy_used=parse_strategy_used,
        repaired=True,
        parse_error_reason=parse_error_reason,
        suspected_truncated=suspected_truncated,
        finish_reason=finish_reason,
    )


def _looks_like_truncated_json_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped[-1] in {'"', "\\", "{", "[", ":", ","}:
        return True

    obj_depth = 0
    arr_depth = 0
    in_string = False
    escape = False
    for ch in stripped:
        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            obj_depth += 1
        elif ch == "}":
            obj_depth = max(0, obj_depth - 1)
        elif ch == "[":
            arr_depth += 1
        elif ch == "]":
            arr_depth = max(0, arr_depth - 1)
    return in_string or obj_depth > 0 or arr_depth > 0


def _extract_json_codeblock(text: str) -> str | None:
    match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if match is None:
        return None
    return match.group(1).strip()


def _extract_outermost_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    first_open = -1
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                first_open = idx
            depth += 1
            continue
        if ch == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and first_open >= 0:
                return text[first_open : idx + 1].strip()
    return None


def default_extracted_json(raw_post: RawPost) -> dict[str, Any]:
    payload = ExtractionPayload(
        source_url=raw_post.url.strip() or "",
        as_of=raw_post.posted_at.date(),
        hasview=0,
        asset_views=[],
        islibrary=0,
    )
    return payload.model_dump(mode="json")


def reset_openai_call_budget_counter() -> None:
    global _OPENAI_CALLS_MADE
    with _OPENAI_CALLS_LOCK:
        _OPENAI_CALLS_MADE = 0


def try_consume_openai_call_budget(settings: Settings, *, budget_total: int | None = None) -> bool:
    budget = max(0, settings.openai_call_budget if budget_total is None else budget_total)
    if budget == 0:
        return True

    global _OPENAI_CALLS_MADE
    with _OPENAI_CALLS_LOCK:
        if _OPENAI_CALLS_MADE >= budget:
            return False
        _OPENAI_CALLS_MADE += 1
        return True


def get_openai_call_budget_remaining(settings: Settings, *, budget_total: int | None = None) -> int | None:
    budget = max(0, settings.openai_call_budget if budget_total is None else budget_total)
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
