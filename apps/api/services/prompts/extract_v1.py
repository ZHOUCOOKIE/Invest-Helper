from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
from typing import Any


MAX_ASSET_LINES = 50


EXTRACT_V1_TEMPLATE = """[InvestPulse Extraction Prompt v1]\nTask: Extract structured investment-view signals from a single post.\n\nContext:\n- platform: {platform}\n- author_handle: {author_handle}\n- url: {url}\n- posted_at: {posted_at}\n- lang: {lang}\n\nReference Assets (may be empty):\n{assets_block}\n\nPost Content:\n{content_text}\n\nOutput contract:\n- Return ONLY one JSON object.\n- Prefer standardized symbols in assets[].symbol.\n- Include: reasoning, assets, event_tags, stance, horizon, confidence, summary, source_url, as_of.\n- stance in bull/bear/neutral; horizon in intraday/1w/1m/3m/1y; confidence in 0-100.\n- as_of must be date-only in YYYY-MM-DD (no time).\n"""


@dataclass
class PromptBundle:
    version: str
    text: str
    hash: str


def _infer_lang(content_text: str) -> str:
    for ch in content_text:
        if "\u4e00" <= ch <= "\u9fff":
            return "zh"
    return "en"


def _format_assets(assets: list[dict[str, Any]], limit: int) -> str:
    if not assets:
        return "- (none)"

    lines: list[str] = []
    safe_limit = max(0, min(limit, MAX_ASSET_LINES))
    for item in assets[:safe_limit]:
        symbol = str(item.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        name = str(item.get("name") or "").strip() or "null"
        market = str(item.get("market") or "").strip() or "null"
        lines.append(f"- {symbol} | name={name} | market={market}")

    return "\n".join(lines) if lines else "- (none)"


def render_extract_v1_prompt(
    *,
    platform: str,
    author_handle: str,
    url: str,
    posted_at: datetime,
    content_text: str,
    assets: list[dict[str, Any]],
    max_assets_in_prompt: int,
) -> PromptBundle:
    lang = _infer_lang(content_text)
    assets_block = _format_assets(assets, max_assets_in_prompt)
    prompt_text = EXTRACT_V1_TEMPLATE.format(
        platform=platform,
        author_handle=author_handle,
        url=url,
        posted_at=posted_at.isoformat(),
        lang=lang,
        assets_block=assets_block,
        content_text=content_text,
    )
    prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
    return PromptBundle(version="extract_v1", text=prompt_text, hash=prompt_hash)
