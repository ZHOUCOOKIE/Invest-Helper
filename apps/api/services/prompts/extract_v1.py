from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
from typing import Any


MAX_ASSET_LINES = 50


EXTRACT_V1_TEMPLATE = """[InvestPulse Extraction Prompt v1]\nTask: Extract structured investment-view signals from a single post.\n\nContext:\n- platform: {platform}\n- author_handle: {author_handle}\n- url: {url}\n- posted_at: {posted_at}\n- lang: {lang}\n\nReference Assets (may be empty):\n{assets_block}\n\nAlias -> Symbol Map (may be empty):\n{aliases_block}\n\nPost Content:\n{content_text}\n\nOutput contract:\n- Return ONLY one JSON object. No markdown, no code fences, no explanation text.\n- Must include asset_views as primary output for persistence/review.\n- Every asset_views[] item must include symbol/stance/horizon/confidence/reasoning/summary.\n- extracted_json.reasoning must be Chinese. Even if post content is English, explain in Chinese.\n- reasoning may keep necessary proper nouns/tickers/URLs, but sentence body must be Chinese.\n- asset_views[].symbol must prioritize Alias -> Symbol map and Reference Assets symbols.\n- asset_views[].symbol must be non-empty. If symbol is uncertain, do not output that asset_view item.\n- Do not invent new symbols. Use only symbols from Alias -> Symbol map or Reference Assets when possible.\n- You may output multiple asset views.\n- Keep global fields for compatibility: reasoning, assets, event_tags, stance, horizon, confidence, summary, source_url, as_of.\n- assets must always exist at top-level.\n- If the post has no investment asset/view target, must output assets exactly as [\"NoneAny\"] (do not use empty assets for this case).\n- When assets is [\"NoneAny\"], must also output asset_views as [], and stance should be neutral.\n- stance in bull/bear/neutral; horizon in intraday/1w/1m/3m/1y; confidence in integer 0-100.\n- Confidence means impact relevance (not truth/probability): 0-30 barely related, 31-60 weak impact, 61-80 direct medium impact, 81-100 strong impact.\n- as_of must be date-only in YYYY-MM-DD (no time).\n"""


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


def _format_aliases(aliases: list[dict[str, str]], limit: int) -> str:
    if not aliases:
        return "- (none)"

    lines: list[str] = []
    safe_limit = max(0, min(limit, 200))
    for item in aliases[:safe_limit]:
        alias = str(item.get("alias") or "").strip()
        symbol = str(item.get("symbol") or "").strip().upper()
        if not alias or not symbol:
            continue
        lines.append(f"- {alias} -> {symbol}")
    return "\n".join(lines) if lines else "- (none)"


def render_extract_v1_prompt(
    *,
    platform: str,
    author_handle: str,
    url: str,
    posted_at: datetime,
    content_text: str,
    assets: list[dict[str, Any]],
    aliases: list[dict[str, str]],
    max_assets_in_prompt: int,
) -> PromptBundle:
    lang = _infer_lang(content_text)
    assets_block = _format_assets(assets, max_assets_in_prompt)
    aliases_block = _format_aliases(aliases, max_assets_in_prompt * 4)
    prompt_text = EXTRACT_V1_TEMPLATE.format(
        platform=platform,
        author_handle=author_handle,
        url=url,
        posted_at=posted_at.isoformat(),
        lang=lang,
        assets_block=assets_block,
        aliases_block=aliases_block,
        content_text=content_text,
    )
    prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
    return PromptBundle(version="extract_v1", text=prompt_text, hash=prompt_hash)
