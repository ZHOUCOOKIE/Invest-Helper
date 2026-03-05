from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
from typing import Any


MAX_ASSET_LINES = 50


EXTRACT_V1_TEMPLATE = """[InvestPulse Extraction Prompt v1]\nTask: Extract structured investment-view signals from one post.\n\nContext:\n- platform: {platform}\n- author_handle: {author_handle}\n- url: {url}\n- posted_at: {posted_at}\n- lang: {lang}\n\nReference Assets (catalog context):\n{assets_block}\n\nAlias -> Symbol Map:\n{aliases_block}\n\nPost Content:\n{content_text}\n\nChecklist (execute in order):\n1) Output structure (fixed)\n- Final output must be one JSON object with:\n  - as_of: \"YYYY-MM-DD\"\n  - source_url: string\n  - content_kind: \"asset\" | \"library\"\n  - assets: array of {{symbol, name|null, market}}\n  - asset_views: array of {{symbol, name|null, market, stance, horizon, confidence, summary}}\n  - library_entry: null or {{confidence, tags, summary}} (only when content_kind=\"library\")\n\n2) Tradable symbol scope and symbol format\n- Only keep tradable symbols explicitly supported by the post text: index/ETF/commodity/forex/crypto/stock name.\n- Do NOT use macro variables or theme words as symbol (e.g., 流动性/风险偏好/通胀/AI硬件).\n- A-share/HK stocks: symbol must be Chinese stock name/short name.\n- US stocks/crypto/ETF/index: symbol can be ticker or English name (e.g., NVDA, Nvidia, BTC, SPX, IGV).\n\n3) asset_views self-check (must pass)\n- Build asset_views first, then self-check each item confidence as int 0..100.\n- If confidence < 70: delete that item and do not output it.\n- Before final output, re-check: asset_views must contain no item with confidence < 70.\n\n4) assets must match final asset_views\n- assets must come only from final kept asset_views (deduplicated by symbol).\n- If asset_views is empty: assets must be exactly [{{\"symbol\":\"NoneAny\",\"name\":null,\"market\":\"OTHER\"}}] and asset_views must be [].\n\n5) library branch\n- If content_kind=\"library\":\n  - assets must be [{{\"symbol\":\"NoneAny\",\"name\":null,\"market\":\"OTHER\"}}]\n  - asset_views must be []\n  - library_entry must exist with tags length 1..2 from: macro, industry, thesis, strategy, risk, events\n"""


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
