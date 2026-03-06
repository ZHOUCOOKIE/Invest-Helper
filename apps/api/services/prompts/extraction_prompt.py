from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib

EXTRACT_V1_TEMPLATE = """[InvestPulse Extraction Prompt]

Task: Extract structured investment signals from ONE post. Return exactly ONE JSON object and nothing else.

Input:

platform: {platform}

author_handle: {author_handle}

url: {url}

posted_at: {posted_at}

Post Content:
{content_text}

Hard rules (must follow):

Output MUST be exactly one JSON object. No markdown, no code fences, no extra text.

Top-level fields MUST be exactly:
as_of, source_url, islibrary, hasview, asset_views, library_entry

JSON schema (fixed):
{
"as_of": "YYYY-MM-DD",
"source_url": "string",
"islibrary": 0,
"hasview": 0,
"asset_views": [
{
"symbol": "string",
"market": "string",
"stance": "bull|bear|neutral",
"horizon": "intraday|1w|1m|3m|1y",
"confidence": 80,
"summary": "string"
}
],
"library_entry": {
  "tag": "macro|industry|thesis|strategy|risk|events",
  "summary": "测试"
}
}

Field constraints:

as_of: must be YYYY-MM-DD.

source_url: must equal the input url exactly.

islibrary: must be integer 0 or 1.

hasview: must be integer 0 or 1.

market: MUST be one of CRYPTO, STOCK, ETF, FOREX, OTHER.

stance: must be one of bull, bear, neutral.
Stance rule: Prefer the author’s explicit stance. If the post describes an event/trend and you map it to a likely impacted tradable target, infer stance only when the direction is strongly implied by the post; otherwise without a clear one-way forecast, set stance to neutral.

horizon: must be one of intraday, 1w, 1m, 3m, 1y.
Horizon rule: Prefer the author’s explicit time point / holding window / expected duration. If none is stated, choose the most reliable horizon implied by the content (nearest reasonable window).

summary: MUST be Chinese (only validate summary language) and concise (avoid long explanations; keep it short).

confidence: MUST be integer 80..100.
Meaning: confidence is your certainty that the post content meaningfully impacts or is strongly associated with THIS asset (including plausible, directly-impacted targets) given the event/trend described.

85..100: clearly about a specific asset / direct event / concrete claim.

80..85: broader but still very likely impacts the asset meaningfully.
If you think confidence < 80 for an asset: DO NOT output that asset at all.

Extraction rules:

You may include asset_views for (a) assets explicitly mentioned OR (b) assets that are very likely impacted by the described event/trend, as long as your confidence is >= 80.

Only set hasview=1 and output an asset_views item if the post makes a real directional investment claim about that specific asset (explicitly or strongly implied).

Do NOT treat hypothetical examples, educational lists, illustrative content, or posts that are mainly sarcasm / moral judgment / rhetorical questions / memes as a directional investment claim. In all such cases, set hasview=0 and output asset_views: [].

Symbol rules:

CN/HK stocks: symbol MUST be the full Chinese stock name.

US/overseas stocks: symbol MUST be a tradable ticker.

ETF/Index: allow common tickers/symbols.

CRYPTO: use standard symbols/pairs.

FOREX/Commodities: prefer standard codes / tradable symbols.

Example symbols: "贵州茅台", "NVDA", "SPX", "IGV", "BTC", "XAUUSD", "WTI"

Library branch:

If islibrary=0: library_entry must be null.

If islibrary=1:

islibrary=1 MUST mean the post is highly valuable and worth saving for repeated reading (deep insights, strong analysis, tight reasoning chain). Use your own strict judgment.

library_entry must include tag(only 1), summary(Chinese)

tag must come from: macro, industry, thesis, strategy, risk, events

IMPORTANT: library_entry.summary must be exactly "测试" (and nothing else).

Now output the final JSON object only.
"""


@dataclass
class PromptBundle:
    version: str
    text: str
    hash: str

def render_prompt_bundle(
    *,
    platform: str,
    author_handle: str,
    url: str,
    posted_at: datetime,
    content_text: str,
) -> PromptBundle:
    prompt_text = (
        EXTRACT_V1_TEMPLATE.replace("{platform}", platform)
        .replace("{author_handle}", author_handle)
        .replace("{url}", url)
        .replace("{posted_at}", posted_at.isoformat())
        .replace("{content_text}", content_text)
    )
    prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
    return PromptBundle(version="extract_v1", text=prompt_text, hash=prompt_hash)
