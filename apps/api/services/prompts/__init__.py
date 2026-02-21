from __future__ import annotations

from datetime import datetime
from typing import Any

from .extract_v1 import PromptBundle, render_extract_v1_prompt


def build_extract_prompt(
    *,
    prompt_version: str,
    platform: str,
    author_handle: str,
    url: str,
    posted_at: datetime,
    content_text: str,
    assets: list[dict[str, Any]],
    max_assets_in_prompt: int,
) -> PromptBundle:
    version = (prompt_version or "").strip() or "extract_v1"
    if version != "extract_v1":
        version = "extract_v1"

    bundle = render_extract_v1_prompt(
        platform=platform,
        author_handle=author_handle,
        url=url,
        posted_at=posted_at,
        content_text=content_text,
        assets=assets,
        max_assets_in_prompt=max_assets_in_prompt,
    )
    return PromptBundle(version=version, text=bundle.text, hash=bundle.hash)
