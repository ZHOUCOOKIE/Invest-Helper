from __future__ import annotations

from datetime import datetime

from .extraction_prompt import PromptBundle, render_prompt_bundle


def build_extract_prompt(
    *,
    prompt_version: str,
    platform: str,
    author_handle: str,
    url: str,
    posted_at: datetime,
    content_text: str,
) -> PromptBundle:
    version = (prompt_version or "").strip() or "extract_v1"
    if version != "extract_v1":
        version = "extract_v1"

    bundle = render_prompt_bundle(
        platform=platform,
        author_handle=author_handle,
        url=url,
        posted_at=posted_at,
        content_text=content_text,
    )
    return PromptBundle(version=version, text=bundle.text, hash=bundle.hash)
