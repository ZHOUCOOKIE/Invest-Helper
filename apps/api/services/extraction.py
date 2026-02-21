from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from models import RawPost


class Extractor(ABC):
    model_name: str

    @abstractmethod
    def extract(self, raw_post: RawPost) -> dict[str, Any]:
        """Return structured candidates inferred from a raw post."""


class DummyExtractor(Extractor):
    model_name = "dummy-v1"

    def extract(self, raw_post: RawPost) -> dict[str, Any]:
        content = raw_post.content_text.strip()
        return {
            "raw_post_id": raw_post.id,
            "platform": raw_post.platform,
            "url": raw_post.url,
            "candidates": [
                {
                    "asset_symbol": None,
                    "stance": "neutral",
                    "horizon": "1w",
                    "confidence": 50,
                    "summary": content[:280] if content else "",
                }
            ],
        }
