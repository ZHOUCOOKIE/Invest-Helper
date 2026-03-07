from __future__ import annotations

from datetime import UTC, datetime
import json
import logging
from typing import Any

import httpx

from schemas import PortfolioAdviceAssetRead, PortfolioAdviceRequest, PortfolioAdviceResponse
from settings import get_settings

logger = logging.getLogger("uvicorn.error")


def _build_portfolio_advice_fallback(
    payload: PortfolioAdviceRequest,
    *,
    model: str,
    status: str,
    error: str | None = None,
) -> PortfolioAdviceResponse:
    asset_rows: list[PortfolioAdviceAssetRead] = []
    for item in payload.holdings:
        support_count = len(item.support_citations)
        risk_count = len(item.risk_citations)
        score = max(0, min(100, 50 + support_count * 6 - risk_count * 7 + (3 if (item.holding_reason_text or "").strip() else 0)))
        stance = "增持观察" if score >= 70 else ("减仓观察" if score <= 40 else "持有观察")
        suggestion = (
            f"当前支持观点{support_count}条、风险观点{risk_count}条，建议先按{stance}执行，"
            "并结合仓位与波动承受能力分批调整。"
        )
        evaluation = (
            f"评分{score}/100。支持证据与风险证据比例="
            f"{support_count}:{risk_count}。若新增风险证据连续增加，优先下调仓位。"
        )
        asset_rows.append(
            PortfolioAdviceAssetRead(
                asset_id=item.asset_id,
                symbol=item.symbol,
                score=score,
                stance=stance,
                suggestion=suggestion,
                evaluation=evaluation,
                key_risks=[row.summary[:120] for row in item.risk_citations[:3] if row.summary.strip()],
                key_triggers=[row.summary[:120] for row in item.support_citations[:3] if row.summary.strip()],
            )
        )
    summary = "已基于持仓理由、警惕点和卖出时机生成规则版建议。"
    if status == "skipped_no_api_key":
        summary = "未检测到 OPENAI_API_KEY，已返回规则版聚合建议。"
    elif status == "failed":
        summary = "AI聚合失败，已返回规则版聚合建议。"
    return PortfolioAdviceResponse(
        generated_at=datetime.now(UTC),
        model=model,
        status=status,
        advice_summary=summary,
        asset_advice=asset_rows,
        error=error,
    )


def _normalize_portfolio_ai_response(
    *,
    parsed: dict[str, Any],
    holdings: list[dict[str, Any]],
) -> tuple[str, list[PortfolioAdviceAssetRead]]:
    by_asset_id = {int(item["asset_id"]): item for item in holdings}
    ai_rows = parsed.get("asset_advice")
    normalized_rows: list[PortfolioAdviceAssetRead] = []
    if isinstance(ai_rows, list):
        for row in ai_rows:
            if not isinstance(row, dict):
                continue
            asset_id_raw = row.get("asset_id")
            if not isinstance(asset_id_raw, int) or asset_id_raw not in by_asset_id:
                continue
            base = by_asset_id[asset_id_raw]
            score_raw = row.get("score")
            score = int(score_raw) if isinstance(score_raw, int) else None
            if score is not None:
                score = max(0, min(100, score))
            normalized_rows.append(
                PortfolioAdviceAssetRead(
                    asset_id=asset_id_raw,
                    symbol=str(row.get("symbol") or base["symbol"]).strip()[:64] or base["symbol"],
                    score=score,
                    stance=(str(row.get("stance")).strip()[:32] if isinstance(row.get("stance"), str) else None),
                    suggestion=str(row.get("suggestion") or "").strip()[:4000] or "信息不足，维持观察。",
                    evaluation=str(row.get("evaluation") or "").strip()[:4000] or "信息不足，暂不做强结论。",
                    key_risks=[
                        str(x).strip()[:500]
                        for x in (row.get("key_risks") if isinstance(row.get("key_risks"), list) else [])
                        if str(x).strip()
                    ][:6],
                    key_triggers=[
                        str(x).strip()[:500]
                        for x in (row.get("key_triggers") if isinstance(row.get("key_triggers"), list) else [])
                        if str(x).strip()
                    ][:6],
                )
            )
    summary = str(parsed.get("advice_summary") or "").strip()[:4000] or "信息不足，建议维持观察并持续补充证据。"
    return summary, normalized_rows


async def generate_portfolio_advice(payload: PortfolioAdviceRequest) -> PortfolioAdviceResponse:
    settings = get_settings()
    api_key = settings.openai_api_key.strip()
    if not api_key:
        return _build_portfolio_advice_fallback(
            payload,
            model=settings.openai_model,
            status="skipped_no_api_key",
        )

    holdings_payload: list[dict[str, Any]] = []
    for item in payload.holdings:
        holdings_payload.append(
            {
                "asset_id": item.asset_id,
                "symbol": item.symbol.strip()[:64],
                "name": (item.name or "").strip()[:255] or None,
                "market": (item.market or "").strip()[:32] or None,
                "holding_reason_text": (item.holding_reason_text or "").strip()[:2000],
                "sell_timing_text": (item.sell_timing_text or "").strip()[:2000],
                "support_citations": [
                    {
                        "source_url": row.source_url.strip()[:1024],
                        "summary": row.summary.strip()[:600],
                        "author_handle": (row.author_handle or "").strip()[:128] or None,
                        "stance": (row.stance or "").strip()[:32] or None,
                        "horizon": (row.horizon or "").strip()[:32] or None,
                        "confidence": row.confidence,
                        "as_of": (row.as_of or "").strip()[:32] or None,
                    }
                    for row in item.support_citations[:30]
                ],
                "risk_citations": [
                    {
                        "source_url": row.source_url.strip()[:1024],
                        "summary": row.summary.strip()[:600],
                        "author_handle": (row.author_handle or "").strip()[:128] or None,
                        "stance": (row.stance or "").strip()[:32] or None,
                        "horizon": (row.horizon or "").strip()[:32] or None,
                        "confidence": row.confidence,
                        "as_of": (row.as_of or "").strip()[:32] or None,
                    }
                    for row in item.risk_citations[:30]
                ],
            }
        )

    prompt_input = {
        "task": "aggregate_portfolio_holdings_advice",
        "language": "zh-CN",
        "user_goal": (payload.user_goal or "").strip()[:1000],
        "holdings": holdings_payload,
        "output_schema": {
            "advice_summary": "string",
            "asset_advice": [
                {
                    "asset_id": "int",
                    "symbol": "string",
                    "score": "int 0..100 or null",
                    "stance": "string",
                    "suggestion": "string",
                    "evaluation": "string",
                    "key_risks": "string[]",
                    "key_triggers": "string[]",
                }
            ],
        },
        "hard_rules": [
            "只能基于输入内容，不得编造外部信息。",
            "优先引用 support_citations/risk_citations 的证据逻辑。",
            "若证据不足必须明确写出“不足”。",
            "输出必须是 JSON 对象，不要 markdown，不要额外说明。",
        ],
    }
    prompt = (
        "你是投资研究助手。请对用户当前持仓生成中文建议与评价。"
        "输出必须严格遵守 input.output_schema。输入如下：\n"
        + json.dumps(prompt_input, ensure_ascii=False)
    )

    request_payload = {
        "model": settings.openai_model,
        "temperature": 0.2,
        "max_tokens": max(300, min(2200, int(settings.openai_max_output_tokens))),
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if settings.openrouter_site_url.strip():
        headers["HTTP-Referer"] = settings.openrouter_site_url.strip()
    if settings.openrouter_app_name.strip():
        headers["X-Title"] = settings.openrouter_app_name.strip()

    try:
        async with httpx.AsyncClient(timeout=settings.openai_timeout_seconds) as client:
            response = await client.post(
                f"{settings.openai_base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=request_payload,
            )
        if response.status_code >= 400:
            return _build_portfolio_advice_fallback(
                payload,
                model=settings.openai_model,
                status="failed",
                error=response.text[:500],
            )
        body = response.json()
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("missing choices")
        message = choices[0].get("message") if isinstance(choices[0], dict) else {}
        content_text = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content_text, str) or not content_text.strip():
            raise RuntimeError("empty ai content")
        parsed = json.loads(content_text)
        if not isinstance(parsed, dict):
            raise RuntimeError("ai response is not object")
        advice_summary, asset_advice = _normalize_portfolio_ai_response(parsed=parsed, holdings=holdings_payload)
        if not asset_advice:
            return _build_portfolio_advice_fallback(
                payload,
                model=settings.openai_model,
                status="failed",
                error="ai response missing asset_advice",
            )
        return PortfolioAdviceResponse(
            generated_at=datetime.now(UTC),
            model=settings.openai_model,
            status="ok",
            advice_summary=advice_summary,
            asset_advice=asset_advice,
            error=None,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("portfolio advice ai failed")
        return _build_portfolio_advice_fallback(
            payload,
            model=settings.openai_model,
            status="failed",
            error=str(exc)[:500],
        )
