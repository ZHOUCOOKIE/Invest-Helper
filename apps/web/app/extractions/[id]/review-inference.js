const VALID_STANCE = new Set(["bull", "bear", "neutral"]);
const VALID_HORIZON = new Set(["intraday", "1w", "1m", "3m", "1y"]);

export function pickAssetSymbols(extracted) {
  const assets = extracted.assets;
  if (!Array.isArray(assets) || assets.length === 0) {
    return [];
  }
  const symbols = [];
  for (const item of assets) {
    const asset = item;
    if (typeof asset.symbol === "string" && asset.symbol.trim()) {
      symbols.push(asset.symbol.trim().toUpperCase());
    }
  }
  return symbols;
}

export function pickAssetViews(extracted) {
  const value = extracted.asset_views;
  if (!Array.isArray(value)) return [];
  const items = [];
  for (const raw of value) {
    if (!raw || typeof raw !== "object") continue;
    const item = raw;
    if (typeof item.symbol !== "string" || !item.symbol.trim()) continue;
    if (typeof item.stance !== "string" || !VALID_STANCE.has(item.stance)) continue;
    if (typeof item.horizon !== "string" || !VALID_HORIZON.has(item.horizon)) continue;
    if (typeof item.confidence !== "number") continue;
    items.push({
      symbol: item.symbol.trim().toUpperCase(),
      stance: item.stance,
      horizon: item.horizon,
      confidence: Math.max(0, Math.min(100, Math.round(item.confidence))),
      summary: typeof item.summary === "string" ? item.summary : null,
      as_of: typeof item.as_of === "string" ? item.as_of : null,
    });
  }
  return items.sort((a, b) => b.confidence - a.confidence);
}

export function pickDefaults(extracted, rawPostUrl) {
  const result = {
    source_url: rawPostUrl,
  };

  const assetViews = pickAssetViews(extracted);
  if (assetViews.length > 0) {
    const firstView = assetViews[0];
    result.stance = firstView.stance;
    result.horizon = firstView.horizon;
    result.confidence = String(firstView.confidence);
    if (firstView.summary) result.summary = firstView.summary;
  } else {
    const candidates = extracted.candidates;
    if (Array.isArray(candidates) && candidates.length > 0) {
      const first = candidates[0];
      if (typeof first.stance === "string" && VALID_STANCE.has(first.stance)) {
        result.stance = first.stance;
      }
      if (typeof first.horizon === "string" && VALID_HORIZON.has(first.horizon)) {
        result.horizon = first.horizon;
      }
      if (typeof first.confidence === "number") {
        result.confidence = String(Math.max(0, Math.min(100, Math.round(first.confidence))));
      }
      if (typeof first.summary === "string") result.summary = first.summary;
      if (typeof first.source_url === "string") result.source_url = first.source_url;
      if (typeof first.as_of === "string") result.as_of = first.as_of;
    }
  }

  if (typeof extracted.summary === "string" && result.summary === undefined) {
    result.summary = extracted.summary;
  }
  if (typeof extracted.source_url === "string") result.source_url = extracted.source_url;
  if (typeof extracted.as_of === "string") result.as_of = extracted.as_of;

  return result;
}

export function buildMissingInferenceHints(extracted) {
  const assetViews = pickAssetViews(extracted);
  if (assetViews.length > 0) {
    return [];
  }

  const hints = [];
  const stanceMissing = typeof extracted.stance !== "string" || !VALID_STANCE.has(extracted.stance);
  const horizonMissing = typeof extracted.horizon !== "string" || !VALID_HORIZON.has(extracted.horizon);
  const assetMissing = pickAssetSymbols(extracted).length === 0;

  if (stanceMissing) hints.push("stance: 模型未判断/信息不足");
  if (horizonMissing) hints.push("horizon: 模型未判断/信息不足");
  if (assetMissing) hints.push("asset: 模型未判断/信息不足");
  return hints;
}
