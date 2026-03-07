"use client";

import { useEffect, useMemo, useState } from "react";

type AssetItem = {
  id: number;
  symbol: string;
  name: string | null;
  market: string | null;
};

type ExtractionRawPost = {
  author_handle: string;
  url: string;
  content_text: string;
  posted_at: string;
};

type ExtractionItem = {
  id: number;
  extracted_json: Record<string, unknown>;
  raw_post: ExtractionRawPost | null;
};

type CitationItem = {
  key: string;
  extraction_id: number;
  source_url: string;
  summary: string;
  author_handle: string | null;
  stance: string | null;
  horizon: string | null;
  confidence: number | null;
  as_of: string | null;
};

type AdviceAsset = {
  asset_id: number;
  symbol: string;
  score: number | null;
  stance: string | null;
  suggestion: string;
  evaluation: string;
  key_risks: string[];
  key_triggers: string[];
};

type AdviceResponse = {
  generated_at: string;
  model: string;
  status: string;
  advice_summary: string;
  asset_advice: AdviceAsset[];
  error: string | null;
};

type HoldingEvaluation = {
  generated_at: string;
  model: string;
  status: string;
  advice_summary: string;
  asset_advice: AdviceAsset | null;
  error: string | null;
};

type HoldingItem = {
  local_id: string;
  asset_id: number;
  symbol: string;
  name: string | null;
  market: string | null;
  holding_reason_text: string;
  sell_timing_text: string;
  support_citations: CitationItem[];
  risk_citations: CitationItem[];
  ai_evaluation: HoldingEvaluation | null;
};

type CitationModalState = {
  open: boolean;
  target_holding_id: string | null;
  mode: "support" | "risk";
};

const STORAGE_KEY = "investpulse:portfolio:holdings:v1";

function safeString(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function toCitationOptions(extractions: ExtractionItem[], symbol: string): CitationItem[] {
  const options: CitationItem[] = [];
  const seen = new Set<string>();
  const symbolKey = symbol.trim().toUpperCase();
  for (const item of extractions) {
    const payload = item.extracted_json ?? {};
    const sourceUrl = safeString(payload.source_url) || (item.raw_post?.url ?? "");
    if (!sourceUrl) continue;

    const assetViewsRaw = payload.asset_views;
    const assetViews = Array.isArray(assetViewsRaw) ? assetViewsRaw : [];
    const matchedViews = assetViews.filter((row) => {
      if (!row || typeof row !== "object") return false;
      const value = safeString((row as Record<string, unknown>).symbol).toUpperCase();
      return value === symbolKey;
    });

    if (matchedViews.length > 0) {
      matchedViews.forEach((row, idx) => {
        const view = row as Record<string, unknown>;
        const summary = safeString(view.summary);
        if (!summary) return;
        const key = `${item.id}-${symbolKey}-${idx}-${summary}`;
        if (seen.has(key)) return;
        seen.add(key);
        options.push({
          key,
          extraction_id: item.id,
          source_url: sourceUrl,
          summary,
          author_handle: item.raw_post?.author_handle ?? null,
          stance: safeString(view.stance) || null,
          horizon: safeString(view.horizon) || null,
          confidence: typeof view.confidence === "number" ? view.confidence : null,
          as_of: safeString(payload.as_of) || null,
        });
      });
      continue;
    }

    const fallbackSummary = safeString(payload.summary) || safeString(item.raw_post?.content_text).slice(0, 240);
    if (!fallbackSummary) continue;
    const key = `${item.id}-${symbolKey}-fallback-${fallbackSummary}`;
    if (seen.has(key)) continue;
    seen.add(key);
    options.push({
      key,
      extraction_id: item.id,
      source_url: sourceUrl,
      summary: fallbackSummary,
      author_handle: item.raw_post?.author_handle ?? null,
      stance: null,
      horizon: null,
      confidence: null,
      as_of: safeString(payload.as_of) || null,
    });
  }
  return options;
}

export default function PortfolioPage() {
  const [assets, setAssets] = useState<AssetItem[]>([]);
  const [assetsLoading, setAssetsLoading] = useState(false);
  const [assetsError, setAssetsError] = useState<string | null>(null);
  const [assetQuery, setAssetQuery] = useState("");

  const [holdings, setHoldings] = useState<HoldingItem[]>([]);
  const [viewHoldingId, setViewHoldingId] = useState<string | null>(null);
  const [editHoldingId, setEditHoldingId] = useState<string | null>(null);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);

  const [modal, setModal] = useState<CitationModalState>({ open: false, target_holding_id: null, mode: "support" });
  const [citationQuery, setCitationQuery] = useState("");
  const [citationLoading, setCitationLoading] = useState(false);
  const [citationError, setCitationError] = useState<string | null>(null);
  const [citationOptions, setCitationOptions] = useState<CitationItem[]>([]);

  const [aiBusyHoldingId, setAiBusyHoldingId] = useState<string | null>(null);
  const [aiErrorByHoldingId, setAiErrorByHoldingId] = useState<Record<string, string>>({});

  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      const raw = window.localStorage.getItem(STORAGE_KEY);
      if (!raw) return;
      const parsed = JSON.parse(raw) as HoldingItem[];
      if (!Array.isArray(parsed)) return;
      setHoldings(
        parsed.map((item) => ({
          ...item,
          support_citations: Array.isArray(item.support_citations) ? item.support_citations : [],
          risk_citations: Array.isArray(item.risk_citations) ? item.risk_citations : [],
          ai_evaluation:
            item.ai_evaluation && typeof item.ai_evaluation === "object"
              ? item.ai_evaluation
              : null,
        })),
      );
    } catch {
      setHoldings([]);
    }
  }, []);

  useEffect(() => {
    const loadAssets = async () => {
      setAssetsLoading(true);
      setAssetsError(null);
      try {
        const res = await fetch("/api/assets", { cache: "no-store" });
        const body = (await res.json()) as AssetItem[] | { detail?: string };
        if (!res.ok || !Array.isArray(body)) {
          throw new Error("detail" in body ? (body.detail ?? "加载资产失败") : "加载资产失败");
        }
        setAssets(body);
      } catch (err) {
        setAssetsError(err instanceof Error ? err.message : "加载资产失败");
      } finally {
        setAssetsLoading(false);
      }
    };
    void loadAssets();
  }, []);

  const filteredAssets = useMemo(() => {
    const keyword = assetQuery.trim().toLowerCase();
    if (!keyword) return [];
    return assets
      .filter((item) => item.symbol.toLowerCase().includes(keyword) || (item.name || "").toLowerCase().includes(keyword))
      .slice(0, 12);
  }, [assetQuery, assets]);

  const modalHolding = useMemo(
    () => holdings.find((item) => item.local_id === modal.target_holding_id) ?? null,
    [holdings, modal.target_holding_id],
  );

  const addHolding = (asset: AssetItem) => {
    setSaveMessage(null);
    setHoldings((prev) => {
      const existed = prev.find((item) => item.asset_id === asset.id);
      if (existed) {
        setViewHoldingId(existed.local_id);
        setEditHoldingId(null);
        return prev;
      }
      const next: HoldingItem = {
        local_id: `${asset.id}-${Date.now()}`,
        asset_id: asset.id,
        symbol: asset.symbol,
        name: asset.name,
        market: asset.market,
        holding_reason_text: "",
        sell_timing_text: "",
        support_citations: [],
        risk_citations: [],
        ai_evaluation: null,
      };
      setViewHoldingId(next.local_id);
      setEditHoldingId(null);
      return [...prev, next];
    });
  };

  const removeHolding = (localId: string) => {
    setSaveMessage(null);
    setHoldings((prev) => prev.filter((item) => item.local_id !== localId));
    if (viewHoldingId === localId) setViewHoldingId(null);
    if (editHoldingId === localId) setEditHoldingId(null);
  };

  const updateHolding = (localId: string, patch: Partial<HoldingItem>) => {
    setSaveMessage(null);
    setHoldings((prev) => prev.map((item) => (item.local_id === localId ? { ...item, ...patch } : item)));
  };

  const saveHoldings = () => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(holdings));
    setSaveMessage(`已保存 ${new Date().toLocaleString()}`);
  };

  const openCitationModal = (holding: HoldingItem, mode: "support" | "risk") => {
    setModal({ open: true, target_holding_id: holding.local_id, mode });
    setCitationQuery(holding.symbol);
    setCitationOptions([]);
    setCitationError(null);
  };

  const closeCitationModal = () => {
    setModal({ open: false, target_holding_id: null, mode: "support" });
    setCitationQuery("");
    setCitationOptions([]);
    setCitationError(null);
  };

  const runCitationSearch = async () => {
    if (!modalHolding) return;
    setCitationLoading(true);
    setCitationError(null);
    try {
      const q = citationQuery.trim() || modalHolding.symbol;
      const params = new URLSearchParams({ status: "all", q, limit: "40", offset: "0" });
      const res = await fetch(`/api/extractions?${params.toString()}`, { cache: "no-store" });
      const body = (await res.json()) as ExtractionItem[] | { detail?: string };
      if (!res.ok || !Array.isArray(body)) {
        throw new Error("detail" in body ? (body.detail ?? "搜索观点失败") : "搜索观点失败");
      }
      setCitationOptions(toCitationOptions(body, modalHolding.symbol));
    } catch (err) {
      setCitationError(err instanceof Error ? err.message : "搜索观点失败");
      setCitationOptions([]);
    } finally {
      setCitationLoading(false);
    }
  };

  const addCitationToHolding = (citation: CitationItem) => {
    if (!modalHolding) return;
    setSaveMessage(null);
    setHoldings((prev) =>
      prev.map((item) => {
        if (item.local_id !== modalHolding.local_id) return item;
        if (modal.mode === "support") {
          if (item.support_citations.some((row) => row.key === citation.key)) return item;
          return { ...item, support_citations: [citation, ...item.support_citations] };
        }
        if (item.risk_citations.some((row) => row.key === citation.key)) return item;
        return { ...item, risk_citations: [citation, ...item.risk_citations] };
      }),
    );
  };

  const removeCitation = (holding: HoldingItem, mode: "support" | "risk", key: string) => {
    setSaveMessage(null);
    if (mode === "support") {
      updateHolding(holding.local_id, { support_citations: holding.support_citations.filter((item) => item.key !== key) });
      return;
    }
    updateHolding(holding.local_id, { risk_citations: holding.risk_citations.filter((item) => item.key !== key) });
  };

  const evaluateSingleHolding = async (holding: HoldingItem) => {
    setAiBusyHoldingId(holding.local_id);
    setAiErrorByHoldingId((prev) => ({ ...prev, [holding.local_id]: "" }));
    try {
      const payload = {
        holdings: [
          {
            asset_id: holding.asset_id,
            symbol: holding.symbol,
            name: holding.name,
            market: holding.market,
            holding_reason_text: holding.holding_reason_text,
            sell_timing_text: holding.sell_timing_text,
            support_citations: holding.support_citations.map((row) => ({
              extraction_id: row.extraction_id,
              source_url: row.source_url,
              summary: row.summary,
              author_handle: row.author_handle,
              stance: row.stance,
              horizon: row.horizon,
              confidence: row.confidence,
              as_of: row.as_of,
            })),
            risk_citations: holding.risk_citations.map((row) => ({
              extraction_id: row.extraction_id,
              source_url: row.source_url,
              summary: row.summary,
              author_handle: row.author_handle,
              stance: row.stance,
              horizon: row.horizon,
              confidence: row.confidence,
              as_of: row.as_of,
            })),
          },
        ],
      };
      const res = await fetch("/api/portfolio/advice", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const body = (await res.json()) as AdviceResponse | { detail?: string };
      if (!res.ok) {
        throw new Error("detail" in body ? (body.detail ?? "AI评价失败") : "AI评价失败");
      }
      const okBody = body as AdviceResponse;
      const singleAssetAdvice = okBody.asset_advice.find((item) => item.asset_id === holding.asset_id) ?? okBody.asset_advice[0] ?? null;
      updateHolding(holding.local_id, {
        ai_evaluation: {
          generated_at: okBody.generated_at,
          model: okBody.model,
          status: okBody.status,
          advice_summary: okBody.advice_summary,
          asset_advice: singleAssetAdvice,
          error: okBody.error,
        },
      });
      setViewHoldingId(holding.local_id);
      setEditHoldingId(null);
    } catch (err) {
      setAiErrorByHoldingId((prev) => ({
        ...prev,
        [holding.local_id]: err instanceof Error ? err.message : "AI评价失败",
      }));
    } finally {
      setAiBusyHoldingId(null);
    }
  };

  return (
    <main style={{ padding: "24px", fontFamily: "monospace", display: "grid", gap: "14px" }}>
      <h1>当前持仓资产</h1>

      <section style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "12px", display: "grid", gap: "10px" }}>
        <div style={{ display: "flex", gap: "10px", alignItems: "center", flexWrap: "wrap" }}>
          <span>先搜索并加入持仓。点击具体资产可查看只读详情，点编辑可修改并做单资产AI评价。</span>
        </div>
        <div style={{ display: "flex", gap: "8px", alignItems: "center", flexWrap: "wrap" }}>
          <input
            value={assetQuery}
            onChange={(event) => setAssetQuery(event.target.value)}
            placeholder="搜索资产代码/名称"
            style={{ minWidth: "260px" }}
          />
          {assetsLoading && <span>资产加载中...</span>}
          {assetsError && <span style={{ color: "crimson" }}>{assetsError}</span>}
        </div>
        <div style={{ display: "grid", gap: "6px" }}>
          {!assetQuery.trim() && <small>请输入关键词后开始搜索资产。</small>}
          {filteredAssets.map((asset) => (
            <div key={asset.id} style={{ border: "1px solid #eee", borderRadius: "6px", padding: "6px 8px", display: "flex", justifyContent: "space-between", gap: "8px" }}>
              <div>
                <strong>{asset.symbol}</strong> {asset.name ? `- ${asset.name}` : ""} {asset.market ? `(${asset.market})` : ""}
              </div>
              <button type="button" onClick={() => addHolding(asset)}>加入持仓</button>
            </div>
          ))}
          {!assetsLoading && !!assetQuery.trim() && filteredAssets.length === 0 && <small>未找到匹配资产。</small>}
        </div>
      </section>

      <section style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "12px", display: "grid", gap: "10px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: "8px" }}>
          <h2 style={{ margin: 0 }}>我的持仓清单</h2>
          <button type="button" onClick={saveHoldings} disabled={holdings.length === 0}>保存持仓信息</button>
        </div>
        {saveMessage && <small style={{ color: "#0b6" }}>{saveMessage}</small>}
        {holdings.length === 0 && <p>暂未添加持仓资产。</p>}
        {holdings.map((holding) => {
          const isViewing = viewHoldingId === holding.local_id;
          const isEditing = editHoldingId === holding.local_id;
          const aiError = aiErrorByHoldingId[holding.local_id] || "";
          return (
            <article key={holding.local_id} style={{ border: "1px solid #eee", borderRadius: "8px", padding: "10px", display: "grid", gap: "8px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", gap: "8px", alignItems: "center" }}>
                <button
                  type="button"
                  onClick={() => {
                    setViewHoldingId(isViewing ? null : holding.local_id);
                    setEditHoldingId(null);
                  }}
                  style={{ textAlign: "left", background: "transparent", border: "none", padding: 0, cursor: "pointer" }}
                  title="查看详情"
                >
                  <strong>{holding.symbol}</strong> {holding.name ? `- ${holding.name}` : ""} {holding.market ? `(${holding.market})` : ""}
                </button>
                <div style={{ display: "flex", gap: "8px" }}>
                  <button
                    type="button"
                    onClick={() => {
                      setEditHoldingId(isEditing ? null : holding.local_id);
                      setViewHoldingId(null);
                    }}
                  >
                    {isEditing ? "收起编辑" : "编辑"}
                  </button>
                  <button type="button" onClick={() => removeHolding(holding.local_id)}>移除</button>
                </div>
              </div>

              {isViewing && (
                <div style={{ border: "1px solid #f0f0f0", borderRadius: "6px", padding: "8px", display: "grid", gap: "8px" }}>
                  <div>
                    <strong>当前持仓理由：</strong>
                    <div style={{ marginTop: "4px", whiteSpace: "pre-wrap" }}>{holding.holding_reason_text || "（未填写）"}</div>
                  </div>
                  <div>
                    <strong>卖出时机：</strong>
                    <div style={{ marginTop: "4px", whiteSpace: "pre-wrap" }}>{holding.sell_timing_text || "（未填写）"}</div>
                  </div>
                  <div>
                    <strong>持仓理由引用观点：</strong>
                    {holding.support_citations.length === 0 ? (
                      <div style={{ marginTop: "4px" }}>（无）</div>
                    ) : (
                      <div style={{ marginTop: "4px", display: "grid", gap: "6px" }}>
                        {holding.support_citations.map((item) => (
                          <div key={item.key} style={{ border: "1px solid #eee", borderRadius: "6px", padding: "6px" }}>
                            <div>{item.summary}</div>
                            <a href={item.source_url} target="_blank" rel="noreferrer">{item.source_url}</a>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                  <div>
                    <strong>需要警惕的点（引用观点）：</strong>
                    {holding.risk_citations.length === 0 ? (
                      <div style={{ marginTop: "4px" }}>（无）</div>
                    ) : (
                      <div style={{ marginTop: "4px", display: "grid", gap: "6px" }}>
                        {holding.risk_citations.map((item) => (
                          <div key={item.key} style={{ border: "1px solid #eee", borderRadius: "6px", padding: "6px" }}>
                            <div>{item.summary}</div>
                            <a href={item.source_url} target="_blank" rel="noreferrer">{item.source_url}</a>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                  <div>
                    <strong>AI评价：</strong>
                    {!holding.ai_evaluation ? (
                      <div style={{ marginTop: "4px" }}>（暂无）</div>
                    ) : (
                      <div style={{ marginTop: "4px", display: "grid", gap: "6px" }}>
                        <div>
                          状态: {holding.ai_evaluation.status} | 模型: {holding.ai_evaluation.model}
                        </div>
                        <div>{holding.ai_evaluation.advice_summary}</div>
                        {holding.ai_evaluation.asset_advice && (
                          <div style={{ border: "1px solid #eee", borderRadius: "6px", padding: "6px", display: "grid", gap: "4px" }}>
                            <div>
                              <strong>{holding.ai_evaluation.asset_advice.symbol}</strong>
                              {holding.ai_evaluation.asset_advice.stance ? ` · ${holding.ai_evaluation.asset_advice.stance}` : ""}
                              {typeof holding.ai_evaluation.asset_advice.score === "number" ? ` · 评分${holding.ai_evaluation.asset_advice.score}` : ""}
                            </div>
                            <div><strong>建议:</strong> {holding.ai_evaluation.asset_advice.suggestion}</div>
                            <div><strong>评价:</strong> {holding.ai_evaluation.asset_advice.evaluation}</div>
                            <div><strong>关键风险:</strong> {holding.ai_evaluation.asset_advice.key_risks.join("；") || "无"}</div>
                            <div><strong>关键信号:</strong> {holding.ai_evaluation.asset_advice.key_triggers.join("；") || "无"}</div>
                          </div>
                        )}
                        {holding.ai_evaluation.error && <div style={{ color: "crimson" }}>错误: {holding.ai_evaluation.error}</div>}
                      </div>
                    )}
                  </div>
                </div>
              )}

              {isEditing && (
                <div style={{ display: "grid", gap: "8px" }}>
                  <label style={{ display: "grid", gap: "4px" }}>
                    <span>当前持仓理由（手写）</span>
                    <textarea
                      value={holding.holding_reason_text}
                      onChange={(event) => updateHolding(holding.local_id, { holding_reason_text: event.target.value })}
                      rows={3}
                      placeholder="写下你当前继续持有的核心理由"
                    />
                  </label>

                  <label style={{ display: "grid", gap: "4px" }}>
                    <span>卖出时机（手写）</span>
                    <textarea
                      value={holding.sell_timing_text}
                      onChange={(event) => updateHolding(holding.local_id, { sell_timing_text: event.target.value })}
                      rows={3}
                      placeholder="例如：达到估值目标、跌破纪律位、基本面拐点失效"
                    />
                  </label>

                  <div style={{ border: "1px solid #f0f0f0", borderRadius: "6px", padding: "8px", display: "grid", gap: "6px" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", gap: "8px", alignItems: "center" }}>
                      <strong>持仓理由引用观点</strong>
                      <button type="button" onClick={() => openCitationModal(holding, "support")}>添加观点</button>
                    </div>
                    {holding.support_citations.length === 0 && <small>暂无引用观点。</small>}
                    {holding.support_citations.map((item) => (
                      <div key={item.key} style={{ border: "1px solid #eee", borderRadius: "6px", padding: "6px", display: "grid", gap: "4px" }}>
                        <div>{item.summary}</div>
                        <a href={item.source_url} target="_blank" rel="noreferrer">{item.source_url}</a>
                        <button type="button" onClick={() => removeCitation(holding, "support", item.key)}>删除观点</button>
                      </div>
                    ))}
                  </div>

                  <div style={{ border: "1px solid #f0f0f0", borderRadius: "6px", padding: "8px", display: "grid", gap: "6px" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", gap: "8px", alignItems: "center" }}>
                      <strong>需要警惕的点（引用观点）</strong>
                      <button type="button" onClick={() => openCitationModal(holding, "risk")}>添加观点</button>
                    </div>
                    {holding.risk_citations.length === 0 && <small>暂无引用观点。</small>}
                    {holding.risk_citations.map((item) => (
                      <div key={item.key} style={{ border: "1px solid #eee", borderRadius: "6px", padding: "6px", display: "grid", gap: "4px" }}>
                        <div>{item.summary}</div>
                        <a href={item.source_url} target="_blank" rel="noreferrer">{item.source_url}</a>
                        <button type="button" onClick={() => removeCitation(holding, "risk", item.key)}>删除观点</button>
                      </div>
                    ))}
                  </div>

                  <button type="button" onClick={() => void evaluateSingleHolding(holding)} disabled={aiBusyHoldingId === holding.local_id}>
                    {aiBusyHoldingId === holding.local_id ? "AI评价中..." : "AI模型评价该资产"}
                  </button>
                  {aiError && <p style={{ color: "crimson" }}>{aiError}</p>}
                </div>
              )}
            </article>
          );
        })}
      </section>

      {modal.open && modalHolding && (
        <section style={{ position: "fixed", inset: "0", background: "rgba(0,0,0,0.35)", display: "grid", placeItems: "center", padding: "16px", zIndex: 1000 }}>
          <div style={{ background: "var(--bg-elev)", border: "1px solid #ddd", borderRadius: "10px", padding: "12px", width: "min(920px, 100%)", maxHeight: "85vh", overflow: "auto", display: "grid", gap: "8px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: "8px" }}>
              <strong>{modal.mode === "support" ? "添加持仓理由观点" : "添加警惕点观点"} · {modalHolding.symbol}</strong>
              <button type="button" onClick={closeCitationModal}>关闭</button>
            </div>
            <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
              <input value={citationQuery} onChange={(event) => setCitationQuery(event.target.value)} placeholder="输入关键词搜索观点" style={{ flex: 1 }} />
              <button type="button" onClick={() => void runCitationSearch()} disabled={citationLoading}>{citationLoading ? "搜索中..." : "搜索"}</button>
            </div>
            {citationError && <p style={{ color: "crimson" }}>{citationError}</p>}
            {citationOptions.length === 0 && !citationLoading && <small>暂无结果，试试不同关键词。</small>}
            <div style={{ display: "grid", gap: "8px" }}>
              {citationOptions.map((item) => (
                <article key={item.key} style={{ border: "1px solid #eee", borderRadius: "8px", padding: "8px", display: "grid", gap: "4px" }}>
                  <div>{item.summary}</div>
                  <small>
                    {item.author_handle ? `@${item.author_handle}` : "匿名"}
                    {item.stance ? ` · ${item.stance}` : ""}
                    {item.horizon ? ` · ${item.horizon}` : ""}
                    {typeof item.confidence === "number" ? ` · ${item.confidence}` : ""}
                  </small>
                  <a href={item.source_url} target="_blank" rel="noreferrer">{item.source_url}</a>
                  <button type="button" onClick={() => addCitationToHolding(item)}>添加到当前持仓</button>
                </article>
              ))}
            </div>
          </div>
        </section>
      )}
    </main>
  );
}
