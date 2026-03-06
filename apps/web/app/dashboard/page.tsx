"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";

type DashboardAssetLatestView = {
  kol_view_id: number;
  horizon: "intraday" | "1w" | "1m" | "3m" | "1y" | string;
  stance: "bull" | "bear" | "neutral" | string;
  confidence: number;
  summary: string;
  as_of: string;
  created_at: string;
  kol_id: number;
  kol_display_name: string | null;
  kol_handle: string | null;
};

type DashboardAsset = {
  id: number;
  symbol: string;
  name: string | null;
  market: string | null;
  new_views_24h: number;
  new_views_7d: number;
  latest_views_by_horizon: DashboardAssetLatestView[];
};

type DashboardActiveKol = {
  kol_id: number;
  display_name: string | null;
  handle: string;
  platform: string;
  views_count_7d: number;
  top_assets: Array<{
    asset_id: number;
    symbol: string;
    views_count: number;
  }>;
};

type DashboardClarityContributor = {
  handle: string;
  contribution: number;
};

type DashboardClarityRankingItem = {
  asset_id: number;
  symbol: string;
  name: string | null;
  market: string | null;
  direction: "bull" | "bear" | "neutral" | string;
  s_raw: number;
  clarity_score: number;
  n: number;
  k: number;
  bull_count: number;
  bear_count: number;
  total_count: number;
  top_contributors: DashboardClarityContributor[];
};

type DashboardResponse = {
  pending_extractions_count: number;
  new_views_24h: number;
  new_views_7d: number;
  total_assets_count: number;
  assets: DashboardAsset[];
  active_kols_7d: DashboardActiveKol[];
  clarity_ranking: DashboardClarityRankingItem[];
};

type WindowKey = "24h" | "7d";

const horizonRank: Record<string, number> = {
  intraday: 0,
  "1w": 1,
  "1m": 2,
  "3m": 3,
  "1y": 4,
};

function horizonLabel(horizon: string): string {
  const map: Record<string, string> = {
    intraday: "日内",
    "1w": "1周",
    "1m": "1月",
    "3m": "3月",
    "1y": "1年",
  };
  return map[horizon] ?? horizon;
}

function stanceLabel(stance: string): string {
  const map: Record<string, string> = {
    bull: "看涨",
    bear: "看跌",
    neutral: "中性",
  };
  return map[stance] ?? stance;
}

function pickLatestViews(asset: DashboardAsset, limit = 3): DashboardAssetLatestView[] {
  return [...asset.latest_views_by_horizon]
    .sort((a, b) => {
      const aTimeRaw = a.as_of || a.created_at;
      const bTimeRaw = b.as_of || b.created_at;
      if (aTimeRaw !== bTimeRaw) return bTimeRaw.localeCompare(aTimeRaw);
      const aRank = horizonRank[a.horizon] ?? 999;
      const bRank = horizonRank[b.horizon] ?? 999;
      if (aRank !== bRank) return aRank - bRank;
      return b.kol_view_id - a.kol_view_id;
    })
    .slice(0, Math.max(1, limit));
}

function cmpByWindow(a: DashboardAsset, b: DashboardAsset, windowKey: WindowKey): number {
  const delta =
    windowKey === "24h" ? b.new_views_24h - a.new_views_24h : b.new_views_7d - a.new_views_7d;
  if (delta !== 0) return delta;
  return a.symbol.localeCompare(b.symbol);
}

export default function DashboardPage() {
  const [data, setData] = useState<DashboardResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const windowKey: WindowKey = "24h";
  const [clarityWindow, setClarityWindow] = useState<WindowKey>("24h");
  const [showAllAssets, setShowAllAssets] = useState(false);
  const todayDigestDate = new Date().toISOString().slice(0, 10);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({
        days: "7",
        window: clarityWindow,
        limit: "10",
        assets_window: windowKey,
      });
      if (showAllAssets) {
        params.set("show_all_assets", "true");
      } else {
        params.set("assets_limit", "15");
      }
      const res = await fetch(`/api/dashboard?${params.toString()}`, { cache: "no-store" });
      const body = (await res.json()) as DashboardResponse | { detail?: string };
      if (!res.ok) {
        throw new Error("detail" in body ? (body.detail ?? "Load failed") : "Load failed");
      }
      setData(body as DashboardResponse);
    } catch (err) {
      setError(err instanceof Error ? err.message : "未知错误");
    } finally {
      setLoading(false);
    }
  }, [clarityWindow, showAllAssets, windowKey]);

  useEffect(() => {
    void load();
  }, [load]);

  const visibleAssets = useMemo(() => {
    const assets = data?.assets ?? [];
    const keyword = query.trim().toLowerCase();
    const filtered = keyword
      ? assets.filter((asset) => {
          const symbol = asset.symbol.toLowerCase();
          const name = (asset.name || "").toLowerCase();
          return symbol.includes(keyword) || name.includes(keyword);
        })
      : assets;
    return [...filtered].sort((a, b) => cmpByWindow(a, b, windowKey));
  }, [data, query, windowKey]);

  return (
    <main style={{ padding: "24px", fontFamily: "monospace", display: "grid", gap: "14px" }}>
      <h1>投资组合看板</h1>

      <section style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "12px" }}>
        <div style={{ display: "flex", gap: "12px", alignItems: "center", flexWrap: "wrap" }}>
          <strong>新增观点: {windowKey === "24h" ? (data?.new_views_24h ?? "-") : (data?.new_views_7d ?? "-")}</strong>
          <span>待处理抽取: {data?.pending_extractions_count ?? "-"}</span>
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="按资产代码/名称搜索"
            style={{ minWidth: "240px" }}
          />
          <button type="button" onClick={() => void load()} disabled={loading}>
            {loading ? "加载中..." : "刷新"}
          </button>
          <Link href="/ingest">手动导入</Link>
          <Link href="/kols">KOL管理</Link>
          <Link href="/extractions">审核队列</Link>
          <Link href={`/digests/${todayDigestDate}`}>今日日报</Link>
        </div>
      </section>

      {error && <p style={{ color: "crimson" }}>{error}</p>}

      {!loading && !error && (
        <>
          <section style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "12px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", gap: "8px", alignItems: "center" }}>
              <h2 style={{ marginTop: 0, marginBottom: "8px" }}>交易清晰度排行</h2>
              <div style={{ display: "flex", gap: "8px" }}>
                <button type="button" onClick={() => setClarityWindow("24h")} disabled={clarityWindow === "24h"}>
                  24h
                </button>
                <button type="button" onClick={() => setClarityWindow("7d")} disabled={clarityWindow === "7d"}>
                  7d
                </button>
              </div>
            </div>
            {(!data?.clarity_ranking || data.clarity_ranking.length === 0) && <p>暂无清晰度数据。</p>}
            {data?.clarity_ranking && data.clarity_ranking.length > 0 && (
              <div style={{ display: "grid", gap: "6px" }}>
                {data.clarity_ranking.map((item, index) => (
                  <Link
                    key={item.asset_id}
                    href={`/assets/${item.asset_id}`}
                    style={{
                      border: "1px solid #eee",
                      borderRadius: "8px",
                      padding: "8px 10px",
                      textDecoration: "none",
                      color: "inherit",
                      display: "grid",
                      gridTemplateColumns: "40px 1fr auto auto auto",
                      gap: "8px",
                      alignItems: "center",
                    }}
                  >
                    <strong>#{index + 1}</strong>
                    <span>
                      <strong>{item.symbol}</strong>
                      {item.name ? ` - ${item.name}` : ""}
                    </span>
                    <span
                      style={{
                        color: item.direction === "bull" ? "green" : item.direction === "bear" ? "crimson" : "#666",
                      }}
                    >
                      {item.direction === "bull" ? "看涨" : item.direction === "bear" ? "看跌" : "中性"}
                    </span>
                    <span>清晰度: {item.clarity_score.toFixed(4)}</span>
                    <span>看涨/看跌/总数: {item.bull_count}/{item.bear_count}/{item.total_count || item.n}</span>
                  </Link>
                ))}
              </div>
            )}
          </section>

          <section style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "12px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: "8px" }}>
              <h2 style={{ marginTop: 0, marginBottom: "8px" }}>
                资产列表 {showAllAssets ? "(全部)" : "(前 15)"}
              </h2>
              <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
                {!showAllAssets && (data?.total_assets_count ?? 0) > 15 && (
                  <button type="button" onClick={() => setShowAllAssets(true)}>
                    查看更多资产（共{data?.total_assets_count ?? 0}）
                  </button>
                )}
                {showAllAssets && (
                  <button type="button" onClick={() => setShowAllAssets(false)}>
                    收起到前 15
                  </button>
                )}
              </div>
            </div>
            {visibleAssets.length === 0 ? (
              <p>暂无资产或未命中搜索条件。</p>
            ) : (
              <div style={{ display: "grid", gap: "10px" }}>
                {visibleAssets.map((asset) => (
                  <article key={asset.id} style={{ border: "1px solid #eee", borderRadius: "8px", padding: "10px" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", gap: "8px", alignItems: "center" }}>
                      <div>
                        <strong>{asset.symbol}</strong> {asset.name ? `- ${asset.name}` : ""}{" "}
                        {asset.market ? `(${asset.market})` : ""}
                      </div>
                      <div style={{ display: "flex", gap: "12px", alignItems: "center" }}>
                        <span>24h: {asset.new_views_24h}</span>
                        <span>7d: {asset.new_views_7d}</span>
                        <Link href={`/assets/${asset.id}`}>查看详情</Link>
                      </div>
                    </div>
                    <div style={{ marginTop: "8px", display: "grid", gap: "6px" }}>
                      {(() => {
                        const items = pickLatestViews(asset, 3);
                        if (items.length === 0) return <small key="none" style={{ color: "#666" }}>暂无观点</small>;
                        return (
                          <>
                            {items.map((item) => (
                              <div
                                key={item.kol_view_id}
                                style={{
                                  background: "color-mix(in srgb, var(--bg-elev-strong) 86%, transparent 14%)",
                                  border: "1px solid var(--line)",
                                  borderRadius: "6px",
                                  padding: "6px 8px",
                                }}
                              >
                                <strong>{horizonLabel(item.horizon)}</strong> [{stanceLabel(item.stance)}/{item.confidence}]{" "}
                                {item.summary || "暂无摘要"}
                                {" · "}
                                {item.kol_display_name || item.kol_handle || `KOL#${item.kol_id}`}
                              </div>
                            ))}
                          </>
                        );
                      })()}
                    </div>
                  </article>
                ))}
              </div>
            )}
          </section>

          <section style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "12px" }}>
            <h2 style={{ marginTop: 0 }}>活跃 KOL（7d）</h2>
            {(!data?.active_kols_7d || data.active_kols_7d.length === 0) && <p>暂无 KOL 聚合数据。</p>}
            {data?.active_kols_7d && data.active_kols_7d.length > 0 && (
              <div style={{ display: "grid", gap: "8px" }}>
                {data.active_kols_7d.map((kol) => (
                  <article key={kol.kol_id} style={{ border: "1px solid #eee", borderRadius: "8px", padding: "10px" }}>
                    <div>
                      <strong>{kol.display_name || kol.handle}</strong> ({kol.platform}/@{kol.handle}) · 观点数:{" "}
                      {kol.views_count_7d}
                    </div>
                    <small>
                      主要资产:{" "}
                      {kol.top_assets.length > 0
                        ? kol.top_assets.map((item) => `${item.symbol}(${item.views_count})`).join(", ")
                        : "暂无"}
                    </small>
                  </article>
                ))}
              </div>
            )}
          </section>
        </>
      )}
    </main>
  );
}
