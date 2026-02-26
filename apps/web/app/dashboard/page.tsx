"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

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

const horizonOrder: Array<DashboardAssetLatestView["horizon"]> = ["intraday", "1w", "1m", "3m", "1y"];

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
  const [windowKey, setWindowKey] = useState<WindowKey>("24h");
  const [clarityWindow, setClarityWindow] = useState<WindowKey>("7d");
  const [showAllAssets, setShowAllAssets] = useState(false);
  const todayDigestDate = new Date().toISOString().slice(0, 10);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({
        days: "7",
        profile_id: "1",
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
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void load();
  }, [clarityWindow, windowKey, showAllAssets]);

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
      <h1>Portfolio Dashboard</h1>

      <section style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "12px" }}>
        <div style={{ display: "flex", gap: "12px", alignItems: "center", flexWrap: "wrap" }}>
          <strong>新增观点: {windowKey === "24h" ? (data?.new_views_24h ?? "-") : (data?.new_views_7d ?? "-")}</strong>
          <span>pending extractions: {data?.pending_extractions_count ?? "-"}</span>
          <button type="button" onClick={() => setWindowKey("24h")} disabled={windowKey === "24h"}>
            24h
          </button>
          <button type="button" onClick={() => setWindowKey("7d")} disabled={windowKey === "7d"}>
            7d
          </button>
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="按 symbol/name 搜索"
            style={{ minWidth: "240px" }}
          />
          <button type="button" onClick={() => void load()} disabled={loading}>
            {loading ? "Loading..." : "Refresh"}
          </button>
          <Link href="/ingest">手动导入</Link>
          <Link href="/kols">KOL管理</Link>
          <Link href="/extractions">审核队列</Link>
          <Link href="/profile">Profile 设置</Link>
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
                      {item.direction}
                    </span>
                    <span>score: {item.clarity_score.toFixed(4)}</span>
                    <span>
                      N/K: {item.n}/{item.k}
                    </span>
                  </Link>
                ))}
              </div>
            )}
          </section>

          <section style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "12px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: "8px" }}>
              <h2 style={{ marginTop: 0, marginBottom: "8px" }}>
                资产列表 {showAllAssets ? "(全部)" : "(Top 15)"}
              </h2>
              <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
                {!showAllAssets && (data?.total_assets_count ?? 0) > 15 && (
                  <button type="button" onClick={() => setShowAllAssets(true)}>
                    查看更多资产（共{data?.total_assets_count ?? 0}）
                  </button>
                )}
                {showAllAssets && (
                  <button type="button" onClick={() => setShowAllAssets(false)}>
                    收起到 Top 15
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
                      {horizonOrder
                        .map((horizon) =>
                          asset.latest_views_by_horizon.find((item) => item.horizon === horizon),
                        )
                        .filter((item): item is DashboardAssetLatestView => Boolean(item))
                        .map((item) => (
                          <div key={item.kol_view_id} style={{ background: "#fafafa", borderRadius: "6px", padding: "6px 8px" }}>
                            <strong>{item.horizon}</strong> [{item.stance}/{item.confidence}] {item.summary || "暂无摘要"}
                            {" · "}
                            {item.kol_display_name || item.kol_handle || `KOL#${item.kol_id}`}
                          </div>
                        ))}
                      {asset.latest_views_by_horizon.length === 0 && <small style={{ color: "#666" }}>暂无观点</small>}
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
                      <strong>{kol.display_name || kol.handle}</strong> ({kol.platform}/@{kol.handle}) · views:{" "}
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
