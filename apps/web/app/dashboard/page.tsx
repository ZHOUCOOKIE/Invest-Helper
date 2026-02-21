"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

type DashboardPendingExtraction = {
  id: number;
  platform: string;
  author_handle: string;
  url: string;
  posted_at: string;
  created_at: string;
};

type DashboardTopAsset = {
  asset_id: number;
  symbol: string;
  market: string | null;
  views_count_7d: number;
  avg_confidence_7d: number;
};

type DashboardClarity = {
  horizon: "intraday" | "1w" | "1m" | "3m" | "1y" | string;
  bull_count: number;
  bear_count: number;
  neutral_count: number;
  clarity: number;
};

type DashboardResponse = {
  pending_extractions_count: number;
  latest_pending_extractions: DashboardPendingExtraction[];
  top_assets: DashboardTopAsset[];
  clarity: DashboardClarity[];
  extraction_stats: {
    window_hours: number;
    extraction_count: number;
    dummy_count: number;
    openai_count: number;
    error_count: number;
  };
};

const DEFAULT_DAYS = 7;

function formatPct(value: number): string {
  return `${(value * 100).toFixed(0)}%`;
}

export default function DashboardPage() {
  const [days, setDays] = useState(String(DEFAULT_DAYS));
  const [data, setData] = useState<DashboardResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedAssetId, setSelectedAssetId] = useState<number | null>(null);

  const load = async (inputDays: string) => {
    setLoading(true);
    setError(null);
    try {
      const safeDays = Math.max(1, Number(inputDays) || DEFAULT_DAYS);
      const res = await fetch(`/api/dashboard?days=${safeDays}`, { cache: "no-store" });
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
    void load(days);
  }, [days]);

  const clarityHint = useMemo(() => {
    if (!data || data.clarity.length === 0) return null;
    const ordered = [...data.clarity].sort((a, b) => b.clarity - a.clarity);
    return {
      highest: ordered[0],
      lowest: ordered[ordered.length - 1],
    };
  }, [data]);

  return (
    <main style={{ padding: "24px", fontFamily: "monospace", display: "grid", gap: "14px" }}>
      <h1>Daily Dashboard</h1>

      <section style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "12px" }}>
        <div style={{ display: "flex", gap: "12px", alignItems: "center", flexWrap: "wrap" }}>
          <strong>Pending Extractions: {data?.pending_extractions_count ?? "-"}</strong>
          <Link href="/extractions">去审核列表</Link>
          <Link href="/ingest">去手动导入</Link>
          <label>
            days
            <input
              type="number"
              min={1}
              max={90}
              value={days}
              onChange={(event) => setDays(event.target.value)}
              style={{ marginLeft: "8px", width: "72px" }}
            />
          </label>
          <button type="button" onClick={() => void load(days)} disabled={loading}>
            {loading ? "Loading..." : "Refresh"}
          </button>
        </div>
      </section>

      {error && <p style={{ color: "crimson" }}>{error}</p>}

      {!loading && !error && data && (
        <>
          <section style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "12px" }}>
            <h2 style={{ marginTop: 0 }}>Extraction Health (24h)</h2>
            <div style={{ display: "flex", gap: "14px", flexWrap: "wrap" }}>
              <strong>提取次数: {data.extraction_stats.extraction_count}</strong>
              <span>dummy: {data.extraction_stats.dummy_count}</span>
              <span>openai: {data.extraction_stats.openai_count}</span>
              <span>错误数: {data.extraction_stats.error_count}</span>
              <span>
                openai 占比:{" "}
                {data.extraction_stats.extraction_count > 0
                  ? `${((data.extraction_stats.openai_count / data.extraction_stats.extraction_count) * 100).toFixed(0)}%`
                  : "0%"}
              </span>
            </div>
          </section>

          <section style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "12px" }}>
            <h2 style={{ marginTop: 0 }}>Top Assets ({days}d)</h2>
            {clarityHint && (
              <p>
                clarity highest: {clarityHint.highest.horizon} {formatPct(clarityHint.highest.clarity)} | lowest:{" "}
                {clarityHint.lowest.horizon} {formatPct(clarityHint.lowest.clarity)}
              </p>
            )}
            {data.top_assets.length === 0 ? (
              <p>No approved views in current window.</p>
            ) : (
              <table style={{ width: "100%", borderCollapse: "collapse" }}>
                <thead>
                  <tr>
                    <th style={{ textAlign: "left", borderBottom: "1px solid #eee", padding: "6px" }}>Asset</th>
                    <th style={{ textAlign: "left", borderBottom: "1px solid #eee", padding: "6px" }}>views</th>
                    <th style={{ textAlign: "left", borderBottom: "1px solid #eee", padding: "6px" }}>avg_conf</th>
                    <th style={{ textAlign: "left", borderBottom: "1px solid #eee", padding: "6px" }}>action</th>
                  </tr>
                </thead>
                <tbody>
                  {data.top_assets.map((asset) => {
                    const selected = selectedAssetId === asset.asset_id;
                    return (
                      <tr
                        key={asset.asset_id}
                        onClick={() => setSelectedAssetId(asset.asset_id)}
                        style={{
                          background: selected ? "#f0f6ff" : undefined,
                          outline: selected ? "1px solid #6ba6ff" : undefined,
                          cursor: "pointer",
                        }}
                      >
                        <td style={{ padding: "8px", borderBottom: "1px solid #f5f5f5" }}>
                          <strong>{asset.symbol}</strong> {asset.market ? `(${asset.market})` : ""}
                        </td>
                        <td style={{ padding: "8px", borderBottom: "1px solid #f5f5f5" }}>{asset.views_count_7d}</td>
                        <td style={{ padding: "8px", borderBottom: "1px solid #f5f5f5" }}>
                          {asset.avg_confidence_7d.toFixed(1)}
                        </td>
                        <td style={{ padding: "8px", borderBottom: "1px solid #f5f5f5" }}>
                          <Link href={`/assets/${asset.asset_id}`}>
                            <strong>查看详情 →</strong>
                          </Link>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
          </section>

          <section style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "12px" }}>
            <h2 style={{ marginTop: 0 }}>Clarity</h2>
            {data.clarity.length === 0 ? (
              <p>No clarity data.</p>
            ) : (
              <div style={{ display: "grid", gap: "8px" }}>
                {data.clarity.map((item) => (
                  <div key={item.horizon} style={{ display: "grid", gap: "4px" }}>
                    <div>
                      {item.horizon}: {formatPct(item.clarity)} (bull={item.bull_count}, bear={item.bear_count},
                      neutral={item.neutral_count})
                    </div>
                    <progress max={100} value={Math.round(item.clarity * 100)} style={{ width: "100%" }} />
                  </div>
                ))}
              </div>
            )}
          </section>

          <section style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "12px" }}>
            <h2 style={{ marginTop: 0 }}>Latest Pending Extractions</h2>
            {data.latest_pending_extractions.length === 0 ? (
              <p>No pending extractions.</p>
            ) : (
              <div style={{ display: "grid", gap: "8px" }}>
                {data.latest_pending_extractions.map((item) => (
                  <article key={item.id} style={{ border: "1px solid #eee", borderRadius: "8px", padding: "10px" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", gap: "8px" }}>
                      <strong>
                        <Link href={`/extractions/${item.id}`}>Extraction #{item.id}</Link>
                      </strong>
                      <small>created_at: {item.created_at}</small>
                    </div>
                    <div>
                      {item.platform} / @{item.author_handle}
                    </div>
                    <div>posted_at: {item.posted_at}</div>
                    <a href={item.url} target="_blank" rel="noreferrer">
                      {item.url}
                    </a>
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
