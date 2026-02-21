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
              <div style={{ display: "grid", gap: "8px" }}>
                {data.top_assets.map((asset) => (
                  <article
                    key={asset.asset_id}
                    style={{
                      border: "1px solid #eee",
                      borderRadius: "8px",
                      padding: "10px",
                      display: "flex",
                      justifyContent: "space-between",
                      gap: "10px",
                    }}
                  >
                    <div>
                      <Link href={`/assets/${asset.asset_id}`}>
                        <strong>{asset.symbol}</strong>
                      </Link>
                      {asset.market ? <span> ({asset.market})</span> : null}
                    </div>
                    <div>
                      views: {asset.views_count_7d} | avg_conf: {asset.avg_confidence_7d.toFixed(1)}
                    </div>
                  </article>
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
