"use client";

import { FormEvent, useEffect, useState } from "react";
import Link from "next/link";

type Asset = {
  id: number;
  symbol: string;
  name: string | null;
  market: string | null;
  created_at: string;
};

export default function AssetsPage() {
  const [assets, setAssets] = useState<Asset[]>([]);
  const [symbol, setSymbol] = useState("");
  const [name, setName] = useState("");
  const [market, setMarket] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);

  const loadAssets = async () => {
    setLoading(true);
    setError(null);

    try {
      const res = await fetch("/api/assets", { cache: "no-store" });
      if (!res.ok) {
        throw new Error(`Request failed with status ${res.status}`);
      }

      const data = (await res.json()) as Asset[];
      setAssets(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadAssets();
  }, []);

  const onSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setError(null);

    try {
      setCreating(true);
      const res = await fetch("/api/assets", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          symbol: symbol.trim().toUpperCase(),
          name: name.trim() || null,
          market: market.trim().toUpperCase() || null,
        }),
      });

      if (!res.ok) {
        const body = (await res.json()) as { detail?: string };
        throw new Error(body.detail ?? `Request failed with status ${res.status}`);
      }

      setSymbol("");
      setName("");
      setMarket("");
      await loadAssets();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setCreating(false);
    }
  };

  return (
    <main style={{ padding: "24px", fontFamily: "monospace" }}>
      <h1>Assets</h1>
      <form onSubmit={onSubmit} style={{ display: "grid", gap: "8px", maxWidth: "360px" }}>
        <input
          value={symbol}
          onChange={(e) => setSymbol(e.target.value)}
          placeholder="symbol (e.g. AAPL)"
          required
        />
        <input value={name} onChange={(e) => setName(e.target.value)} placeholder="name (optional)" />
        <input
          value={market}
          onChange={(e) => setMarket(e.target.value)}
          placeholder="market (optional, e.g. US)"
        />
        <button type="submit" disabled={creating}>
          {creating ? "Creating..." : "Create Asset"}
        </button>
        <button type="button" onClick={() => void loadAssets()} disabled={loading || creating}>
          {loading ? "Loading..." : "Refresh"}
        </button>
      </form>

      {error && <p style={{ color: "crimson" }}>{error}</p>}
      {loading ? (
        <p>Loading...</p>
      ) : (
        <div style={{ marginTop: "16px", display: "grid", gap: "8px" }}>
          {assets.length === 0 ? (
            <p>No assets yet.</p>
          ) : (
            assets.map((asset) => (
              <div
                key={asset.id}
                style={{
                  border: "1px solid #ddd",
                  borderRadius: "8px",
                  padding: "10px",
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                }}
              >
                <div>
                  <strong>{asset.symbol}</strong> {asset.name ? `- ${asset.name}` : ""}
                </div>
                <Link href={`/assets/${asset.id}`}>详情</Link>
              </div>
            ))
          )}
        </div>
      )}
    </main>
  );
}
