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

type AdminHardDeleteResponse = {
  counts: Record<string, number>;
};

export default function AssetsPage() {
  const [assets, setAssets] = useState<Asset[]>([]);
  const [symbol, setSymbol] = useState("");
  const [name, setName] = useState("");
  const [market, setMarket] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [cleanupError, setCleanupError] = useState<string | null>(null);
  const [cleanupMessage, setCleanupMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [cleanupBusy, setCleanupBusy] = useState(false);
  const [cleanupAssetId, setCleanupAssetId] = useState("");
  const [cleanupAssetCascade, setCleanupAssetCascade] = useState(false);

  const loadAssets = async () => {
    setLoading(true);
    setError(null);

    try {
      const res = await fetch("/api/assets", { cache: "no-store" });
      if (!res.ok) {
        throw new Error(`请求失败（状态码 ${res.status}）`);
      }

      const data = (await res.json()) as Asset[];
      setAssets(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "未知错误");
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
        throw new Error(body.detail ?? `请求失败（状态码 ${res.status}）`);
      }

      setSymbol("");
      setName("");
      setMarket("");
      await loadAssets();
    } catch (err) {
      setError(err instanceof Error ? err.message : "未知错误");
    } finally {
      setCreating(false);
    }
  };

  const askYes = (message: string): boolean => {
    const confirmText = window.prompt(message, "");
    return confirmText === "YES";
  };

  const deleteAssetCleanup = async () => {
    const assetId = Number(cleanupAssetId);
    if (!Number.isFinite(assetId) || assetId <= 0) {
      setCleanupError("请输入合法的 Asset ID");
      return;
    }
    if (!askYes("高风险操作：将删除该资产相关数据。输入 YES 继续。")) return;
    if (cleanupAssetCascade && !askYes("你已选择级联删除资产基础信息。请再次输入 YES 确认。")) {
      return;
    }
    setCleanupBusy(true);
    setCleanupError(null);
    setCleanupMessage(null);
    try {
      const params = new URLSearchParams({
        confirm: "YES",
        enable_cascade: cleanupAssetCascade ? "true" : "false",
      });
      const res = await fetch(`/api/admin/assets/${assetId}?${params.toString()}`, { method: "DELETE" });
      const body = (await res.json()) as AdminHardDeleteResponse | { detail?: string };
      if (!res.ok) {
        throw new Error("detail" in body ? (body.detail ?? "删除资产失败") : "删除资产失败");
      }
      const done = body as AdminHardDeleteResponse;
      setCleanupMessage(`资产清理完成：${JSON.stringify(done.counts)}`);
      setCleanupAssetId("");
      await loadAssets();
    } catch (err) {
      setCleanupError(err instanceof Error ? err.message : "删除资产失败");
    } finally {
      setCleanupBusy(false);
    }
  };

  return (
    <main style={{ padding: "24px", fontFamily: "monospace" }}>
      <h1>资产管理</h1>
      <form onSubmit={onSubmit} style={{ display: "grid", gap: "8px", maxWidth: "360px" }}>
        <input
          value={symbol}
          onChange={(e) => setSymbol(e.target.value)}
          placeholder="资产代码（如 AAPL）"
          required
        />
        <input value={name} onChange={(e) => setName(e.target.value)} placeholder="资产名称（可选）" />
        <input
          value={market}
          onChange={(e) => setMarket(e.target.value)}
          placeholder="市场（可选，如 US）"
        />
        <button type="submit" disabled={creating}>
          {creating ? "创建中..." : "创建资产"}
        </button>
        <button type="button" onClick={() => void loadAssets()} disabled={loading || creating}>
          {loading ? "加载中..." : "刷新"}
        </button>
      </form>

      {error && <p style={{ color: "crimson" }}>{error}</p>}
      {loading ? (
        <p>加载中...</p>
      ) : (
        <div style={{ marginTop: "16px", display: "grid", gap: "8px" }}>
          {assets.length === 0 ? (
            <p>暂无资产。</p>
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

      <section style={{ marginTop: "16px", border: "1px solid #eee", borderRadius: "8px", padding: "10px", maxWidth: "560px" }}>
        <h2 style={{ marginTop: 0, marginBottom: "8px" }}>管理清理 - 删除资产</h2>
        <p style={{ marginTop: 0 }}>所有操作都需要输入 YES。</p>
        <label>
          资产 ID
          <input
            list="cleanup-asset-list"
            value={cleanupAssetId}
            onChange={(event) => setCleanupAssetId(event.target.value)}
            style={{ display: "block", width: "100%" }}
          />
          <datalist id="cleanup-asset-list">
            {assets.map((item) => (
              <option key={item.id} value={item.id}>
                {item.symbol}
              </option>
            ))}
          </datalist>
        </label>
        <label style={{ display: "block", marginTop: "8px" }}>
          <input
            type="checkbox"
            checked={cleanupAssetCascade}
            onChange={(event) => setCleanupAssetCascade(event.target.checked)}
          />{" "}
          enable_cascade（删除资产及别名）
        </label>
        <button type="button" onClick={() => void deleteAssetCleanup()} disabled={cleanupBusy} style={{ marginTop: "8px" }}>
          {cleanupBusy ? "删除中..." : "硬删除资产"}
        </button>
        {cleanupError && <p style={{ color: "crimson", marginBottom: 0 }}>{cleanupError}</p>}
        {cleanupMessage && <p style={{ marginBottom: 0 }}>{cleanupMessage}</p>}
      </section>
    </main>
  );
}
