"use client";

import { FormEvent, useEffect, useState } from "react";

type Kol = {
  id: number;
  platform: string;
  handle: string;
  display_name: string | null;
  enabled: boolean;
  created_at: string;
};

type KolAssetSummaryItem = {
  asset_id: number;
  symbol: string;
  name: string | null;
  market: string | null;
  views_count: number;
};

type KolAssetSummary = {
  kol_id: number;
  total_views: number;
  top_assets: KolAssetSummaryItem[];
};

type KolViewItem = {
  id: number;
  asset_id: number;
  asset_symbol: string;
  asset_name: string | null;
  stance: "bull" | "bear" | "neutral" | string;
  horizon: "intraday" | "1w" | "1m" | "3m" | "1y" | string;
  confidence: number;
  summary: string;
  source_url: string;
  as_of: string;
  created_at: string;
};

type KolViewsResponse = {
  kol_id: number;
  asset_id: number | null;
  total: number;
  limit: number;
  offset: number;
  has_more: boolean;
  items: KolViewItem[];
};

type AdminHardDeleteResponse = {
  counts: Record<string, number>;
};

function getHorizonLabel(horizon: string): string {
  const labels: Record<string, string> = {
    intraday: "日内",
    "1w": "1周",
    "1m": "1月",
    "3m": "3月",
    "1y": "1年",
  };
  return labels[horizon] ?? horizon;
}

function getStanceLabel(stance: string): string {
  const labels: Record<string, string> = {
    bull: "看涨",
    bear: "看跌",
    neutral: "中性",
  };
  return labels[stance] ?? stance;
}

export default function KolsPage() {
  const [kols, setKols] = useState<Kol[]>([]);
  const [platform, setPlatform] = useState("");
  const [handle, setHandle] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [cleanupError, setCleanupError] = useState<string | null>(null);
  const [cleanupMessage, setCleanupMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [cleanupBusy, setCleanupBusy] = useState(false);
  const [cleanupKolId, setCleanupKolId] = useState("");
  const [cleanupKolCascade, setCleanupKolCascade] = useState(false);
  const [cleanupKolDeleteRawPosts, setCleanupKolDeleteRawPosts] = useState(false);
  const [summaryByKolId, setSummaryByKolId] = useState<Record<number, KolAssetSummary>>({});
  const [summaryLoadingByKolId, setSummaryLoadingByKolId] = useState<Record<number, boolean>>({});
  const [summaryErrorByKolId, setSummaryErrorByKolId] = useState<Record<number, string>>({});
  const [activeAssetByKolId, setActiveAssetByKolId] = useState<Record<number, number | null>>({});
  const [viewsByKolAsset, setViewsByKolAsset] = useState<Record<string, KolViewsResponse>>({});
  const [viewsLoadingByKolAsset, setViewsLoadingByKolAsset] = useState<Record<string, boolean>>({});
  const [viewsErrorByKolAsset, setViewsErrorByKolAsset] = useState<Record<string, string>>({});

  const loadKols = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/kols", { cache: "no-store" });
      if (!res.ok) {
        throw new Error(`请求失败（状态码 ${res.status}）`);
      }
      const data = (await res.json()) as Kol[];
      setKols(data);
      void Promise.all(
        data.map(async (kol) => {
          setSummaryLoadingByKolId((prev) => ({ ...prev, [kol.id]: true }));
          setSummaryErrorByKolId((prev) => {
            const next = { ...prev };
            delete next[kol.id];
            return next;
          });
          try {
            const summaryRes = await fetch(`/api/kols/${kol.id}/assets-summary`, { cache: "no-store" });
            const summaryBody = (await summaryRes.json()) as KolAssetSummary | { detail?: string };
            if (!summaryRes.ok) {
              throw new Error("detail" in summaryBody ? (summaryBody.detail ?? "加载 KOL 资产汇总失败") : "加载 KOL 资产汇总失败");
            }
            setSummaryByKolId((prev) => ({ ...prev, [kol.id]: summaryBody as KolAssetSummary }));
          } catch (summaryErr) {
            setSummaryErrorByKolId((prev) => ({
              ...prev,
              [kol.id]: summaryErr instanceof Error ? summaryErr.message : "加载 KOL 资产汇总失败",
            }));
          } finally {
            setSummaryLoadingByKolId((prev) => ({ ...prev, [kol.id]: false }));
          }
        }),
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "未知错误");
    } finally {
      setLoading(false);
    }
  };

  const loadViewsByKolAndAsset = async (kolId: number, assetId: number) => {
    const key = `${kolId}-${assetId}`;
    setViewsLoadingByKolAsset((prev) => ({ ...prev, [key]: true }));
    setViewsErrorByKolAsset((prev) => {
      const next = { ...prev };
      delete next[key];
      return next;
    });
    try {
      const res = await fetch(`/api/kols/${kolId}/views?asset_id=${assetId}&limit=500&offset=0`, { cache: "no-store" });
      const body = (await res.json()) as KolViewsResponse | { detail?: string };
      if (!res.ok) {
        throw new Error("detail" in body ? (body.detail ?? "加载观点失败") : "加载观点失败");
      }
      setViewsByKolAsset((prev) => ({ ...prev, [key]: body as KolViewsResponse }));
    } catch (err) {
      setViewsErrorByKolAsset((prev) => ({ ...prev, [key]: err instanceof Error ? err.message : "加载观点失败" }));
    } finally {
      setViewsLoadingByKolAsset((prev) => ({ ...prev, [key]: false }));
    }
  };

  useEffect(() => {
    void loadKols();
  }, []);

  const onSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setError(null);
    try {
      setCreating(true);
      const res = await fetch("/api/kols", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          platform: platform.trim().toLowerCase(),
          handle: handle.trim().replace(/^@+/, ""),
          display_name: displayName.trim() || null,
        }),
      });

      if (!res.ok) {
        const body = (await res.json()) as { detail?: string };
        throw new Error(body.detail ?? `请求失败（状态码 ${res.status}）`);
      }

      setPlatform("");
      setHandle("");
      setDisplayName("");
      await loadKols();
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

  const deleteKolCleanup = async () => {
    const kolId = Number(cleanupKolId);
    if (!Number.isFinite(kolId) || kolId <= 0) {
      setCleanupError("请输入合法的 KOL ID");
      return;
    }
    if (!askYes("高风险操作：将删除该 KOL 相关数据。输入 YES 继续。")) return;
    if ((cleanupKolCascade || cleanupKolDeleteRawPosts) && !askYes("你已选择级联或原始数据删除。请再次输入 YES 确认。")) {
      return;
    }
    setCleanupBusy(true);
    setCleanupError(null);
    setCleanupMessage(null);
    try {
      const params = new URLSearchParams({
        confirm: "YES",
        enable_cascade: cleanupKolCascade ? "true" : "false",
        also_delete_raw_posts: cleanupKolDeleteRawPosts ? "true" : "false",
      });
      const res = await fetch(`/api/admin/kols/${kolId}?${params.toString()}`, { method: "DELETE" });
      const body = (await res.json()) as AdminHardDeleteResponse | { detail?: string };
      if (!res.ok) {
        throw new Error("detail" in body ? (body.detail ?? "删除 KOL 失败") : "删除 KOL 失败");
      }
      const done = body as AdminHardDeleteResponse;
      setCleanupMessage(`KOL 清理完成：${JSON.stringify(done.counts)}`);
      setCleanupKolId("");
      await loadKols();
    } catch (err) {
      setCleanupError(err instanceof Error ? err.message : "删除 KOL 失败");
    } finally {
      setCleanupBusy(false);
    }
  };

  return (
    <main style={{ padding: "24px", fontFamily: "monospace" }}>
      <h1>KOL 管理</h1>
      <form onSubmit={onSubmit} style={{ display: "grid", gap: "8px", maxWidth: "360px" }}>
        <input
          value={platform}
          onChange={(e) => setPlatform(e.target.value)}
          placeholder="平台（如 x）"
          required
        />
        <input value={handle} onChange={(e) => setHandle(e.target.value)} placeholder="账号（handle）" required />
        <input
          value={displayName}
          onChange={(e) => setDisplayName(e.target.value)}
          placeholder="显示名称（可选）"
        />
        <button type="submit" disabled={creating}>
          {creating ? "创建中..." : "创建 KOL"}
        </button>
        <button type="button" onClick={() => void loadKols()} disabled={loading || creating}>
          {loading ? "加载中..." : "刷新"}
        </button>
      </form>

      {error && <p style={{ color: "crimson" }}>{error}</p>}
      {loading ? (
        <p>加载中...</p>
      ) : (
        <div style={{ marginTop: "16px", display: "grid", gap: "8px" }}>
          {kols.length === 0 ? (
            <p>暂无 KOL。</p>
          ) : (
            kols.map((kol) => (
              <div key={kol.id} style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "10px" }}>
                <strong>{kol.display_name || kol.handle}</strong>{" "}
                <span style={{ color: "#666" }}>
                  ({kol.platform}/@{kol.handle})
                </span>
                {summaryLoadingByKolId[kol.id] ? (
                  <div style={{ marginTop: "6px", color: "#666" }}>资产讨论概览加载中...</div>
                ) : summaryErrorByKolId[kol.id] ? (
                  <div style={{ marginTop: "6px", color: "crimson" }}>{summaryErrorByKolId[kol.id]}</div>
                ) : (
                  <div style={{ marginTop: "6px", display: "grid", gap: "6px" }}>
                    <div style={{ color: "#666" }}>观点数: {summaryByKolId[kol.id]?.total_views ?? 0}</div>
                    <div style={{ display: "flex", flexWrap: "wrap", gap: "6px", alignItems: "center" }}>
                      <span style={{ color: "#666" }}>主要资产:</span>
                      {(summaryByKolId[kol.id]?.top_assets ?? []).length === 0 && <span>暂无</span>}
                      {(summaryByKolId[kol.id]?.top_assets ?? []).map((asset) => {
                        const active = activeAssetByKolId[kol.id] === asset.asset_id;
                        return (
                          <button
                            key={`${kol.id}-${asset.asset_id}`}
                            type="button"
                            onClick={() => {
                              setActiveAssetByKolId((prev) => ({
                                ...prev,
                                [kol.id]: prev[kol.id] === asset.asset_id ? null : asset.asset_id,
                              }));
                              if (activeAssetByKolId[kol.id] !== asset.asset_id) {
                                void loadViewsByKolAndAsset(kol.id, asset.asset_id);
                              }
                            }}
                            style={{
                              border: "1px solid #cfd6df",
                              borderRadius: "999px",
                              padding: "2px 8px",
                              cursor: "pointer",
                              background: active ? "#dbeafe" : "#f8fafc",
                            }}
                          >
                            {asset.symbol}({asset.views_count})
                          </button>
                        );
                      })}
                    </div>
                    {(() => {
                      const activeAssetId = activeAssetByKolId[kol.id];
                      if (!activeAssetId) return null;
                      const key = `${kol.id}-${activeAssetId}`;
                      const payload = viewsByKolAsset[key];
                      const loadingViews = viewsLoadingByKolAsset[key];
                      const errorViews = viewsErrorByKolAsset[key];
                      return (
                        <section style={{ border: "1px solid #e5e7eb", borderRadius: "8px", padding: "8px", marginTop: "4px" }}>
                          {loadingViews && <p style={{ margin: 0, color: "#666" }}>观点加载中...</p>}
                          {errorViews && <p style={{ margin: 0, color: "crimson" }}>{errorViews}</p>}
                          {!loadingViews && !errorViews && payload && (
                            <div style={{ display: "grid", gap: "8px" }}>
                              <div style={{ color: "#666" }}>
                                {payload.items[0]?.asset_symbol ?? activeAssetId} 全部观点（近到远）：{payload.total}
                              </div>
                              {payload.items.length === 0 ? (
                                <p style={{ margin: 0 }}>暂无观点。</p>
                              ) : (
                                payload.items.map((view) => (
                                  <article key={view.id} style={{ border: "1px solid #e5e7eb", borderRadius: "8px", padding: "8px" }}>
                                    <div style={{ display: "flex", flexWrap: "wrap", gap: "8px", fontSize: "12px" }}>
                                      <strong>{getStanceLabel(view.stance)}</strong>
                                      <span>{getHorizonLabel(view.horizon)}</span>
                                      <span>置信度: {view.confidence}</span>
                                      <span>观点日期: {view.as_of}</span>
                                    </div>
                                    <div style={{ marginTop: "6px" }}>{view.summary || "暂无摘要"}</div>
                                    <div style={{ marginTop: "4px", fontSize: "12px" }}>
                                      来源:{" "}
                                      {view.source_url ? (
                                        <a href={view.source_url} target="_blank" rel="noreferrer">
                                          {view.source_url}
                                        </a>
                                      ) : (
                                        "无"
                                      )}
                                    </div>
                                  </article>
                                ))
                              )}
                            </div>
                          )}
                        </section>
                      );
                    })()}
                  </div>
                )}
                </div>
            ))
          )}
        </div>
      )}

      <section style={{ marginTop: "16px", border: "1px solid #eee", borderRadius: "8px", padding: "10px", maxWidth: "560px" }}>
        <h2 style={{ marginTop: 0, marginBottom: "8px" }}>管理清理 - 删除 KOL</h2>
        <p style={{ marginTop: 0 }}>所有操作都需要输入 YES；raw_posts 删除必须同时满足 enable_cascade=true。</p>
        <label>
          KOL ID
          <input
            list="cleanup-kol-list"
            value={cleanupKolId}
            onChange={(event) => setCleanupKolId(event.target.value)}
            style={{ display: "block", width: "100%" }}
          />
          <datalist id="cleanup-kol-list">
            {kols.map((item) => (
              <option key={item.id} value={item.id}>
                @{item.handle}
              </option>
            ))}
          </datalist>
        </label>
        <label style={{ display: "block", marginTop: "8px" }}>
          <input
            type="checkbox"
            checked={cleanupKolCascade}
            onChange={(event) => setCleanupKolCascade(event.target.checked)}
          />{" "}
          enable_cascade
        </label>
        <label style={{ display: "block", marginTop: "4px" }}>
          <input
            type="checkbox"
            checked={cleanupKolDeleteRawPosts}
            onChange={(event) => setCleanupKolDeleteRawPosts(event.target.checked)}
          />{" "}
          also_delete_raw_posts
        </label>
        <button type="button" onClick={() => void deleteKolCleanup()} disabled={cleanupBusy} style={{ marginTop: "8px" }}>
          {cleanupBusy ? "删除中..." : "硬删除 KOL"}
        </button>
        {cleanupError && <p style={{ color: "crimson", marginBottom: 0 }}>{cleanupError}</p>}
        {cleanupMessage && <p style={{ marginBottom: 0 }}>{cleanupMessage}</p>}
      </section>
    </main>
  );
}
