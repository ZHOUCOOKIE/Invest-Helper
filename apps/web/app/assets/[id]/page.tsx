"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useState } from "react";
import { AssetViewsTimeline } from "./views-timeline";

type KolView = {
  id: number;
  kol_id: number;
  kol_display_name?: string | null;
  kol_handle?: string | null;
  asset_id: number;
  stance: "bull" | "bear" | "neutral" | string;
  horizon: string;
  confidence: number;
  summary: string;
  source_url: string;
  as_of: string;
  created_at: string;
  posted_at?: string | null;
  posted_at_raw?: string | null;
  extraction_id?: number | null;
};

type AssetViewsResponse = {
  asset_id: number;
  meta: {
    sort: string;
    generated_at: string;
    version_policy: string;
  };
};

type AssetListItem = {
  id: number;
  symbol: string;
  name: string | null;
};

type AssetViewsFeedResponse = {
  asset_id: number;
  horizon: "intraday" | "1w" | "1m" | "3m" | "1y" | null;
  total: number;
  limit: number;
  offset: number;
  has_more: boolean;
  items: KolView[];
};

type AssetViewsTimelineResponse = {
  asset_id: number;
  days: number;
  since_date: string;
  generated_at: string;
  items: KolView[];
};

type AssetViewPostDetail = {
  asset_id: number;
  view_id: number;
  extraction_id: number | null;
  raw_post_id: number | null;
  source_url: string;
  summary: string;
  posted_at: string | null;
  posted_at_raw?: string | null;
  author_handle: string | null;
  content_text: string | null;
};

const stanceStyle: Record<string, { color: string; borderColor: string; backgroundColor: string }> = {
  bull: { color: "#0a7f2e", borderColor: "#0a7f2e", backgroundColor: "#e8f8ee" },
  bear: { color: "#b42318", borderColor: "#b42318", backgroundColor: "#fdecec" },
  neutral: { color: "#6b7280", borderColor: "#6b7280", backgroundColor: "#f4f4f5" },
};
const horizonRank: Record<string, number> = {
  intraday: 0,
  "1w": 1,
  "1m": 2,
  "3m": 3,
  "1y": 4,
};

function sortViewsByTimeThenHorizon(items: KolView[]): KolView[] {
  return [...items].sort((a, b) => {
    const aDate = (a.as_of || "").slice(0, 10);
    const bDate = (b.as_of || "").slice(0, 10);
    if (bDate !== aDate) return bDate.localeCompare(aDate);
    const aRank = horizonRank[a.horizon] ?? 999;
    const bRank = horizonRank[b.horizon] ?? 999;
    if (aRank !== bRank) return aRank - bRank;
    return b.id - a.id;
  });
}

const FEED_PAGE_SIZE = 10;
const FEED_MAX_FETCH_ROUNDS = 200;

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

function stripTimezoneSuffix(value: string | null | undefined): string {
  if (typeof value !== "string") return "-";
  const trimmed = value.trim();
  if (!trimmed) return "-";
  return trimmed.replace(/\s?(?:Z|[+-]\d{2}:\d{2})$/i, "").trim();
}

export default function AssetDetailPage() {
  const params = useParams<{ id: string }>();
  const assetId = useMemo(() => Number(params.id), [params.id]);
  const [data, setData] = useState<AssetViewsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [feed, setFeed] = useState<AssetViewsFeedResponse | null>(null);
  const [feedLoading, setFeedLoading] = useState(false);
  const [feedError, setFeedError] = useState<string | null>(null);
  const [feedHorizon, setFeedHorizon] = useState<"all" | "intraday" | "1w" | "1m" | "3m" | "1y">("all");
  const [feedOffset, setFeedOffset] = useState(0);
  const [timeline, setTimeline] = useState<AssetViewsTimelineResponse | null>(null);
  const [timelineLoading, setTimelineLoading] = useState(false);
  const [timelineError, setTimelineError] = useState<string | null>(null);
  const [selectedView, setSelectedView] = useState<KolView | null>(null);
  const [selectedViewDetail, setSelectedViewDetail] = useState<AssetViewPostDetail | null>(null);
  const [selectedViewLoading, setSelectedViewLoading] = useState(false);
  const [selectedViewError, setSelectedViewError] = useState<string | null>(null);
  const [assetDisplayName, setAssetDisplayName] = useState<string | null>(null);

  const loadFeed = useCallback(async (nextOffset: number, nextHorizon: typeof feedHorizon) => {
    setFeedLoading(true);
    setFeedError(null);
    try {
      if (nextHorizon === "all" && nextOffset === 0) {
        const mergedItems: KolView[] = [];
        let currentOffset = 0;
        let rounds = 0;
        let finalPayload: AssetViewsFeedResponse | null = null;
        while (rounds < FEED_MAX_FETCH_ROUNDS) {
          rounds += 1;
          const res = await fetch(
            `/api/assets/${assetId}/views/feed?limit=${FEED_PAGE_SIZE}&offset=${currentOffset}`,
            { cache: "no-store" },
          );
          const body = (await res.json()) as AssetViewsFeedResponse | { detail?: string };
          if (!res.ok) {
            throw new Error("detail" in body ? (body.detail ?? "Load feed failed") : "Load feed failed");
          }
          const payload = body as AssetViewsFeedResponse;
          mergedItems.push(...payload.items);
          finalPayload = payload;
          if (!payload.has_more) {
            break;
          }
          currentOffset += payload.limit;
        }
        setFeed({
          asset_id: assetId,
          horizon: null,
          total: finalPayload?.total ?? mergedItems.length,
          limit: FEED_PAGE_SIZE,
          offset: 0,
          has_more: false,
          items: mergedItems,
        });
        return;
      }

      const horizonParam = nextHorizon === "all" ? "" : `&horizon=${nextHorizon}`;
      const res = await fetch(
        `/api/assets/${assetId}/views/feed?limit=${FEED_PAGE_SIZE}&offset=${nextOffset}${horizonParam}`,
        { cache: "no-store" },
      );
      const body = (await res.json()) as AssetViewsFeedResponse | { detail?: string };
      if (!res.ok) {
        throw new Error("detail" in body ? (body.detail ?? "Load feed failed") : "Load feed failed");
      }
      const payload = body as AssetViewsFeedResponse;
      setFeed((prev) => (nextOffset === 0 || !prev ? payload : { ...payload, items: [...prev.items, ...payload.items] }));
    } catch (err) {
      setFeedError(err instanceof Error ? err.message : "Load feed failed");
    } finally {
      setFeedLoading(false);
    }
  }, [assetId]);

  const loadTimeline = useCallback(async () => {
    setTimelineLoading(true);
    setTimelineError(null);
    try {
      const res = await fetch(`/api/assets/${assetId}/views/timeline?days=365`, { cache: "no-store" });
      const body = (await res.json()) as AssetViewsTimelineResponse | { detail?: string };
      if (!res.ok) {
        throw new Error("detail" in body ? (body.detail ?? "Load timeline failed") : "Load timeline failed");
      }
      setTimeline(body as AssetViewsTimelineResponse);
    } catch (err) {
      setTimelineError(err instanceof Error ? err.message : "Load timeline failed");
    } finally {
      setTimelineLoading(false);
    }
  }, [assetId]);

  const refreshFeed = useCallback(async () => {
    setFeedOffset(0);
    await loadFeed(0, feedHorizon);
  }, [feedHorizon, loadFeed]);

  const openViewDetail = useCallback(async (view: KolView) => {
    setSelectedView(view);
    setSelectedViewLoading(true);
    setSelectedViewError(null);
    setSelectedViewDetail(null);
    try {
      const res = await fetch(`/api/assets/${assetId}/views/${view.id}/post-detail`, { cache: "no-store" });
      const body = (await res.json()) as AssetViewPostDetail | { detail?: string };
      if (!res.ok) {
        throw new Error("detail" in body ? (body.detail ?? "加载贴文详情失败") : "加载贴文详情失败");
      }
      setSelectedViewDetail(body as AssetViewPostDetail);
    } catch (err) {
      setSelectedViewError(err instanceof Error ? err.message : "加载贴文详情失败");
    } finally {
      setSelectedViewLoading(false);
    }
  }, [assetId]);

  const closeViewDetail = useCallback(() => {
    setSelectedView(null);
    setSelectedViewDetail(null);
    setSelectedViewError(null);
    setSelectedViewLoading(false);
  }, []);

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      setError(null);

      if (Number.isNaN(assetId)) {
        setError("无效的资产 ID");
        setLoading(false);
        return;
      }

      try {
        const [viewsRes, assetsRes] = await Promise.all([
          fetch(`/api/assets/${assetId}/views`, { cache: "no-store" }),
          fetch("/api/assets", { cache: "no-store" }),
        ]);

        const viewsBody = (await viewsRes.json()) as AssetViewsResponse | { detail?: string };
        if (!viewsRes.ok) {
          throw new Error(
            "detail" in viewsBody ? (viewsBody.detail ?? "请求失败") : `请求失败 ${viewsRes.status}`,
          );
        }
        setData(viewsBody as AssetViewsResponse);

        const assetsBody = (await assetsRes.json()) as AssetListItem[] | { detail?: string };
        if (assetsRes.ok && Array.isArray(assetsBody)) {
          const matched = assetsBody.find((item) => item.id === assetId);
          if (matched) {
            const label = (matched.name || "").trim() || (matched.symbol || "").trim();
            setAssetDisplayName(label || String(assetId));
          } else {
            setAssetDisplayName(String(assetId));
          }
        } else {
          setAssetDisplayName(String(assetId));
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "未知错误");
      } finally {
        setLoading(false);
      }
    };
    void load();
  }, [assetId]);

  useEffect(() => {
    if (Number.isNaN(assetId)) return;
    void loadFeed(feedOffset, feedHorizon);
  }, [assetId, feedOffset, feedHorizon, loadFeed]);

  useEffect(() => {
    if (Number.isNaN(assetId)) return;
    void loadTimeline();
  }, [assetId, loadTimeline]);

  return (
    <main style={{ padding: "24px", fontFamily: "monospace" }}>
      <h1>资产 {Number.isNaN(assetId) ? "?" : (assetDisplayName ?? String(assetId))} 观点</h1>
      <p>
        <Link href="/assets">返回资产列表</Link>
      </p>
      {loading && <p>加载中...</p>}
      {error && <p style={{ color: "crimson" }}>{error}</p>}

      {!loading && !error && data && (
        <div style={{ display: "grid", gap: "12px", marginTop: "12px" }}>
          <AssetViewsTimeline data={timeline} loading={timelineLoading} error={timelineError} />
          <section style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "12px" }}>
            <h2 style={{ marginTop: 0 }}>观点时间线</h2>
            <div style={{ display: "flex", gap: "8px", flexWrap: "wrap", alignItems: "center" }}>
              <label>
                影响周期
                <select
                  value={feedHorizon}
                  onChange={(event) => {
                    setFeedHorizon(event.target.value as typeof feedHorizon);
                    setFeedOffset(0);
                  }}
                  style={{ marginLeft: "8px" }}
                >
                  <option value="all">全部</option>
                  <option value="intraday">日内</option>
                  <option value="1w">1周</option>
                  <option value="1m">1月</option>
                  <option value="3m">3月</option>
                  <option value="1y">1年</option>
                </select>
              </label>
              <button type="button" onClick={() => void refreshFeed()} disabled={feedLoading}>
                {feedLoading ? "加载中..." : "刷新"}
              </button>
              <small style={{ color: "#666" }}>总数: {feed?.total ?? 0}</small>
            </div>
            {feedError && <p style={{ color: "crimson" }}>{feedError}</p>}
            {!feedLoading && !feedError && feed && feed.items.length === 0 && <p>暂无观点。</p>}
            {feed && feed.items.length > 0 && (
              <div style={{ display: "grid", gap: "8px", marginTop: "10px" }}>
                {sortViewsByTimeThenHorizon(feed.items).map((view) => {
                  const badge = stanceStyle[view.stance] ?? stanceStyle.neutral;
                  return (
                    <article key={`feed-${view.id}`} style={{ border: "1px solid #eee", borderRadius: "8px", padding: "10px" }}>
                      <div style={{ display: "flex", gap: "8px", flexWrap: "wrap", alignItems: "center" }}>
                        <span
                          style={{
                            border: `1px solid ${badge.borderColor}`,
                            backgroundColor: badge.backgroundColor,
                            color: badge.color,
                            borderRadius: "999px",
                            padding: "2px 8px",
                            fontWeight: 700,
                            textTransform: "uppercase",
                            fontSize: "12px",
                          }}
                        >
                          {getStanceLabel(view.stance)}
                        </span>
                        <span>{getHorizonLabel(view.horizon)}</span>
                        <span>置信度: {view.confidence}</span>
                        <span>观点日期: {view.as_of}</span>
                        <span>KOL: {view.kol_display_name || view.kol_handle || view.kol_id}</span>
                      </div>
                      <div style={{ marginTop: "6px" }}>{view.summary || "暂无摘要"}</div>
                      <div style={{ marginTop: "4px" }}>
                        来源:{" "}
                        {view.source_url ? (
                          <a href={view.source_url} target="_blank" rel="noreferrer">
                            {view.source_url}
                          </a>
                        ) : (
                          <span>无</span>
                        )}
                      </div>
                      <div style={{ marginTop: "8px" }}>
                        <button type="button" onClick={() => void openViewDetail(view)}>
                          查看贴文原文
                        </button>
                      </div>
                    </article>
                  );
                })}
                {feedHorizon !== "all" && (
                  <div>
                    <button
                      type="button"
                      onClick={() => setFeedOffset((prev) => prev + FEED_PAGE_SIZE)}
                      disabled={!feed.has_more || feedLoading}
                    >
                      加载更多
                    </button>
                  </div>
                )}
              </div>
            )}
          </section>
          <small style={{ color: "#666" }}>
            排序策略={data.meta.sort}，版本策略={data.meta.version_policy}，生成时间={stripTimezoneSuffix(data.meta.generated_at)}
          </small>
        </div>
      )}
      {selectedView && (
        <div
          role="dialog"
          aria-modal="true"
          onClick={closeViewDetail}
          style={{
            position: "fixed",
            inset: 0,
            backgroundColor: "rgba(0,0,0,0.35)",
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            zIndex: 80,
            padding: "12px",
          }}
        >
          <section
            onClick={(event) => event.stopPropagation()}
            style={{
              width: "min(920px, 100%)",
              maxHeight: "88vh",
              overflowY: "auto",
              borderRadius: "10px",
              backgroundColor: "var(--bg-elev-strong)",
              border: "1px solid var(--line)",
              padding: "14px",
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: "8px" }}>
              <h3 style={{ margin: 0 }}>观点详情</h3>
              <button type="button" onClick={closeViewDetail}>
                关闭
              </button>
            </div>
            <div style={{ marginTop: "8px", display: "grid", gap: "6px", fontSize: "13px" }}>
              <div>观点 ID: {selectedView.id}</div>
              <div>周期: {getHorizonLabel(selectedView.horizon)} | 置信度: {selectedView.confidence}</div>
              <div>观点摘要: {selectedView.summary || "暂无摘要"}</div>
            </div>
            {selectedViewLoading && <p style={{ marginTop: "10px" }}>详情加载中...</p>}
            {selectedViewError && <p style={{ marginTop: "10px", color: "crimson" }}>{selectedViewError}</p>}
            {!selectedViewLoading && !selectedViewError && selectedViewDetail && (
              <div style={{ marginTop: "10px", display: "grid", gap: "8px" }}>
                <div>
                  <strong>来源链接:</strong>{" "}
                  {selectedViewDetail.source_url ? (
                    <a href={selectedViewDetail.source_url} target="_blank" rel="noreferrer">
                      {selectedViewDetail.source_url}
                    </a>
                  ) : (
                    "无"
                  )}
                </div>
                <div>
                  <strong>作者:</strong> {selectedViewDetail.author_handle ? `@${selectedViewDetail.author_handle}` : "无"}
                </div>
                <div>
                  <strong>发布时间:</strong> {stripTimezoneSuffix(selectedViewDetail.posted_at_raw ?? selectedViewDetail.posted_at)}
                </div>
                <div>
                  <strong>贴文原文:</strong>
                  <pre
                    style={{
                      marginTop: "6px",
                      whiteSpace: "pre-wrap",
                      border: "1px solid var(--line)",
                      borderRadius: "8px",
                      padding: "10px",
                    }}
                  >
                    {selectedViewDetail.content_text || "未找到原文。"}
                  </pre>
                </div>
              </div>
            )}
          </section>
        </div>
      )}
    </main>
  );
}
