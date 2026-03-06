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
  extraction_id?: number | null;
};

type AssetViewsResponse = {
  asset_id: number;
  groups: Array<{
    horizon: string;
    bull: KolView[];
    bear: KolView[];
    neutral: KolView[];
  }>;
  meta: {
    sort: string;
    generated_at: string;
    version_policy: string;
  };
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

const stanceStyle: Record<string, { color: string; borderColor: string; backgroundColor: string }> = {
  bull: { color: "#0a7f2e", borderColor: "#0a7f2e", backgroundColor: "#e8f8ee" },
  bear: { color: "#b42318", borderColor: "#b42318", backgroundColor: "#fdecec" },
  neutral: { color: "#6b7280", borderColor: "#6b7280", backgroundColor: "#f4f4f5" },
};
const FEED_PAGE_SIZE = 10;

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
  const loadFeed = useCallback(async (nextOffset: number, nextHorizon: typeof feedHorizon) => {
    setFeedLoading(true);
    setFeedError(null);
    try {
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
      setFeed((prev) => {
        if (nextOffset === 0 || !prev) {
          return payload;
        }
        return {
          ...payload,
          items: [...prev.items, ...payload.items],
        };
      });
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

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      setError(null);

      if (Number.isNaN(assetId)) {
        setError("Invalid asset id");
        setLoading(false);
        return;
      }

      try {
        const viewsRes = await fetch(`/api/assets/${assetId}/views`, { cache: "no-store" });

        const viewsBody = (await viewsRes.json()) as AssetViewsResponse | { detail?: string };
        if (!viewsRes.ok) {
          throw new Error(
            "detail" in viewsBody ? (viewsBody.detail ?? "Request failed") : `Request failed ${viewsRes.status}`,
          );
        }
        setData(viewsBody as AssetViewsResponse);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unknown error");
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
      <h1>Asset {Number.isNaN(assetId) ? "?" : assetId} Views</h1>
      <p>
        <Link href="/assets">返回资产列表</Link>
        {" | "}
        <Link href="/dashboard">返回 Dashboard</Link>
      </p>
      {loading && <p>Loading...</p>}
      {error && <p style={{ color: "crimson" }}>{error}</p>}

      {!loading && !error && data && (
        <div style={{ display: "grid", gap: "12px", marginTop: "12px" }}>
          <AssetViewsTimeline data={timeline} loading={timelineLoading} error={timelineError} />
          <section style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "12px" }}>
            <h2 style={{ marginTop: 0 }}>观点时间线</h2>
            <div style={{ display: "flex", gap: "8px", flexWrap: "wrap", alignItems: "center" }}>
              <label>
                horizon
                <select
                  value={feedHorizon}
                  onChange={(event) => {
                    setFeedHorizon(event.target.value as typeof feedHorizon);
                    setFeedOffset(0);
                  }}
                  style={{ marginLeft: "8px" }}
                >
                  <option value="all">all</option>
                  <option value="intraday">intraday</option>
                  <option value="1w">1w</option>
                  <option value="1m">1m</option>
                  <option value="3m">3m</option>
                  <option value="1y">1y</option>
                </select>
              </label>
              <button type="button" onClick={() => void refreshFeed()} disabled={feedLoading}>
                {feedLoading ? "Loading..." : "Refresh"}
              </button>
              <small style={{ color: "#666" }}>total: {feed?.total ?? 0}</small>
            </div>
            {feedError && <p style={{ color: "crimson" }}>{feedError}</p>}
            {!feedLoading && !feedError && feed && feed.items.length === 0 && <p>暂无观点。</p>}
            {feed && feed.items.length > 0 && (
              <div style={{ display: "grid", gap: "8px", marginTop: "10px" }}>
                {feed.items.map((view) => {
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
                          {view.stance}
                        </span>
                        <span>{view.horizon}</span>
                        <span>confidence: {view.confidence}</span>
                        <span>as_of: {view.as_of}</span>
                        <span>KOL: {view.kol_display_name || view.kol_handle || view.kol_id}</span>
                      </div>
                      <div style={{ marginTop: "6px" }}>{view.summary || "暂无摘要"}</div>
                      <div style={{ marginTop: "4px" }}>
                        source:{" "}
                        {view.source_url ? (
                          <a href={view.source_url} target="_blank" rel="noreferrer">
                            {view.source_url}
                          </a>
                        ) : (
                          <span>N/A</span>
                        )}
                      </div>
                    </article>
                  );
                })}
                <div>
                  <button
                    type="button"
                    onClick={() => setFeedOffset((prev) => prev + FEED_PAGE_SIZE)}
                    disabled={!feed.has_more || feedLoading}
                  >
                    Load More
                  </button>
                </div>
              </div>
            )}
          </section>
          {data.groups.length === 0 ? (
            <p>No KOL views yet.</p>
          ) : (
            data.groups.map((group) => (
              <section
                key={group.horizon}
                style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "12px" }}
              >
                <h2 style={{ marginTop: 0 }}>horizon: {group.horizon}</h2>
                <div style={{ display: "grid", gap: "8px" }}>
                  {[...group.bull, ...group.bear, ...group.neutral].map((view) => {
                    const badge = stanceStyle[view.stance] ?? stanceStyle.neutral;
                    return (
                      <article
                        key={view.id}
                        style={{ border: "1px solid #eee", borderRadius: "8px", padding: "10px" }}
                      >
                        <div style={{ display: "flex", gap: "8px", alignItems: "center", marginBottom: "6px" }}>
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
                            {view.stance}
                          </span>
                          <span>confidence: {view.confidence}</span>
                          <span>as_of: {view.as_of}</span>
                        </div>
                        <div>{view.summary}</div>
                        <div style={{ marginTop: "4px" }}>
                          source:{" "}
                          {view.source_url ? (
                            <a href={view.source_url} target="_blank" rel="noreferrer">
                              {view.source_url}
                            </a>
                          ) : (
                            <span>N/A</span>
                          )}
                        </div>
                      </article>
                    );
                  })}
                </div>
              </section>
            ))
          )}
          <small style={{ color: "#666" }}>
            sort={data.meta.sort}, policy={data.meta.version_policy}, generated_at={data.meta.generated_at}
          </small>
        </div>
      )}
    </main>
  );
}
