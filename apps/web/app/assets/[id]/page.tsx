"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import type { FormEvent } from "react";
import { useEffect, useMemo, useState } from "react";

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

type Kol = {
  id: number;
  platform: string;
  handle: string;
  display_name: string | null;
  enabled: boolean;
  created_at: string;
};

type FormState = {
  kol_id: string;
  stance: "bull" | "bear" | "neutral";
  horizon: "intraday" | "1w" | "1m" | "3m" | "1y";
  confidence: string;
  summary: string;
  source_url: string;
  as_of: string;
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
  const [kols, setKols] = useState<Kol[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [submitMessage, setSubmitMessage] = useState<string | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [feed, setFeed] = useState<AssetViewsFeedResponse | null>(null);
  const [feedLoading, setFeedLoading] = useState(false);
  const [feedError, setFeedError] = useState<string | null>(null);
  const [feedHorizon, setFeedHorizon] = useState<"all" | "intraday" | "1w" | "1m" | "3m" | "1y">("all");
  const [feedOffset, setFeedOffset] = useState(0);
  const [form, setForm] = useState<FormState>({
    kol_id: "",
    stance: "bull",
    horizon: "1w",
    confidence: "50",
    summary: "",
    source_url: "",
    as_of: "",
  });

  const loadViews = async () => {
    const res = await fetch(`/api/assets/${assetId}/views`, { cache: "no-store" });
    const body = (await res.json()) as AssetViewsResponse | { detail?: string };
    if (!res.ok) {
      throw new Error("detail" in body ? (body.detail ?? "Request failed") : `Request failed ${res.status}`);
    }
    setData(body as AssetViewsResponse);
  };

  const loadFeed = async (nextOffset: number, nextHorizon: typeof feedHorizon) => {
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
  };

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      setError(null);
      setSubmitMessage(null);
      setSubmitError(null);

      if (Number.isNaN(assetId)) {
        setError("Invalid asset id");
        setLoading(false);
        return;
      }

      try {
        const [viewsRes, kolsRes] = await Promise.all([
          fetch(`/api/assets/${assetId}/views`, { cache: "no-store" }),
          fetch("/api/kols?enabled=true", { cache: "no-store" }),
        ]);

        const viewsBody = (await viewsRes.json()) as AssetViewsResponse | { detail?: string };
        if (!viewsRes.ok) {
          throw new Error(
            "detail" in viewsBody ? (viewsBody.detail ?? "Request failed") : `Request failed ${viewsRes.status}`,
          );
        }
        setData(viewsBody as AssetViewsResponse);

        const kolsBody = (await kolsRes.json()) as Kol[] | { detail?: string };
        if (!kolsRes.ok) {
          throw new Error("detail" in kolsBody ? (kolsBody.detail ?? "Load kols failed") : "Load kols failed");
        }
        const enabledKols = (kolsBody as Kol[]).filter((item) => item.enabled);
        setKols(enabledKols);
        setForm((prev) => ({
          ...prev,
          kol_id: prev.kol_id || (enabledKols[0] ? String(enabledKols[0].id) : ""),
        }));
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
  }, [assetId, feedOffset, feedHorizon]);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSubmitMessage(null);
    setSubmitError(null);

    if (!form.kol_id) {
      setSubmitError("请选择 KOL");
      return;
    }

    if (Number.isNaN(assetId)) {
      setSubmitError("Invalid asset id");
      return;
    }

    setSubmitting(true);
    try {
      const payload: Record<string, number | string> = {
        kol_id: Number(form.kol_id),
        asset_id: assetId,
        stance: form.stance,
        horizon: form.horizon,
        confidence: Number(form.confidence),
      };
      if (form.summary.trim()) payload.summary = form.summary.trim();
      if (form.source_url.trim()) payload.source_url = form.source_url.trim();
      if (form.as_of) payload.as_of = form.as_of;

      const res = await fetch("/api/kol-views", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const body = (await res.json()) as { detail?: string };
      if (!res.ok) {
        throw new Error(body.detail ?? `Create failed ${res.status}`);
      }

      await loadViews();
      setFeedOffset(0);
      setSubmitMessage("观点创建成功，列表已刷新。");
      setForm((prev) => ({ ...prev, summary: "", source_url: "", as_of: "" }));
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : "提交失败");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <main style={{ padding: "24px", fontFamily: "monospace" }}>
      <h1>Asset {Number.isNaN(assetId) ? "?" : assetId} Views</h1>
      <p>
        <Link href="/assets">返回资产列表</Link>
      </p>
      {loading && <p>Loading...</p>}
      {error && <p style={{ color: "crimson" }}>{error}</p>}

      {!loading && !error && data && (
        <div style={{ display: "grid", gap: "12px", marginTop: "12px" }}>
          <section style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "12px" }}>
            <h2 style={{ marginTop: 0 }}>新增观点</h2>
            <form onSubmit={handleSubmit} style={{ display: "grid", gap: "8px", maxWidth: "720px" }}>
              <label>
                KOL
                <select
                  value={form.kol_id}
                  onChange={(event) => setForm((prev) => ({ ...prev, kol_id: event.target.value }))}
                  style={{ display: "block", width: "100%" }}
                  required
                >
                  <option value="" disabled>
                    请选择 KOL
                  </option>
                  {kols.map((kol) => (
                    <option key={kol.id} value={kol.id}>
                      {kol.display_name || kol.handle} ({kol.platform}/@{kol.handle})
                    </option>
                  ))}
                </select>
              </label>
              <label>
                stance
                <select
                  value={form.stance}
                  onChange={(event) =>
                    setForm((prev) => ({ ...prev, stance: event.target.value as FormState["stance"] }))
                  }
                  style={{ display: "block", width: "100%" }}
                  required
                >
                  <option value="bull">bull</option>
                  <option value="bear">bear</option>
                  <option value="neutral">neutral</option>
                </select>
              </label>
              <label>
                horizon
                <select
                  value={form.horizon}
                  onChange={(event) =>
                    setForm((prev) => ({ ...prev, horizon: event.target.value as FormState["horizon"] }))
                  }
                  style={{ display: "block", width: "100%" }}
                  required
                >
                  <option value="intraday">intraday</option>
                  <option value="1w">1w</option>
                  <option value="1m">1m</option>
                  <option value="3m">3m</option>
                  <option value="1y">1y</option>
                </select>
              </label>
              <label>
                confidence (0-100)
                <input
                  type="number"
                  min={0}
                  max={100}
                  value={form.confidence}
                  onChange={(event) => setForm((prev) => ({ ...prev, confidence: event.target.value }))}
                  style={{ display: "block", width: "100%" }}
                  required
                />
              </label>
              <label>
                summary (optional)
                <textarea
                  value={form.summary}
                  onChange={(event) => setForm((prev) => ({ ...prev, summary: event.target.value }))}
                  rows={3}
                  style={{ display: "block", width: "100%" }}
                />
              </label>
              <label>
                source_url (optional)
                <input
                  type="url"
                  value={form.source_url}
                  onChange={(event) => setForm((prev) => ({ ...prev, source_url: event.target.value }))}
                  style={{ display: "block", width: "100%" }}
                />
              </label>
              <label>
                as_of (optional)
                <input
                  type="date"
                  value={form.as_of}
                  onChange={(event) => setForm((prev) => ({ ...prev, as_of: event.target.value }))}
                  style={{ display: "block", width: "100%" }}
                />
              </label>
              <button type="submit" disabled={submitting || kols.length === 0}>
                {submitting ? "Submitting..." : "提交观点"}
              </button>
            </form>
            {kols.length === 0 && <p style={{ color: "#666" }}>暂无可用 KOL，请先在 KOL 页面创建并启用。</p>}
            {submitMessage && <p style={{ color: "green" }}>{submitMessage}</p>}
            {submitError && <p style={{ color: "crimson" }}>{submitError}</p>}
          </section>
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
              <button type="button" onClick={() => setFeedOffset(0)} disabled={feedLoading}>
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
