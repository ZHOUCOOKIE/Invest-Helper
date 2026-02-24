"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useMemo, useState } from "react";

type ProfileSummary = {
  id: number;
  name: string;
  created_at: string;
};

type TopAsset = {
  asset_id: number;
  symbol: string;
  name: string | null;
  market: string | null;
  new_views_24h: number;
  new_views_7d: number;
  weighted_views_24h: number;
  weighted_views_7d: number;
};

type HorizonCount = {
  horizon: "intraday" | "1w" | "1m" | "3m" | "1y" | string;
  bull_count: number;
  bear_count: number;
  neutral_count: number;
};

type TopView = {
  kol_id: number;
  kol_display_name: string | null;
  kol_handle: string | null;
  stance: "bull" | "bear" | "neutral" | string;
  horizon: "intraday" | "1w" | "1m" | "3m" | "1y" | string;
  confidence: number;
  summary: string;
  source_url: string;
  as_of: string;
  created_at: string;
  kol_weight: number;
  weighted_score: number;
};

type AssetSummary = {
  asset_id: number;
  symbol: string;
  name: string | null;
  market: string | null;
  horizon_counts: HorizonCount[];
  clarity: number;
  top_views_bull: TopView[];
  top_views_bear: TopView[];
  top_views_neutral: TopView[];
};

type DailyDigest = {
  id: number;
  profile_id: number;
  digest_date: string;
  version: number;
  generated_at: string;
  top_assets: TopAsset[];
  per_asset_summary: AssetSummary[];
  metadata: {
    generated_at: string;
    days: number;
    summary_window_start: string;
    summary_window_end: string;
    generated_from_ts: string;
    generated_to_ts: string;
    time_field_used: "as_of" | "posted_at" | "created_at";
  };
};

export default function DailyDigestPage() {
  const params = useParams<{ date: string }>();
  const digestDate = params?.date;
  const [digest, setDigest] = useState<DailyDigest | null>(null);
  const [dates, setDates] = useState<string[]>([]);
  const [profiles, setProfiles] = useState<ProfileSummary[]>([]);
  const [profileId, setProfileId] = useState(1);
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadProfiles = async () => {
    const res = await fetch("/api/profiles", { cache: "no-store" });
    const body = (await res.json()) as ProfileSummary[] | { detail?: string };
    if (!res.ok || !Array.isArray(body)) {
      throw new Error("Load profiles failed");
    }
    setProfiles(body);
    if (body.length > 0 && !body.some((item) => item.id === profileId)) {
      setProfileId(body[0].id);
    }
  };

  const load = async () => {
    if (!digestDate) return;
    setLoading(true);
    setError(null);
    try {
      await loadProfiles();
      const [datesRes, digestRes] = await Promise.all([
        fetch(`/api/digests/dates?profile_id=${profileId}`, { cache: "no-store" }),
        fetch(`/api/digests?date=${digestDate}&profile_id=${profileId}`, { cache: "no-store" }),
      ]);

      const datesBody = (await datesRes.json()) as string[] | { detail?: string };
      if (datesRes.ok && Array.isArray(datesBody)) {
        setDates(datesBody);
      } else {
        setDates([]);
      }

      const digestBody = (await digestRes.json()) as DailyDigest | { detail?: string };
      if (!digestRes.ok) {
        throw new Error("detail" in digestBody ? (digestBody.detail ?? "Load digest failed") : "Load digest failed");
      }
      setDigest(digestBody as DailyDigest);
    } catch (err) {
      setDigest(null);
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [digestDate, profileId]);

  const generate = async () => {
    if (!digestDate) return;
    setGenerating(true);
    setError(null);
    try {
      const res = await fetch(`/api/digests/generate?date=${digestDate}&days=7&profile_id=${profileId}`, { method: "POST" });
      const body = (await res.json()) as DailyDigest | { detail?: string };
      if (!res.ok) {
        throw new Error("detail" in body ? (body.detail ?? "Generate digest failed") : "Generate digest failed");
      }
      setDigest(body as DailyDigest);
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Generate digest failed");
    } finally {
      setGenerating(false);
    }
  };

  const latestDate = useMemo(() => dates[0] ?? null, [dates]);

  return (
    <main style={{ padding: "24px", fontFamily: "monospace", display: "grid", gap: "14px" }}>
      <h1>Daily Digest</h1>

      <section style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "12px", display: "grid", gap: "8px" }}>
        <div style={{ display: "flex", gap: "12px", alignItems: "center", flexWrap: "wrap" }}>
          <strong>date: {digestDate}</strong>
          <label style={{ display: "inline-flex", alignItems: "center", gap: "6px" }}>
            profile
            <select value={profileId} onChange={(event) => setProfileId(Number(event.target.value))}>
              {profiles.map((item) => (
                <option key={item.id} value={item.id}>
                  #{item.id} {item.name}
                </option>
              ))}
            </select>
          </label>
          <button type="button" onClick={() => void generate()} disabled={generating}>
            {generating ? "Generating..." : "生成日报"}
          </button>
          <button type="button" onClick={() => void load()} disabled={loading}>
            {loading ? "Loading..." : "Refresh"}
          </button>
          <Link href="/dashboard">返回 Dashboard</Link>
          <Link href="/profile">Profile 设置</Link>
          {latestDate && <Link href={`/digests/${latestDate}`}>最新日期</Link>}
        </div>
        {dates.length > 0 && (
          <small style={{ color: "#666" }}>
            可用日期: {dates.slice(0, 14).join(", ")}
          </small>
        )}
      </section>

      {error && <p style={{ color: "crimson" }}>{error}</p>}

      {!loading && !error && !digest && <p>该日期暂无日报，可点击“生成日报”。</p>}

      {!loading && !error && digest && (
        <>
          <section style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "12px" }}>
            <div>
              <strong>profile:</strong> {digest.profile_id} | <strong>version:</strong> {digest.version} | <strong>days:</strong> {digest.metadata.days}
            </div>
            <small style={{ color: "#666" }}>
              window: {new Date(digest.metadata.generated_from_ts).toLocaleString()} ~ {new Date(digest.metadata.generated_to_ts).toLocaleString()} | time_field: {digest.metadata.time_field_used}
            </small>
          </section>

          <section style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "12px" }}>
            <h2 style={{ marginTop: 0 }}>Top Assets (Weighted)</h2>
            {digest.top_assets.length === 0 ? (
              <p>暂无数据。</p>
            ) : (
              <div style={{ display: "grid", gap: "8px" }}>
                {digest.top_assets.map((asset) => (
                  <article key={asset.asset_id} style={{ border: "1px solid #eee", borderRadius: "8px", padding: "10px" }}>
                    <strong>{asset.symbol}</strong> {asset.name ? `- ${asset.name}` : ""} {asset.market ? `(${asset.market})` : ""}
                    <div>
                      <small>
                        weighted 24h/7d: {asset.weighted_views_24h.toFixed(2)} / {asset.weighted_views_7d.toFixed(2)} | raw 24h/7d: {asset.new_views_24h} / {asset.new_views_7d}
                      </small>
                    </div>
                  </article>
                ))}
              </div>
            )}
          </section>

          <section style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "12px" }}>
            <h2 style={{ marginTop: 0 }}>Per Asset Summary</h2>
            {digest.per_asset_summary.length === 0 ? (
              <p>暂无数据。</p>
            ) : (
              <div style={{ display: "grid", gap: "10px" }}>
                {digest.per_asset_summary.map((asset) => (
                  <article key={asset.asset_id} style={{ border: "1px solid #eee", borderRadius: "8px", padding: "10px" }}>
                    <div>
                      <strong>{asset.symbol}</strong> {asset.name ? `- ${asset.name}` : ""} {asset.market ? `(${asset.market})` : ""}
                    </div>
                    <div>
                      <small>clarity: {asset.clarity.toFixed(3)}</small>
                    </div>
                    <div style={{ marginTop: "6px" }}>
                      <small>
                        horizon counts: {asset.horizon_counts.map((item) => `${item.horizon}[B:${item.bull_count}/S:${item.bear_count}/N:${item.neutral_count}]`).join(", ") || "-"}
                      </small>
                    </div>

                    <div style={{ marginTop: "8px", display: "grid", gap: "6px" }}>
                      {["bull", "bear", "neutral"].map((stance) => {
                        const rows = stance === "bull" ? asset.top_views_bull : stance === "bear" ? asset.top_views_bear : asset.top_views_neutral;
                        return (
                          <div key={`${asset.asset_id}-${stance}`}>
                            <strong>{stance}</strong>
                            {rows.length === 0 ? (
                              <small> -</small>
                            ) : (
                              <ul>
                                {rows.map((view) => (
                                  <li key={`${stance}-${view.kol_id}-${view.source_url}-${view.created_at}`}>
                                    [score={view.weighted_score.toFixed(1)}={view.kol_weight.toFixed(2)}*{view.confidence}] {view.kol_display_name || view.kol_handle || `KOL#${view.kol_id}`}: {view.summary} ·{" "}
                                    <a href={view.source_url} target="_blank" rel="noreferrer">
                                      link
                                    </a>{" "}
                                    ({view.as_of})
                                  </li>
                                ))}
                              </ul>
                            )}
                          </div>
                        );
                      })}
                    </div>
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
