"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useMemo, useState } from "react";

type KolView = {
  id: number;
  kol_id: number;
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

const stanceStyle: Record<string, { color: string; borderColor: string; backgroundColor: string }> = {
  bull: { color: "#0a7f2e", borderColor: "#0a7f2e", backgroundColor: "#e8f8ee" },
  bear: { color: "#b42318", borderColor: "#b42318", backgroundColor: "#fdecec" },
  neutral: { color: "#6b7280", borderColor: "#6b7280", backgroundColor: "#f4f4f5" },
};

export default function AssetDetailPage() {
  const params = useParams<{ id: string }>();
  const assetId = useMemo(() => Number(params.id), [params.id]);
  const [data, setData] = useState<AssetViewsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

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
        const res = await fetch(`/api/assets/${assetId}/views`, { cache: "no-store" });
        const body = (await res.json()) as AssetViewsResponse | { detail?: string };
        if (!res.ok) {
          throw new Error("detail" in body ? (body.detail ?? "Request failed") : `Request failed ${res.status}`);
        }
        setData(body as AssetViewsResponse);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unknown error");
      } finally {
        setLoading(false);
      }
    };

    void load();
  }, [assetId]);

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
                          <a href={view.source_url} target="_blank" rel="noreferrer">
                            {view.source_url}
                          </a>
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
