"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

type ProfileSummary = {
  id: number;
  name: string;
  created_at: string;
};

type ProfileKol = {
  kol_id: number;
  weight: number;
  enabled: boolean;
  kol_display_name: string | null;
  kol_handle: string;
  kol_platform: string;
};

type ProfileDetail = {
  id: number;
  name: string;
  created_at: string;
  kols: ProfileKol[];
  markets: string[];
};

const MARKET_OPTIONS = ["CRYPTO", "US", "HK", "CN", "FX", "AUTO"];

export default function ProfilePage() {
  const [profiles, setProfiles] = useState<ProfileSummary[]>([]);
  const [profileId, setProfileId] = useState(1);
  const [detail, setDetail] = useState<ProfileDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadProfiles = useCallback(async (): Promise<number | null> => {
    const res = await fetch("/api/profiles", { cache: "no-store" });
    const body = (await res.json()) as ProfileSummary[] | { detail?: string };
    if (!res.ok || !Array.isArray(body)) {
      throw new Error("Load profiles failed");
    }
    setProfiles(body);
    if (body.length === 0) {
      return null;
    }
    const matched = body.some((item) => item.id === profileId) ? profileId : body[0].id;
    if (matched !== profileId) {
      setProfileId(matched);
    }
    return matched;
  }, [profileId]);

  const loadDetail = useCallback(async (id: number) => {
    const res = await fetch(`/api/profiles/${id}`, { cache: "no-store" });
    const body = (await res.json()) as ProfileDetail | { detail?: string };
    if (!res.ok) {
      throw new Error("detail" in body ? (body.detail ?? "Load profile failed") : "Load profile failed");
    }
    setDetail(body as ProfileDetail);
  }, []);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const effectiveProfileId = await loadProfiles();
      if (effectiveProfileId === null) {
        setDetail(null);
        return;
      }
      await loadDetail(effectiveProfileId);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, [loadDetail, loadProfiles]);

  useEffect(() => {
    void load();
  }, [load, profileId]);

  const saveKols = async () => {
    if (!detail) return;
    setSaving(true);
    setError(null);
    try {
      const payload = {
        items: detail.kols.map((item) => ({
          kol_id: item.kol_id,
          weight: item.weight,
          enabled: item.enabled,
        })),
      };
      const res = await fetch(`/api/profiles/${detail.id}/kols`, {
        method: "PUT",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(payload),
      });
      const body = (await res.json()) as ProfileDetail | { detail?: string };
      if (!res.ok) {
        throw new Error("detail" in body ? (body.detail ?? "Save KOL weights failed") : "Save KOL weights failed");
      }
      setDetail(body as ProfileDetail);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Save KOL weights failed");
    } finally {
      setSaving(false);
    }
  };

  const saveMarkets = async () => {
    if (!detail) return;
    setSaving(true);
    setError(null);
    try {
      const res = await fetch(`/api/profiles/${detail.id}/markets`, {
        method: "PUT",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ markets: detail.markets }),
      });
      const body = (await res.json()) as ProfileDetail | { detail?: string };
      if (!res.ok) {
        throw new Error("detail" in body ? (body.detail ?? "Save markets failed") : "Save markets failed");
      }
      setDetail(body as ProfileDetail);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Save markets failed");
    } finally {
      setSaving(false);
    }
  };

  return (
    <main style={{ padding: "24px", fontFamily: "monospace", display: "grid", gap: "14px" }}>
      <h1>Profile Settings</h1>

      <section style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "12px" }}>
        <div style={{ display: "flex", gap: "12px", alignItems: "center", flexWrap: "wrap" }}>
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
          <button type="button" onClick={() => void load()} disabled={loading}>
            {loading ? "Loading..." : "Refresh"}
          </button>
          <Link href="/dashboard">返回 Dashboard</Link>
        </div>
      </section>

      {error && <p style={{ color: "crimson" }}>{error}</p>}

      {!loading && detail && (
        <>
          <section style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "12px" }}>
            <h2 style={{ marginTop: 0 }}>Markets</h2>
            <div style={{ display: "flex", gap: "10px", flexWrap: "wrap" }}>
              {MARKET_OPTIONS.map((market) => {
                const checked = detail.markets.includes(market);
                return (
                  <label key={market} style={{ display: "inline-flex", alignItems: "center", gap: "4px" }}>
                    <input
                      type="checkbox"
                      checked={checked}
                      onChange={(event) => {
                        setDetail((prev) => {
                          if (!prev) return prev;
                          const next = event.target.checked
                            ? [...prev.markets, market]
                            : prev.markets.filter((item) => item !== market);
                          return { ...prev, markets: Array.from(new Set(next)) };
                        });
                      }}
                    />
                    {market}
                  </label>
                );
              })}
            </div>
            <p style={{ marginBottom: 0, color: "#666" }}>未勾选任何 market = 不做 market 过滤。</p>
            <button type="button" onClick={() => void saveMarkets()} disabled={saving}>
              {saving ? "Saving..." : "保存 Markets"}
            </button>
          </section>

          <section style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "12px" }}>
            <h2 style={{ marginTop: 0 }}>KOL权重</h2>
            <div style={{ display: "grid", gap: "8px" }}>
              {detail.kols.map((item) => (
                <article key={item.kol_id} style={{ border: "1px solid #eee", borderRadius: "8px", padding: "8px" }}>
                  <div style={{ display: "flex", gap: "12px", alignItems: "center", flexWrap: "wrap" }}>
                    <strong>#{item.kol_id}</strong>
                    <span>{item.kol_display_name || item.kol_handle}</span>
                    <small>({item.kol_platform}/@{item.kol_handle})</small>
                    <label style={{ display: "inline-flex", alignItems: "center", gap: "4px" }}>
                      enabled
                      <input
                        type="checkbox"
                        checked={item.enabled}
                        onChange={(event) => {
                          setDetail((prev) => {
                            if (!prev) return prev;
                            return {
                              ...prev,
                              kols: prev.kols.map((row) =>
                                row.kol_id === item.kol_id ? { ...row, enabled: event.target.checked } : row,
                              ),
                            };
                          });
                        }}
                      />
                    </label>
                    <label style={{ display: "inline-flex", alignItems: "center", gap: "4px" }}>
                      weight
                      <input
                        type="number"
                        min={0}
                        step={0.1}
                        value={item.weight}
                        onChange={(event) => {
                          const value = Number(event.target.value);
                          setDetail((prev) => {
                            if (!prev) return prev;
                            return {
                              ...prev,
                              kols: prev.kols.map((row) =>
                                row.kol_id === item.kol_id ? { ...row, weight: Number.isFinite(value) ? value : 0 } : row,
                              ),
                            };
                          });
                        }}
                        style={{ width: "90px" }}
                      />
                    </label>
                  </div>
                </article>
              ))}
            </div>
            <button type="button" onClick={() => void saveKols()} disabled={saving}>
              {saving ? "Saving..." : "保存 KOL 权重"}
            </button>
          </section>
        </>
      )}
    </main>
  );
}
