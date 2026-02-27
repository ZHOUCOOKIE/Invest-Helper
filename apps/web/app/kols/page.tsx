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

export default function KolsPage() {
  const [kols, setKols] = useState<Kol[]>([]);
  const [platform, setPlatform] = useState("");
  const [handle, setHandle] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);

  const loadKols = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/kols", { cache: "no-store" });
      if (!res.ok) {
        throw new Error(`Request failed with status ${res.status}`);
      }
      const data = (await res.json()) as Kol[];
      setKols(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
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
        throw new Error(body.detail ?? `Request failed with status ${res.status}`);
      }

      setPlatform("");
      setHandle("");
      setDisplayName("");
      await loadKols();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setCreating(false);
    }
  };

  return (
    <main style={{ padding: "24px", fontFamily: "monospace" }}>
      <h1>KOLs</h1>
      <form onSubmit={onSubmit} style={{ display: "grid", gap: "8px", maxWidth: "360px" }}>
        <input
          value={platform}
          onChange={(e) => setPlatform(e.target.value)}
          placeholder="platform (e.g. x)"
          required
        />
        <input value={handle} onChange={(e) => setHandle(e.target.value)} placeholder="handle" required />
        <input
          value={displayName}
          onChange={(e) => setDisplayName(e.target.value)}
          placeholder="display_name (optional)"
        />
        <button type="submit" disabled={creating}>
          {creating ? "Creating..." : "Create KOL"}
        </button>
        <button type="button" onClick={() => void loadKols()} disabled={loading || creating}>
          {loading ? "Loading..." : "Refresh"}
        </button>
      </form>

      {error && <p style={{ color: "crimson" }}>{error}</p>}
      {loading ? (
        <p>Loading...</p>
      ) : (
        <div style={{ marginTop: "16px", display: "grid", gap: "8px" }}>
          {kols.length === 0 ? (
            <p>No KOLs yet.</p>
          ) : (
            kols.map((kol) => (
              <div key={kol.id} style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "10px" }}>
                <strong>{kol.display_name || kol.handle}</strong>
                <div>
                  {kol.platform} / @{kol.handle}
                </div>
              </div>
            ))
          )}
        </div>
      )}
    </main>
  );
}
