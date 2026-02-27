"use client";

import { useCallback, useEffect, useState } from "react";

type HealthResponse = {
  ok?: boolean;
  [key: string]: unknown;
};

export default function HealthPage() {
  const [data, setData] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const loadHealth = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/health", { cache: "no-store" });
      const json = (await res.json()) as HealthResponse;
      if (!res.ok) {
        throw new Error(`Request failed with status ${res.status}`);
      }
      setData(json);
    } catch (err) {
      setData(null);
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadHealth();
  }, [loadHealth]);

  return (
    <main style={{ padding: "24px", fontFamily: "monospace" }}>
      <h1>/health</h1>
      <button type="button" onClick={() => void loadHealth()} disabled={loading}>
        {loading ? "Loading..." : "Refresh"}
      </button>
      {error ? (
        <pre style={{ color: "crimson" }}>{error}</pre>
      ) : loading ? (
        <p>Loading...</p>
      ) : data ? (
        <pre>{JSON.stringify(data, null, 2)}</pre>
      ) : (
        <p>Empty health payload.</p>
      )}
    </main>
  );
}
