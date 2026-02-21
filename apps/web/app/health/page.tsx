"use client";

import { useEffect, useState } from "react";

type HealthResponse = {
  ok?: boolean;
  [key: string]: unknown;
};

export default function HealthPage() {
  const [data, setData] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;

    const loadHealth = async () => {
      try {
        const res = await fetch("/api/health", { cache: "no-store" });
        const json = (await res.json()) as HealthResponse;

        if (!res.ok) {
          throw new Error(`Request failed with status ${res.status}`);
        }

        if (isMounted) {
          setData(json);
        }
      } catch (err) {
        if (isMounted) {
          setError(err instanceof Error ? err.message : "Unknown error");
        }
      }
    };

    void loadHealth();

    return () => {
      isMounted = false;
    };
  }, []);

  return (
    <main style={{ padding: "24px", fontFamily: "monospace" }}>
      <h1>/health</h1>
      {error ? (
        <pre style={{ color: "crimson" }}>{error}</pre>
      ) : (
        <pre>{JSON.stringify(data, null, 2)}</pre>
      )}
    </main>
  );
}
