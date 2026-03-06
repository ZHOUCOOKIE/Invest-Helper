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
        throw new Error(`请求失败（状态码 ${res.status}）`);
      }
      setData(json);
    } catch (err) {
      setData(null);
      setError(err instanceof Error ? err.message : "未知错误");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadHealth();
  }, [loadHealth]);

  return (
    <main style={{ padding: "24px", fontFamily: "monospace" }}>
      <h1>健康检查 /health</h1>
      <button type="button" onClick={() => void loadHealth()} disabled={loading}>
        {loading ? "加载中..." : "刷新"}
      </button>
      {error ? (
        <pre style={{ color: "crimson" }}>{error}</pre>
      ) : loading ? (
        <p>加载中...</p>
      ) : data ? (
        <pre>{JSON.stringify(data, null, 2)}</pre>
      ) : (
        <p>健康检查返回为空。</p>
      )}
    </main>
  );
}
