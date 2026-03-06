"use client";

import { useParams, useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useState } from "react";

type DigestPostSummary = {
  raw_post_id: number;
  extraction_id: number;
  kol_id: number | null;
  business_ts: string;
  time_field_used: "as_of" | "posted_at" | "created_at";
  posted_at: string | null;
  author_handle: string;
  author_display_name: string | null;
  title: string | null;
  source_url: string;
  summary: string;
};

type DigestAIAnalysis = {
  market_overview: string;
  market_signals: string;
  focus_points: string[];
  key_news: string[];
  trading_observations: string | null;
};

type DailyDigest = {
  id: number;
  profile_id: number;
  digest_date: string;
  generated_at: string;
  post_summaries: DigestPostSummary[];
  ai_analysis: DigestAIAnalysis;
  metadata: {
    generated_at: string;
    window_start: string;
    window_end: string;
    source_post_count: number;
    ai_status: string;
    ai_error: string | null;
    time_field_priority: Array<"as_of" | "posted_at" | "created_at">;
  };
};

type ApiErrorBody = {
  request_id?: string;
  message?: string;
  detail?: unknown;
};

type ParsedApiResponse<T> = {
  data: T | ApiErrorBody | null;
  textBody: string;
  requestId: string | null;
  statusCode: number;
  requestPath: string;
};

type AdminHardDeleteResponse = {
  counts: Record<string, number>;
};

function formatDate(d: Date): string {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

function getRecent7Days(): string[] {
  const today = new Date();
  const rows: string[] = [];
  for (let i = 0; i < 7; i += 1) {
    const d = new Date(today);
    d.setDate(today.getDate() - i);
    rows.push(formatDate(d));
  }
  return rows;
}

function displayRawTime(value: string | null | undefined): string {
  if (typeof value !== "string") return "-";
  const trimmed = value.trim();
  if (!trimmed) return "-";
  return trimmed.replace(/\s?(?:Z|[+-]\d{2}:\d{2})$/i, "").trim();
}

function isDailyDigest(value: unknown): value is DailyDigest {
  return Boolean(
    value &&
      typeof value === "object" &&
      "id" in (value as Record<string, unknown>) &&
      "digest_date" in (value as Record<string, unknown>),
  );
}

async function parseApiResponse<T>(res: Response): Promise<ParsedApiResponse<T>> {
  const requestId = res.headers.get("x-request-id");
  let requestPath = res.url || "unknown";
  try {
    const parsedUrl = new URL(res.url);
    requestPath = `${parsedUrl.pathname}${parsedUrl.search}`;
  } catch {
    requestPath = res.url || "unknown";
  }
  const statusCode = res.status;
  const textBody = await res.text();
  try {
    const parsed = JSON.parse(textBody) as T | ApiErrorBody;
    if (parsed && typeof parsed === "object") {
      const bodyRequestId = (parsed as ApiErrorBody).request_id;
      return { data: parsed, textBody: "", requestId: bodyRequestId ?? requestId, statusCode, requestPath };
    }
    return { data: parsed, textBody: "", requestId, statusCode, requestPath };
  } catch {
    return { data: null, textBody, requestId, statusCode, requestPath };
  }
}

function formatApiError(
  fallbackMessage: string,
  body: unknown,
  textBody: string,
  requestId: string | null,
  statusCode?: number,
  requestPath?: string,
): string {
  let message = fallbackMessage;
  if (body && typeof body === "object") {
    const obj = body as Record<string, unknown>;
    if (typeof obj.message === "string" && obj.message) message = obj.message;
    else if (typeof obj.detail === "string" && obj.detail) message = obj.detail;
  } else if (textBody) {
    const statusPart = typeof statusCode === "number" ? `状态=${statusCode}` : "状态=未知";
    const pathPart = requestPath ? `路径=${requestPath}` : "路径=未知";
    message = `非 JSON 错误响应（${statusPart}, ${pathPart}）：${textBody.slice(0, 300)}`;
  }
  return requestId ? `${message} (request_id=${requestId})` : message;
}

export default function DailyDigestPage() {
  const params = useParams<{ date: string }>();
  const router = useRouter();
  const digestDate = params?.date;

  const [availableDates, setAvailableDates] = useState<string[]>([]);
  const [digest, setDigest] = useState<DailyDigest | null>(null);
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [notFound, setNotFound] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [cleanupDigestDate, setCleanupDigestDate] = useState("");
  const [cleanupBusy, setCleanupBusy] = useState(false);
  const [cleanupError, setCleanupError] = useState<string | null>(null);
  const [cleanupMessage, setCleanupMessage] = useState<string | null>(null);

  const recent7Days = useMemo(() => getRecent7Days(), []);
  const displayDates = useMemo(() => {
    return Array.from(new Set([...availableDates, ...recent7Days])).sort((a, b) => b.localeCompare(a));
  }, [availableDates, recent7Days]);

  const loadDigestDates = useCallback(async () => {
    const res = await fetch("/api/digests/dates", { cache: "no-store" });
    const parsed = await parseApiResponse<string[]>(res);
    if (!res.ok || !Array.isArray(parsed.data)) {
      throw new Error(
        formatApiError("加载日报日期失败", parsed.data, parsed.textBody, parsed.requestId, parsed.statusCode, parsed.requestPath),
      );
    }
    const rows = (parsed.data as unknown[])
      .filter((item): item is string => typeof item === "string")
      .map((item) => item.slice(0, 10));
    setAvailableDates(rows);
  }, []);

  const loadDigest = useCallback(
    async (): Promise<DailyDigest | null> => {
      if (!digestDate) return null;
      const res = await fetch(`/api/digests?date=${digestDate}`, { cache: "no-store" });
      const parsed = await parseApiResponse<DailyDigest>(res);
      if (!res.ok) {
        const message = formatApiError(
          "加载日报失败",
          parsed.data,
          parsed.textBody,
          parsed.requestId,
          parsed.statusCode,
          parsed.requestPath,
        );
        if (res.status === 404 && message.toLowerCase().includes("digest not found")) {
          setDigest(null);
          setNotFound(true);
          return null;
        }
        throw new Error(message);
      }
      if (!isDailyDigest(parsed.data)) {
        throw new Error(
          formatApiError(
            "加载日报失败：返回格式无效",
            parsed.data,
            parsed.textBody,
            parsed.requestId,
            parsed.statusCode,
            parsed.requestPath,
          ),
        );
      }
      const loadedDigest = parsed.data as DailyDigest;
      setDigest(loadedDigest);
      setNotFound(false);
      return loadedDigest;
    },
    [digestDate],
  );

  const load = useCallback(async () => {
    if (!digestDate) return;
    setLoading(true);
    setError(null);
    setNotFound(false);
    try {
      await Promise.all([loadDigest(), loadDigestDates()]);
    } catch (err) {
      setDigest(null);
      setError(err instanceof Error ? err.message : "加载日报失败");
    } finally {
      setLoading(false);
    }
  }, [digestDate, loadDigest, loadDigestDates]);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    setCleanupDigestDate(digestDate ?? "");
  }, [digestDate]);

  const generate = async () => {
    if (!digestDate) return;
    setGenerating(true);
    setError(null);
    setNotFound(false);
    try {
      const res = await fetch(`/api/digests/generate?date=${digestDate}`, { method: "POST" });
      const parsed = await parseApiResponse<DailyDigest>(res);
      if (!res.ok || !isDailyDigest(parsed.data)) {
        throw new Error(
          formatApiError("生成日报失败", parsed.data, parsed.textBody, parsed.requestId, parsed.statusCode, parsed.requestPath),
        );
      }
      setDigest(parsed.data as DailyDigest);
      await Promise.all([loadDigest(), loadDigestDates()]);
      router.refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "生成日报失败");
      try {
        await Promise.all([loadDigest(), loadDigestDates()]);
        router.refresh();
      } catch {
        // keep original generate error when refresh fallback also fails
      }
    } finally {
      setGenerating(false);
    }
  };

  const askYes = (message: string): boolean => {
    const confirmText = window.prompt(message, "");
    return confirmText === "YES";
  };

  const deleteDigestByDateCleanup = async () => {
    if (!cleanupDigestDate) {
      setCleanupError("请选择 digest_date");
      return;
    }
    if (!askYes("Dangerous operation: delete digest by date. Type YES to continue.")) return;
    setCleanupBusy(true);
    setCleanupError(null);
    setCleanupMessage(null);
    try {
      const params = new URLSearchParams({
        confirm: "YES",
        digest_date: cleanupDigestDate,
      });
      const res = await fetch(`/api/admin/digests?${params.toString()}`, { method: "DELETE" });
      const body = (await res.json()) as AdminHardDeleteResponse | { detail?: string };
      if (!res.ok) {
        throw new Error("detail" in body ? (body.detail ?? "删除日报失败") : "删除日报失败");
      }
      const done = body as AdminHardDeleteResponse;
      setCleanupMessage(`日报清理完成：${JSON.stringify(done.counts)}`);
      await load();
    } catch (err) {
      setCleanupError(err instanceof Error ? err.message : "删除日报失败");
    } finally {
      setCleanupBusy(false);
    }
  };

  return (
    <main style={{ padding: "24px", fontFamily: "monospace", display: "grid", gap: "14px" }}>
      <h1>每日日报</h1>

      <section style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "12px", display: "grid", gap: "8px" }}>
        <div style={{ display: "flex", gap: "12px", alignItems: "center", flexWrap: "wrap" }}>
          <strong>日期: {digestDate}</strong>
          <button type="button" onClick={() => void generate()} disabled={generating}>
            {generating ? "生成中..." : "生成日报"}
          </button>
          <button type="button" onClick={() => void load()} disabled={loading}>
            {loading ? "加载中..." : "刷新"}
          </button>
        </div>
        <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
          {displayDates.map((d) => (
            <button
              key={d}
              type="button"
              onClick={() => router.push(`/digests/${d}`)}
              style={{
                border: "1px solid #ccc",
                borderRadius: "6px",
                padding: "4px 8px",
                background: d === digestDate ? "#111" : "#fff",
                color: d === digestDate ? "#fff" : "#111",
              }}
            >
              {d}
            </button>
          ))}
        </div>
      </section>

      {loading && <p>正在加载日报...</p>}
      {error && <p style={{ color: "crimson" }}>{error}</p>}
      {!loading && !error && notFound && <p>未找到该日报。</p>}

      {!loading && !error && digest && (
        <>
          <section style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "12px" }}>
            <div>
              <strong>日期:</strong> {digest.digest_date}
            </div>
            <div>
              <strong>生成时间:</strong> {displayRawTime(digest.generated_at)}
            </div>
            <small style={{ color: "#666" }}>
              时间窗: {displayRawTime(digest.metadata.window_start)} ~ {displayRawTime(digest.metadata.window_end)} | 贴文数: {digest.metadata.source_post_count} | AI状态: {digest.metadata.ai_status}
            </small>
            {digest.metadata.ai_error && <div style={{ color: "#a00" }}>AI错误: {digest.metadata.ai_error}</div>}
          </section>

          <section style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "12px" }}>
            <h2 style={{ marginTop: 0 }}>第一部分：AI 中文日报分析</h2>
            <div style={{ display: "grid", gap: "12px" }}>
              <article style={{ padding: "10px 12px", lineHeight: 1.7, overflowWrap: "anywhere" }}>
                <strong>市场概览</strong>
                <p>{digest.ai_analysis.market_overview || "(无)"}</p>
              </article>
              <article style={{ padding: "10px 12px", lineHeight: 1.7, overflowWrap: "anywhere" }}>
                <strong>行情提示</strong>
                <p>{digest.ai_analysis.market_signals || "(无)"}</p>
              </article>
              <article style={{ padding: "10px 12px", lineHeight: 1.7, overflowWrap: "anywhere" }}>
                <strong>关注重点</strong>
                {digest.ai_analysis.focus_points.length === 0 ? <p>(无)</p> : <ul>{digest.ai_analysis.focus_points.map((x, i) => <li key={`${i}-${x}`}>{x}</li>)}</ul>}
              </article>
              <article style={{ padding: "10px 12px", lineHeight: 1.7, overflowWrap: "anywhere" }}>
                <strong>要闻提炼</strong>
                {digest.ai_analysis.key_news.length === 0 ? <p>(无)</p> : <ul>{digest.ai_analysis.key_news.map((x, i) => <li key={`${i}-${x}`}>{x}</li>)}</ul>}
              </article>
              <article style={{ padding: "10px 12px", lineHeight: 1.7, overflowWrap: "anywhere" }}>
                <strong>交易观察</strong>
                <p>{digest.ai_analysis.trading_observations || "信息不足，未输出交易结论。"}</p>
              </article>
            </div>
          </section>

          <section style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "12px" }}>
            <h2 style={{ marginTop: 0 }}>第二部分：时间顺序贴文摘要流</h2>
            {digest.post_summaries.length === 0 ? (
              <p>当天+昨天没有 hasview=1 的贴文。</p>
            ) : (
              <div style={{ display: "grid", gap: "10px" }}>
                {digest.post_summaries.map((item) => (
                  <article key={`${item.raw_post_id}-${item.extraction_id}`} style={{ border: "1px solid #eee", borderRadius: "8px", padding: "10px" }}>
                    <div>
                      <strong>{displayRawTime(item.posted_at ?? item.business_ts)}</strong> {item.author_display_name || item.author_handle}
                    </div>
                    {item.title && <div>标题: {item.title}</div>}
                    <div style={{ marginTop: "4px" }}>{item.summary}</div>
                    <small style={{ color: "#666" }}>
                      来源: {item.source_url ? (
                        <a href={item.source_url} target="_blank" rel="noreferrer">
                          链接
                        </a>
                      ) : (
                        "-"
                      )}
                    </small>
                  </article>
                ))}
              </div>
            )}
          </section>
        </>
      )}

      <section style={{ border: "1px solid #eee", borderRadius: "8px", padding: "12px", maxWidth: "560px" }}>
        <h2 style={{ marginTop: 0, marginBottom: "8px" }}>管理清理 - 按日期删除日报</h2>
        <p style={{ marginTop: 0 }}>所有操作都需要输入 YES。</p>
        <label>
          日报日期
          <input
            type="date"
            value={cleanupDigestDate}
            onChange={(event) => setCleanupDigestDate(event.target.value)}
            style={{ display: "block", width: "100%" }}
          />
        </label>
        <button type="button" onClick={() => void deleteDigestByDateCleanup()} disabled={cleanupBusy} style={{ marginTop: "8px" }}>
          {cleanupBusy ? "删除中..." : "按日期删除日报"}
        </button>
        {cleanupError && <p style={{ color: "crimson", marginBottom: 0 }}>{cleanupError}</p>}
        {cleanupMessage && <p style={{ marginBottom: 0 }}>{cleanupMessage}</p>}
      </section>
    </main>
  );
}
