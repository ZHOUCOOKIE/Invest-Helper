"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

type WeeklyDigestKind = "recent_week" | "this_week" | "last_week";

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

type WeeklyDigest = {
  id: number;
  profile_id: number;
  report_kind: WeeklyDigestKind;
  anchor_date: string;
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
  };
};

type ApiErrorBody = {
  detail?: string;
  message?: string;
  request_id?: string;
};

type ParsedApiResponse<T> = {
  data: T | ApiErrorBody | null;
  textBody: string;
  requestId: string | null;
  statusCode: number;
  requestPath: string;
};

function formatDate(d: Date): string {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

function parseDateInput(value: string): Date | null {
  if (!value) return null;
  const [yearRaw, monthRaw, dayRaw] = value.split("-");
  const year = Number(yearRaw);
  const month = Number(monthRaw);
  const day = Number(dayRaw);
  if (!Number.isFinite(year) || !Number.isFinite(month) || !Number.isFinite(day)) return null;
  const parsed = new Date(year, month - 1, day);
  if (Number.isNaN(parsed.getTime())) return null;
  return parsed;
}

function weekStartSunday(target: Date): Date {
  const d = new Date(target);
  const daysSinceSunday = d.getDay();
  d.setDate(d.getDate() - daysSinceSunday);
  return d;
}

function inferAnchorDate(kind: WeeklyDigestKind): string {
  const today = new Date();
  if (kind === "recent_week") return formatDate(today);
  const thisSunday = weekStartSunday(today);
  if (kind === "this_week") return formatDate(thisSunday);
  const lastSunday = new Date(thisSunday);
  lastSunday.setDate(thisSunday.getDate() - 7);
  return formatDate(lastSunday);
}

function toReferenceDateForGenerate(kind: WeeklyDigestKind, anchorDate: string): string | null {
  const parsed = parseDateInput(anchorDate);
  if (!parsed) return null;
  if (kind === "recent_week") return formatDate(parsed);
  if (kind === "this_week") return formatDate(parsed);
  const reference = new Date(parsed);
  reference.setDate(reference.getDate() + 7);
  return formatDate(reference);
}

function displayKindLabel(kind: WeeklyDigestKind): string {
  if (kind === "recent_week") return "近一周周报";
  if (kind === "this_week") return "本周周报";
  return "上周周报";
}

function displayLocalTime(value: string | null | undefined): string {
  if (!value || typeof value !== "string") return "-";
  const dt = new Date(value);
  if (Number.isNaN(dt.getTime())) return value;
  return new Intl.DateTimeFormat(undefined, {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(dt);
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

async function wait(ms: number): Promise<void> {
  await new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

export default function WeeklyDigestPage() {
  const [kind, setKind] = useState<WeeklyDigestKind>("recent_week");
  const [anchorDate, setAnchorDate] = useState(inferAnchorDate("recent_week"));
  const [availableDates, setAvailableDates] = useState<string[]>([]);
  const [digest, setDigest] = useState<WeeklyDigest | null>(null);
  const [loading, setLoading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notFound, setNotFound] = useState(false);

  const kindOptions: Array<{ value: WeeklyDigestKind; label: string }> = useMemo(
    () => [
      { value: "recent_week", label: "近一周周报" },
      { value: "this_week", label: "本周周报" },
      { value: "last_week", label: "上周周报" },
    ],
    [],
  );

  const loadDates = useCallback(async (k: WeeklyDigestKind) => {
    const res = await fetch(`/api/weekly-digests/dates?kind=${k}`, { cache: "no-store" });
    const parsed = await parseApiResponse<string[]>(res);
    if (!res.ok || !Array.isArray(parsed.data)) {
      throw new Error(formatApiError("加载周报日期失败", parsed.data, parsed.textBody, parsed.requestId, parsed.statusCode, parsed.requestPath));
    }
    const rows = parsed.data.filter((item): item is string => typeof item === "string").map((item) => item.slice(0, 10));
    setAvailableDates(rows);
    return rows;
  }, []);

  const loadDigest = useCallback(async (k: WeeklyDigestKind, dateStr: string) => {
    const res = await fetch(`/api/weekly-digests?kind=${k}&anchor_date=${dateStr}`, { cache: "no-store" });
    const parsed = await parseApiResponse<WeeklyDigest>(res);
    if (!res.ok) {
      if (res.status === 404) {
        setDigest(null);
        setNotFound(true);
        return;
      }
      throw new Error(formatApiError("加载周报失败", parsed.data, parsed.textBody, parsed.requestId, parsed.statusCode, parsed.requestPath));
    }
    if (!parsed.data || typeof parsed.data !== "object" || !("id" in parsed.data)) {
      throw new Error(formatApiError("加载周报失败", parsed.data, parsed.textBody, parsed.requestId, parsed.statusCode, parsed.requestPath));
    }
    setDigest(parsed.data as WeeklyDigest);
    setNotFound(false);
  }, []);

  const refreshAll = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const rows = await loadDates(kind);
      const inferredDate = inferAnchorDate(kind);
      const effectiveDate = rows.includes(anchorDate) ? anchorDate : inferredDate;
      setAnchorDate(effectiveDate);
      await loadDigest(kind, effectiveDate);
    } catch (err) {
      setError(err instanceof Error ? err.message : "加载周报失败");
    } finally {
      setLoading(false);
    }
  }, [anchorDate, kind, loadDates, loadDigest]);

  useEffect(() => {
    setAnchorDate(inferAnchorDate(kind));
  }, [kind]);

  useEffect(() => {
    void refreshAll();
  }, [refreshAll]);

  const generate = useCallback(async () => {
    setGenerating(true);
    setError(null);
    try {
      const params = new URLSearchParams({ kind });
      const referenceDate = toReferenceDateForGenerate(kind, anchorDate);
      if (referenceDate) {
        params.set("date", referenceDate);
      }
      const res = await fetch(`/api/weekly-digests/generate?${params.toString()}`, { method: "POST" });
      const parsed = await parseApiResponse<WeeklyDigest>(res);
      if (!res.ok) {
        throw new Error(formatApiError("生成周报失败", parsed.data, parsed.textBody, parsed.requestId, parsed.statusCode, parsed.requestPath));
      }
      if (!parsed.data || typeof parsed.data !== "object" || !("id" in parsed.data)) {
        throw new Error(formatApiError("生成周报失败", parsed.data, parsed.textBody, parsed.requestId, parsed.statusCode, parsed.requestPath));
      }
      setDigest(parsed.data as WeeklyDigest);
      setNotFound(false);
      setAnchorDate((parsed.data as WeeklyDigest).anchor_date);
      await loadDates(kind);
    } catch (err) {
      // The proxy request may fail while backend generation still succeeds; poll briefly to recover.
      let recovered = false;
      for (const delayMs of [1200, 2000, 3000, 4000]) {
        await wait(delayMs);
        try {
          await loadDigest(kind, anchorDate);
          await loadDates(kind);
          recovered = true;
          break;
        } catch {
          // Keep polling; report the original generation error if all retries fail.
        }
      }
      if (!recovered) {
        setError(err instanceof Error ? err.message : "生成周报失败");
      }
    } finally {
      setGenerating(false);
    }
  }, [anchorDate, kind, loadDates, loadDigest]);

  return (
    <main style={{ padding: "24px", fontFamily: "monospace", display: "grid", gap: "14px" }}>
      <h1>每周周报</h1>

      <section style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "12px", display: "grid", gap: "8px" }}>
        <div style={{ display: "flex", gap: "8px", flexWrap: "wrap", alignItems: "center" }}>
          <label>
            周报类型
            <select
              value={kind}
              onChange={(event) => setKind(event.target.value as WeeklyDigestKind)}
              style={{ marginLeft: "8px" }}
            >
              {kindOptions.map((option) => (
                <option key={option.value} value={option.value}>{option.label}</option>
              ))}
            </select>
          </label>
          {kind !== "recent_week" && (
            <label>
              锚点日期
              <input type="date" value={anchorDate} onChange={(event) => setAnchorDate(event.target.value)} style={{ marginLeft: "8px" }} />
            </label>
          )}
          <button type="button" onClick={() => void generate()} disabled={generating}>
            {generating ? "生成中..." : "生成周报"}
          </button>
          <button type="button" onClick={() => void refreshAll()} disabled={loading}>
            {loading ? "加载中..." : "刷新"}
          </button>
        </div>
        {kind !== "recent_week" && (
          <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
            {availableDates.map((d) => (
              <button key={d} type="button" onClick={() => setAnchorDate(d)} style={{ background: d === anchorDate ? "#111" : "#fff", color: d === anchorDate ? "#fff" : "#111" }}>
                {d}
              </button>
            ))}
          </div>
        )}
      </section>

      {error && <p style={{ color: "crimson" }}>{error}</p>}
      {notFound && !error && <p>未找到 {displayKindLabel(kind)}（锚点日期 {anchorDate}）。</p>}

      {digest && !error && (
        <>
          <section style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "12px" }}>
            <div><strong>类型:</strong> {displayKindLabel(digest.report_kind)}</div>
            <div><strong>锚点日期:</strong> {digest.anchor_date}</div>
            <div><strong>生成时间:</strong> {displayLocalTime(digest.generated_at)}</div>
            <small style={{ color: "#666" }}>
              时间窗: {displayLocalTime(digest.metadata.window_start)} ~ {displayLocalTime(digest.metadata.window_end)} | 贴文数: {digest.metadata.source_post_count} | AI状态: {digest.metadata.ai_status}
            </small>
            {digest.metadata.ai_error && <div style={{ color: "#a00" }}>AI错误: {digest.metadata.ai_error}</div>}
          </section>

          <section style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "12px" }}>
            <h2 style={{ marginTop: 0 }}>第一部分：AI 中文周报分析</h2>
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
        </>
      )}
    </main>
  );
}
