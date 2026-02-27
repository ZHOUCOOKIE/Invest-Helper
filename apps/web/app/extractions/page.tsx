"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

type RawPost = {
  id: number;
  platform: string;
  author_handle: string;
  external_id: string;
  url: string;
  content_text: string;
  posted_at: string;
  fetched_at: string;
  raw_json: Record<string, unknown> | null;
};

type ExtractionFilterStatus = "all" | "pending" | "approved" | "rejected";

type Extraction = {
  id: number;
  raw_post_id: number;
  status: "pending" | "approved" | "rejected";
  extracted_json: Record<string, unknown>;
  model_name: string;
  reviewed_at: string | null;
  reviewed_by: string | null;
  review_note: string | null;
  applied_kol_view_id: number | null;
  created_at: string;
  raw_post: RawPost;
};

type ClearPendingResponse = {
  deleted_extractions_count: number;
  deleted_raw_posts_count: number;
  scoped_author_handle: string | null;
};

type XProgress = {
  scope: string;
  author_handle: string | null;
  total_raw_posts: number;
  extracted_success_count: number;
  pending_count: number;
  failed_count: number;
  no_extraction_count: number;
  latest_error_summary: string | null;
  latest_extraction_at: string | null;
};

type BackfillAutoReviewResponse = {
  scanned: number;
  approved_count: number;
  rejected_count: number;
  skipped_no_result_count: number;
  skipped_no_confidence_count: number;
  skipped_already_terminal_count: number;
  errors: Array<{
    extraction_id: number | null;
    raw_post_id: number | null;
    error: string;
  }>;
};

type ExtractionsStats = {
  bad_count: number;
  total_count: number;
};

type RefreshWrongJsonResponse = {
  scanned: number;
  updated: number;
  dry_run: boolean;
  updated_ids: number[];
};

type ReextractPendingResponse = {
  scanned: number;
  created: number;
  triggered: number;
  conflict_count: number;
  succeeded_parse: number;
  failed_parse: number;
  auto_approved: number;
  auto_rejected: number;
  noneany_rejected: number;
  skipped_terminal: number;
  errors: Array<{
    extraction_id: number | null;
    raw_post_id: number | null;
    error: string;
  }>;
};

const PAGE_SIZE = 20;
const statusOptions: ExtractionFilterStatus[] = ["all", "pending", "approved", "rejected"];

function summarizeExtraction(extracted: Record<string, unknown>): string {
  if (typeof extracted.summary === "string" && extracted.summary.trim()) {
    return extracted.summary;
  }

  const candidates = extracted.candidates;
  if (Array.isArray(candidates) && candidates.length > 0) {
    const first = candidates[0] as Record<string, unknown>;
    if (typeof first?.summary === "string" && first.summary.trim()) {
      return first.summary;
    }
  }

  return "(no summary)";
}

function autoRejectInfo(extracted: Record<string, unknown>): { reason: string; threshold: string } | null {
  const rawMeta = extracted.meta;
  if (!rawMeta || typeof rawMeta !== "object") return null;
  const meta = rawMeta as Record<string, unknown>;
  if (meta.auto_rejected !== true) return null;
  return {
    reason: String(meta.auto_reject_reason ?? "-"),
    threshold: String(meta.auto_reject_threshold ?? "-"),
  };
}

function parseUrlNumericId(url: string): string | null {
  const trimmed = url.trim();
  if (!trimmed) return null;
  try {
    const { pathname } = new URL(trimmed);
    const segments = pathname.split("/").filter(Boolean);
    for (let i = segments.length - 1; i >= 0; i -= 1) {
      if (/^\d+$/.test(segments[i])) return segments[i];
    }
    return null;
  } catch {
    const segments = trimmed.split(/[/?#]/).filter(Boolean);
    for (let i = segments.length - 1; i >= 0; i -= 1) {
      if (/^\d+$/.test(segments[i])) return segments[i];
    }
    return null;
  }
}

function buildPublicId(item: Extraction): string {
  const platform = (item.raw_post.platform || "unknown").trim().toLowerCase() || "unknown";
  const externalId = (item.raw_post.external_id || "").trim();
  const parsedId = parseUrlNumericId(item.raw_post.url || "");
  const fallbackId = String(item.raw_post_id || item.id);
  const stableId = externalId || parsedId || fallbackId || String(item.id);
  return `${platform}:${stableId}`;
}

function getInitialStatus(searchStatus: string | null): ExtractionFilterStatus {
  if (searchStatus === "pending" || searchStatus === "approved" || searchStatus === "rejected" || searchStatus === "all") {
    return searchStatus;
  }
  return "all";
}

export default function ExtractionsPage() {
  const [searchReady, setSearchReady] = useState(false);
  const [status, setStatus] = useState<ExtractionFilterStatus>("all");
  const [q, setQ] = useState("");
  const [badOnly, setBadOnly] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [offset, setOffset] = useState(0);
  const [items, setItems] = useState<Extraction[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [clearAlsoDeleteRawPosts, setClearAlsoDeleteRawPosts] = useState(false);
  const [progress, setProgress] = useState<XProgress | null>(null);
  const [backfillBusy, setBackfillBusy] = useState(false);
  const [backfillResult, setBackfillResult] = useState<BackfillAutoReviewResponse | null>(null);
  const [stats, setStats] = useState<ExtractionsStats | null>(null);
  const [refreshWrongBusy, setRefreshWrongBusy] = useState(false);
  const [reextractPendingBusy, setReextractPendingBusy] = useState(false);
  const requestSeqRef = useRef(0);

  const canPrev = useMemo(() => offset > 0, [offset]);
  const canNext = useMemo(() => items.length === PAGE_SIZE, [items.length]);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    setStatus(getInitialStatus(params.get("status")));
    setQ(params.get("q") ?? "");
    setBadOnly(params.get("bad_only") === "true");
    setShowHistory(params.get("show_history") === "true");
    setMessage(params.get("msg"));
    setSearchReady(true);
  }, []);

  const load = useCallback(async () => {
    if (!searchReady) return;
    const requestSeq = requestSeqRef.current + 1;
    requestSeqRef.current = requestSeq;
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({
        status,
        limit: String(PAGE_SIZE),
        offset: String(offset),
        q: q.trim(),
        bad_only: badOnly ? "true" : "false",
        show_history: showHistory ? "true" : "false",
      });
      const res = await fetch(`/api/extractions?${params.toString()}`, {
        cache: "no-store",
      });
      const body = (await res.json()) as Extraction[] | { detail?: string };
      if (!res.ok) {
        throw new Error("detail" in body ? (body.detail ?? "Load failed") : "Load failed");
      }
      if (requestSeq !== requestSeqRef.current) return;
      const dedupedMap = new Map<number, Extraction>();
      for (const item of body as Extraction[]) {
        dedupedMap.set(item.id, item);
      }
      const deduped = Array.from(dedupedMap.values()).sort((a, b) => {
        if (a.created_at === b.created_at) return b.id - a.id;
        return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
      });
      setItems(deduped);
    } catch (err) {
      if (requestSeq !== requestSeqRef.current) return;
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      if (requestSeq === requestSeqRef.current) {
        setLoading(false);
      }
    }
  }, [badOnly, offset, q, searchReady, showHistory, status]);

  useEffect(() => {
    void load();
  }, [load]);

  const refreshStats = useCallback(async () => {
    const res = await fetch("/api/extractions/stats", { cache: "no-store" });
    const body = (await res.json()) as ExtractionsStats | { detail?: string };
    if (!res.ok) {
      throw new Error("detail" in body ? (body.detail ?? "Load stats failed") : "Load stats failed");
    }
    setStats(body as ExtractionsStats);
  }, []);

  const refreshProgress = useCallback(async () => {
    const res = await fetch("/api/ingest/x/progress", { cache: "no-store" });
    const body = (await res.json()) as XProgress | { detail?: string };
    if (!res.ok) {
      throw new Error("detail" in body ? (body.detail ?? "Load progress failed") : "Load progress failed");
    }
    setProgress(body as XProgress);
  }, []);

  useEffect(() => {
    const loadAux = async () => {
      try {
        await Promise.all([refreshProgress(), refreshStats()]);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Load progress/stats failed");
      }
    };
    void loadAux();
  }, [refreshProgress, refreshStats]);

  const rejectInline = async (extractionId: number) => {
    const reason = window.prompt("拒绝原因（可选）：", "") ?? "";
    setError(null);
    setMessage(null);
    try {
      const res = await fetch(`/api/extractions/${extractionId}/reject`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reason: reason.trim() || null }),
      });
      const body = (await res.json()) as { detail?: string };
      if (!res.ok) {
        throw new Error(body.detail ?? `Reject failed: ${res.status}`);
      }
      setMessage(`Extraction #${extractionId} rejected.`);
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Reject failed");
    }
  };

  const reextractAllPendingForce = async () => {
    const confirmText = window.prompt("将强制重试所有待审核记录并生成新 extraction。输入 YES 继续：", "");
    if (confirmText !== "YES") return;
    setReextractPendingBusy(true);
    setError(null);
    setMessage(null);
    try {
      const res = await fetch("/api/admin/extractions/reextract-pending?confirm=YES", { method: "POST" });
      const rawText = await res.text();
      let body: ReextractPendingResponse | { detail?: string } | null = null;
      try {
        body = rawText ? (JSON.parse(rawText) as ReextractPendingResponse | { detail?: string }) : null;
      } catch {
        body = null;
      }
      if (!res.ok) {
        const detail =
          body && typeof body === "object" && "detail" in body
            ? (body.detail ?? "重试待审核失败")
            : (rawText || `HTTP ${res.status}`).slice(0, 200);
        throw new Error(detail);
      }
      const done = body as ReextractPendingResponse;
      const shouldReloadNow = offset === 0;
      setOffset(0);
      setMessage(
        `已完成：scanned=${done.scanned}, approved=${done.auto_approved}, rejected=${done.auto_rejected}, failed_parse=${done.failed_parse}, noneany_rejected=${done.noneany_rejected}, conflict=${done.conflict_count}.`,
      );
      await Promise.all([refreshProgress(), refreshStats()]);
      if (shouldReloadNow) {
        await load();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "重试待审核失败");
    } finally {
      setReextractPendingBusy(false);
    }
  };

  const clearPending = async () => {
    const confirmText = window.prompt(
      "Dangerous operation: this will hard delete pending/failed extractions. Type YES to continue.",
      "",
    );
    if (confirmText !== "YES") return;
    setError(null);
    setMessage(null);
    try {
      const params = new URLSearchParams({
        confirm: "YES",
        also_delete_raw_posts: clearAlsoDeleteRawPosts ? "true" : "false",
      });
      const res = await fetch(`/api/admin/extractions/pending?${params.toString()}`, { method: "DELETE" });
      const body = (await res.json()) as ClearPendingResponse | { detail?: string };
      if (!res.ok) {
        throw new Error("detail" in body ? (body.detail ?? "Clear pending failed") : "Clear pending failed");
      }
      const done = body as ClearPendingResponse;
      setMessage(
        `Deleted extractions=${done.deleted_extractions_count}, raw_posts=${done.deleted_raw_posts_count}.`,
      );
      await Promise.all([load(), refreshProgress()]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Clear pending failed");
    }
  };

  const backfillAutoReview = async () => {
    const confirmText = window.prompt(
      "This updates historical extractions by confidence only, with no model calls. Type YES to continue.",
      "",
    );
    if (confirmText !== "YES") return;
    setBackfillBusy(true);
    setError(null);
    setMessage(null);
    setBackfillResult(null);
    try {
      const res = await fetch("/api/admin/extractions/backfill-auto-review?confirm=YES", { method: "POST" });
      const body = (await res.json()) as BackfillAutoReviewResponse | { detail?: string };
      if (!res.ok) {
        throw new Error("detail" in body ? (body.detail ?? "Backfill auto review failed") : "Backfill auto review failed");
      }
      const done = body as BackfillAutoReviewResponse;
      setBackfillResult(done);
      setMessage(
        `Backfill done: scanned=${done.scanned}, approved=${done.approved_count}, rejected=${done.rejected_count}.`,
      );
      await Promise.all([load(), refreshProgress()]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Backfill auto review failed");
    } finally {
      setBackfillBusy(false);
    }
  };

  const refreshWrongExtractedJson = async () => {
    const confirmText = window.prompt(
      "This refreshes approved wrong extracted_json from applied views. Type YES to continue.",
      "",
    );
    if (confirmText !== "YES") return;
    setRefreshWrongBusy(true);
    setError(null);
    setMessage(null);
    try {
      const res = await fetch(
        "/api/admin/extractions/refresh-wrong-extracted-json?confirm=YES&days=365&limit=2000&dry_run=false",
        { method: "POST" },
      );
      const body = (await res.json()) as RefreshWrongJsonResponse | { detail?: string };
      if (!res.ok) {
        throw new Error("detail" in body ? (body.detail ?? "Refresh wrong extracted_json failed") : "Refresh wrong extracted_json failed");
      }
      const done = body as RefreshWrongJsonResponse;
      setMessage(
        `Refresh done: scanned=${done.scanned}, updated=${done.updated}, ids=${done.updated_ids.slice(0, 10).join(",") || "-"}.`,
      );
      await Promise.all([load(), refreshStats(), refreshProgress()]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Refresh wrong extracted_json failed");
    } finally {
      setRefreshWrongBusy(false);
    }
  };

  return (
    <main style={{ padding: "24px", fontFamily: "monospace" }}>
      <h1>Extraction 审核</h1>
      <p>
        默认仅展示每个 raw_post 最新 extraction；开启 history 才会显示历史版本。auto-review 阈值=70，手动 force re-extract 默认进入待人工审核。
      </p>

      <div style={{ display: "flex", gap: "8px", alignItems: "center", marginBottom: "12px", flexWrap: "wrap" }}>
        <label>
          status
          <select
            value={status}
            onChange={(event) => {
              setStatus(event.target.value as ExtractionFilterStatus);
              setOffset(0);
            }}
            style={{ marginLeft: "8px" }}
          >
            {statusOptions.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </label>
        <label>
          q
          <input
            value={q}
            onChange={(event) => {
              setQ(event.target.value);
              setOffset(0);
            }}
            placeholder="keyword"
            style={{ marginLeft: "8px" }}
          />
        </label>
        <label>
          <input
            type="checkbox"
            checked={badOnly}
            onChange={(event) => {
              setBadOnly(event.target.checked);
              setOffset(0);
            }}
          />{" "}
          仅显示异常
        </label>
        <label>
          <input
            type="checkbox"
            checked={showHistory}
            onChange={(event) => {
              setShowHistory(event.target.checked);
              setOffset(0);
            }}
          />{" "}
          显示历史版本
        </label>
        <button type="button" onClick={() => void load()} disabled={loading}>
          {loading ? "加载中..." : "刷新"}
        </button>
        <button type="button" onClick={() => void refreshStats()} disabled={loading}>
          刷新统计
        </button>
        {status === "pending" && (
          <>
            <button
              type="button"
              onClick={() => void reextractAllPendingForce()}
              disabled={loading || reextractPendingBusy}
            >
              {reextractPendingBusy ? "执行中..." : "重试所有待审核（强制）"}
            </button>
            <button type="button" onClick={() => void backfillAutoReview()} disabled={loading || backfillBusy}>
              {backfillBusy ? "执行中..." : "回填自动审核（置信度规则）"}
            </button>
            <label>
              <input
                type="checkbox"
                checked={clearAlsoDeleteRawPosts}
                onChange={(event) => setClearAlsoDeleteRawPosts(event.target.checked)}
              />{" "}
              同时删除 raw posts
            </label>
            <button type="button" onClick={() => void clearPending()} disabled={loading}>
              清空待处理（删除）
            </button>
            <button type="button" onClick={() => void refreshWrongExtractedJson()} disabled={loading || refreshWrongBusy}>
              {refreshWrongBusy ? "执行中..." : "刷新错误 extracted_json"}
            </button>
          </>
        )}
      </div>

      {message && <p style={{ color: "green" }}>{message}</p>}
      {error && <p style={{ color: "crimson" }}>{error}</p>}
      {progress && (
        <p style={{ color: "#555" }}>
          progress[{progress.scope}] total={progress.total_raw_posts}, success={progress.extracted_success_count}, pending=
          {progress.pending_count}, failed={progress.failed_count}, no_extraction={progress.no_extraction_count}
        </p>
      )}
      {stats && <p style={{ color: "#555" }}>bad_json_count={stats.bad_count} / total={stats.total_count}</p>}
      {backfillResult && (
        <div style={{ border: "1px solid #eee", borderRadius: "8px", padding: "10px", marginBottom: "10px" }}>
          <div>
            scanned={backfillResult.scanned}, approved={backfillResult.approved_count}, rejected={backfillResult.rejected_count}
          </div>
          <div>
            skipped_no_result={backfillResult.skipped_no_result_count}, skipped_no_confidence=
            {backfillResult.skipped_no_confidence_count}, skipped_already_terminal=
            {backfillResult.skipped_already_terminal_count}
          </div>
          {backfillResult.errors.length > 0 && (
            <details style={{ marginTop: "6px" }}>
              <summary>errors (showing first {Math.min(20, backfillResult.errors.length)})</summary>
              <ul>
                {backfillResult.errors.slice(0, 20).map((item, index) => (
                  <li key={`${item.extraction_id ?? "na"}-${index}`}>
                    extraction_id={item.extraction_id ?? "null"}, raw_post_id={item.raw_post_id ?? "null"}: {item.error}
                  </li>
                ))}
              </ul>
            </details>
          )}
        </div>
      )}

      {!loading && items.length === 0 && <p>暂无 extraction。</p>}

      <div style={{ display: "grid", gap: "10px" }}>
        {items.map((item, idx) => {
          const autoRejected = autoRejectInfo(item.extracted_json);
          const serialNo = offset + idx + 1;
          const publicId = buildPublicId(item);
          return (
            <article key={item.id} style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "10px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", gap: "8px", marginBottom: "6px" }}>
                <strong>No.{serialNo} [{item.status}] public_id={publicId}</strong>
                <small>created_at: {item.created_at}</small>
              </div>
              <div>
                <small>extraction_id={item.id}</small>
              </div>
              <div>
                {item.raw_post.platform} / @{item.raw_post.author_handle}
              </div>
              <div>
                <a href={item.raw_post.url} target="_blank" rel="noreferrer">
                  {item.raw_post.url}
                </a>
              </div>
              <div>posted_at: {item.raw_post.posted_at}</div>
              <div style={{ marginTop: "6px" }}>summary: {summarizeExtraction(item.extracted_json)}</div>
              {autoRejected && (
                <div style={{ marginTop: "4px", color: "#8a5800" }}>
                  auto_rejected=true, reason={autoRejected.reason}, threshold={autoRejected.threshold}
                </div>
              )}
              <div style={{ display: "flex", gap: "8px", marginTop: "8px" }}>
                <Link href={`/extractions/${item.id}`}>通过</Link>
                <button type="button" onClick={() => void rejectInline(item.id)} disabled={item.status !== "pending"}>
                  拒绝
                </button>
              </div>
            </article>
          );
        })}
      </div>

      <div style={{ display: "flex", gap: "8px", marginTop: "12px" }}>
        <button
          type="button"
          onClick={() => setOffset((prev) => Math.max(0, prev - PAGE_SIZE))}
          disabled={!canPrev || loading}
        >
          上一页
        </button>
        <button type="button" onClick={() => setOffset((prev) => prev + PAGE_SIZE)} disabled={!canNext || loading}>
          下一页
        </button>
        <small>offset={offset}</small>
      </div>
    </main>
  );
}
