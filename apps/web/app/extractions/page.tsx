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

type ExtractionFilterStatus = "all" | "pending" | "approved" | "rejected" | "library";

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

type RecomputeStatusesResponse = {
  scanned: number;
  updated: number;
  dry_run: boolean;
  pending_count: number;
  approved_count: number;
  rejected_count: number;
  skipped_terminal_count: number;
  skipped_no_result_count: number;
  updated_ids: number[];
};

const PAGE_SIZE = 20;
const statusOptions: ExtractionFilterStatus[] = ["all", "pending", "approved", "rejected", "library"];

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
  if (
    searchStatus === "pending" ||
    searchStatus === "approved" ||
    searchStatus === "rejected" ||
    searchStatus === "library" ||
    searchStatus === "all"
  ) {
    return searchStatus;
  }
  return "all";
}

export default function ExtractionsPage() {
  const [searchReady, setSearchReady] = useState(false);
  const [status, setStatus] = useState<ExtractionFilterStatus>("all");
  const [q, setQ] = useState("");
  const [offset, setOffset] = useState(0);
  const [items, setItems] = useState<Extraction[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [clearAlsoDeleteRawPosts, setClearAlsoDeleteRawPosts] = useState(false);
  const [progress, setProgress] = useState<XProgress | null>(null);
  const [recomputeBusy, setRecomputeBusy] = useState(false);
  const requestSeqRef = useRef(0);

  const canPrev = useMemo(() => offset > 0, [offset]);
  const canNext = useMemo(() => items.length === PAGE_SIZE, [items.length]);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    setStatus(getInitialStatus(params.get("status")));
    setQ(params.get("q") ?? "");
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
  }, [offset, q, searchReady, status]);

  useEffect(() => {
    void load();
  }, [load]);

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
        await refreshProgress();
      } catch (err) {
        setError(err instanceof Error ? err.message : "Load progress failed");
      }
    };
    void loadAux();
  }, [refreshProgress]);

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

  const recomputeStatuses = async () => {
    const confirmText = window.prompt(
      "这会按当前规则批量重算 extraction 状态。输入 YES 继续。",
      "",
    );
    if (confirmText !== "YES") return;
    setRecomputeBusy(true);
    setError(null);
    setMessage(null);
    try {
      const res = await fetch(
        "/api/admin/extractions/recompute-statuses?confirm=YES&days=365&limit=5000&dry_run=false",
        { method: "POST" },
      );
      const body = (await res.json()) as RecomputeStatusesResponse | { detail?: string };
      if (!res.ok) {
        throw new Error("detail" in body ? (body.detail ?? "Recompute statuses failed") : "Recompute statuses failed");
      }
      const done = body as RecomputeStatusesResponse;
      setMessage(
        `Recomputed: scanned=${done.scanned}, updated=${done.updated}, pending=${done.pending_count}, approved=${done.approved_count}, rejected=${done.rejected_count}.`,
      );
      await Promise.all([load(), refreshProgress()]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Recompute statuses failed");
    } finally {
      setRecomputeBusy(false);
    }
  };

  return (
    <main style={{ padding: "24px", fontFamily: "monospace" }}>
      <h1>Extraction 审核</h1>
      <p>
        <Link href="/dashboard">返回 Dashboard</Link>
      </p>
      <p>
        默认仅展示每个 raw_post 最新 extraction。auto-review 阈值=70，手动 force re-extract 默认进入待人工审核。
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
        <button type="button" onClick={() => void recomputeStatuses()} disabled={loading || recomputeBusy}>
          {recomputeBusy ? "执行中..." : "刷新状态（按新规则）"}
        </button>
        {status === "pending" && (
          <>
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
