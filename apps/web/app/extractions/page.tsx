"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";

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
  const searchParams = useSearchParams();
  const initialStatus = getInitialStatus(searchParams.get("status"));
  const initialQ = searchParams.get("q") ?? "";
  const initialBadOnly = searchParams.get("bad_only") === "true";
  const [status, setStatus] = useState<ExtractionFilterStatus>(initialStatus);
  const [q, setQ] = useState(initialQ);
  const [badOnly, setBadOnly] = useState(initialBadOnly);
  const [offset, setOffset] = useState(0);
  const [items, setItems] = useState<Extraction[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(searchParams.get("msg"));
  const [clearAlsoDeleteRawPosts, setClearAlsoDeleteRawPosts] = useState(false);
  const [progress, setProgress] = useState<XProgress | null>(null);
  const [backfillBusy, setBackfillBusy] = useState(false);
  const [backfillResult, setBackfillResult] = useState<BackfillAutoReviewResponse | null>(null);
  const [stats, setStats] = useState<ExtractionsStats | null>(null);
  const [refreshWrongBusy, setRefreshWrongBusy] = useState(false);
  const requestSeqRef = useRef(0);

  const canPrev = useMemo(() => offset > 0, [offset]);
  const canNext = useMemo(() => items.length === PAGE_SIZE, [items.length]);

  const load = async () => {
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
  };

  useEffect(() => {
    void load();
  }, [status, offset, q, badOnly]);

  const refreshStats = async () => {
    const res = await fetch("/api/extractions/stats", { cache: "no-store" });
    const body = (await res.json()) as ExtractionsStats | { detail?: string };
    if (!res.ok) {
      throw new Error("detail" in body ? (body.detail ?? "Load stats failed") : "Load stats failed");
    }
    setStats(body as ExtractionsStats);
  };

  const refreshProgress = async () => {
    const res = await fetch("/api/ingest/x/progress", { cache: "no-store" });
    const body = (await res.json()) as XProgress | { detail?: string };
    if (!res.ok) {
      throw new Error("detail" in body ? (body.detail ?? "Load progress failed") : "Load progress failed");
    }
    setProgress(body as XProgress);
  };

  useEffect(() => {
    void refreshProgress();
    void refreshStats();
  }, []);

  const rejectInline = async (extractionId: number) => {
    const reason = window.prompt("Reject reason (optional):", "") ?? "";
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
      <h1>Extractions Review</h1>
      <p>
        默认展示 pending，可切换状态；Approve 进入详情页，Reject 可直接操作。
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
          Show bad only
        </label>
        <button type="button" onClick={() => void load()} disabled={loading}>
          {loading ? "Loading..." : "Refresh"}
        </button>
        <button type="button" onClick={() => void refreshStats()} disabled={loading}>
          Refresh Stats
        </button>
        {status === "pending" && (
          <>
            <button type="button" onClick={() => void backfillAutoReview()} disabled={loading || backfillBusy}>
              {backfillBusy ? "Running..." : "Backfill Auto Review (confidence rule)"}
            </button>
            <label>
              <input
                type="checkbox"
                checked={clearAlsoDeleteRawPosts}
                onChange={(event) => setClearAlsoDeleteRawPosts(event.target.checked)}
              />{" "}
              also delete raw posts
            </label>
            <button type="button" onClick={() => void clearPending()} disabled={loading}>
              Clear Pending (Delete)
            </button>
            <button type="button" onClick={() => void refreshWrongExtractedJson()} disabled={loading || refreshWrongBusy}>
              {refreshWrongBusy ? "Running..." : "Refresh Wrong Extracted JSON"}
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

      {!loading && items.length === 0 && <p>No extractions.</p>}

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
                <Link href={`/extractions/${item.id}`}>Approve</Link>
                <button type="button" onClick={() => void rejectInline(item.id)} disabled={item.status !== "pending"}>
                  Reject
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
          Prev
        </button>
        <button type="button" onClick={() => setOffset((prev) => prev + PAGE_SIZE)} disabled={!canNext || loading}>
          Next
        </button>
        <small>offset={offset}</small>
      </div>
    </main>
  );
}
