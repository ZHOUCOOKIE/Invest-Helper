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

type RetryPendingAllResponse = {
  author_handle: string | null;
  pending_total: number;
  failed_pending_count: number;
  active_pending_count: number;
  skipped_not_followed_count: number;
  submitted_count: number;
  job_id: string | null;
};

type ExtractionStats = {
  raw_posts_count: number;
  post_extractions_count: number;
  duplicate_raw_post_count: number;
};

const PAGE_SIZE = 20;
const statusOptions: ExtractionFilterStatus[] = ["all", "pending", "approved", "rejected", "library"];

function summarizeExtraction(extracted: Record<string, unknown>): string {
  const rawAssetViews = extracted.asset_views;
  if (Array.isArray(rawAssetViews) && rawAssetViews.length > 0) {
    let bestSummary = "";
    let bestConfidence = -1;
    for (const item of rawAssetViews) {
      if (!item || typeof item !== "object") continue;
      const view = item as Record<string, unknown>;
      const summary = typeof view.summary === "string" ? view.summary.trim() : "";
      if (!summary) continue;
      const confidence = typeof view.confidence === "number" ? view.confidence : -1;
      if (confidence >= bestConfidence) {
        bestConfidence = confidence;
        bestSummary = summary;
      }
    }
    if (bestSummary) return bestSummary;
  }

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

  return "（无摘要）";
}

function buildRejectReason(extracted: Record<string, unknown>): string {
  const rawMeta = extracted.meta;
  const meta = rawMeta && typeof rawMeta === "object" ? (rawMeta as Record<string, unknown>) : null;
  const assetViews = extracted.asset_views;
  const hasAssetViews = Array.isArray(assetViews) && assetViews.length > 0;
  const hasview = extracted.hasview;

  if (meta?.auto_rejected === true) {
    const code = String(meta.auto_review_reason ?? "-");
    const threshold = String(meta.auto_review_threshold ?? "-");
    const modelConfidence = String(meta.model_confidence ?? "-");
    if (code === "hasview_zero") return "未识别到可审核资产观点（hasview=0 或 asset_views 为空）";
    if (code === "confidence_below_threshold") return `模型置信度不足（${modelConfidence} < ${threshold}）`;
    return code;
  }
  if (hasview === 0 || !hasAssetViews) {
    return "未识别到可审核资产观点（hasview=0 或 asset_views 为空）";
  }
  return "规则判定拒绝";
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

function statusText(status: Extraction["status"] | ExtractionFilterStatus): string {
  if (status === "pending") return "待处理";
  if (status === "approved") return "已通过";
  if (status === "rejected") return "已拒绝";
  if (status === "library") return "入库";
  return "全部";
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

function pickRawCreatedAtValue(rawJson: Record<string, unknown> | null): string | null {
  if (!rawJson) return null;
  const candidates: Array<Record<string, unknown>> = [rawJson];
  for (const key of ["tweet", "post", "row"]) {
    const nested = rawJson[key];
    if (nested && typeof nested === "object" && !Array.isArray(nested)) {
      candidates.push(nested as Record<string, unknown>);
    }
  }
  for (const item of candidates) {
    for (const key of ["created_at", "createdAt"]) {
      const value = item[key];
      if (typeof value === "string" && value.trim()) {
        return value.trim();
      }
    }
  }
  return null;
}

function stripTimezoneSuffix(value: string | null | undefined): string {
  if (typeof value !== "string") return "-";
  const trimmed = value.trim();
  if (!trimmed) return "-";
  return trimmed.replace(/\s?(?:Z|[+-]\d{2}:\d{2})$/i, "").trim();
}

function displayPostedTime(rawPost: RawPost): string {
  return stripTimezoneSuffix(pickRawCreatedAtValue(rawPost.raw_json) ?? rawPost.posted_at);
}

function toSortMs(value: string | null | undefined): number {
  if (typeof value !== "string" || !value.trim()) return 0;
  const ms = Date.parse(value);
  return Number.isNaN(ms) ? 0 : ms;
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
  const [stats, setStats] = useState<ExtractionStats | null>(null);
  const [recomputeBusy, setRecomputeBusy] = useState(false);
  const [retryPendingBusy, setRetryPendingBusy] = useState(false);
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
        throw new Error("detail" in body ? (body.detail ?? "加载失败") : "加载失败");
      }
      if (requestSeq !== requestSeqRef.current) return;
      const dedupedMap = new Map<number, Extraction>();
      for (const item of body as Extraction[]) {
        dedupedMap.set(item.id, item);
      }
      const deduped = Array.from(dedupedMap.values()).sort((a, b) => {
        const aMs = toSortMs(a.raw_post?.posted_at) || toSortMs(a.created_at);
        const bMs = toSortMs(b.raw_post?.posted_at) || toSortMs(b.created_at);
        if (aMs !== bMs) return bMs - aMs;
        return b.id - a.id;
      });
      setItems(deduped);
    } catch (err) {
      if (requestSeq !== requestSeqRef.current) return;
      setError(err instanceof Error ? err.message : "未知错误");
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
      throw new Error("detail" in body ? (body.detail ?? "加载进度失败") : "加载进度失败");
    }
    setProgress(body as XProgress);
  }, []);

  const refreshStats = useCallback(async () => {
    const res = await fetch("/api/extractions/stats", { cache: "no-store" });
    const body = (await res.json()) as ExtractionStats | { detail?: string };
    if (!res.ok) {
      throw new Error("detail" in body ? (body.detail ?? "加载统计失败") : "加载统计失败");
    }
    setStats(body as ExtractionStats);
  }, []);

  useEffect(() => {
    const loadAux = async () => {
      try {
        await Promise.all([refreshProgress(), refreshStats()]);
      } catch (err) {
        setError(err instanceof Error ? err.message : "加载进度失败");
      }
    };
    void loadAux();
  }, [refreshProgress, refreshStats]);

  const clearPending = async () => {
    const confirmText = window.prompt(
      "高风险操作：将硬删除 pending/failed 的抽取记录。输入 YES 继续。",
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
        throw new Error("detail" in body ? (body.detail ?? "清空待处理失败") : "清空待处理失败");
      }
      const done = body as ClearPendingResponse;
      setMessage(
        `已删除抽取=${done.deleted_extractions_count}，已删原始贴文=${done.deleted_raw_posts_count}。`,
      );
      await Promise.all([load(), refreshProgress(), refreshStats()]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "清空待处理失败");
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
        throw new Error("detail" in body ? (body.detail ?? "重算状态失败") : "重算状态失败");
      }
      const done = body as RecomputeStatusesResponse;
      setMessage(
        `重算完成：扫描=${done.scanned}，更新=${done.updated}，待处理=${done.pending_count}，通过=${done.approved_count}，拒绝=${done.rejected_count}。`,
      );
      await Promise.all([load(), refreshProgress(), refreshStats()]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "重算状态失败");
    } finally {
      setRecomputeBusy(false);
    }
  };

  const retryAllPending = async () => {
    setRetryPendingBusy(true);
    setError(null);
    setMessage(null);
    try {
      const res = await fetch("/api/ingest/x/retry-pending-all", { method: "POST" });
      const body = (await res.json()) as RetryPendingAllResponse | { detail?: string };
      if (!res.ok) {
        throw new Error("detail" in body ? (body.detail ?? "提交重传任务失败") : "提交重传任务失败");
      }
      const done = body as RetryPendingAllResponse;
      if (!done.job_id || done.submitted_count === 0) {
        setMessage("当前待处理列表没有可重传的贴文。");
      } else {
        setMessage(
          `已提交重传任务 job_id=${done.job_id}，待处理总计=${done.pending_total}，失败待重传=${done.failed_pending_count}，进行中=${done.active_pending_count}。`,
        );
      }
      await Promise.all([load(), refreshProgress(), refreshStats()]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "提交重传任务失败");
    } finally {
      setRetryPendingBusy(false);
    }
  };

  return (
    <main style={{ padding: "24px", fontFamily: "monospace" }}>
      <h1>抽取审核</h1>
      <p>
        默认仅展示每个 raw_post 最新 extraction。auto-review 阈值=80，手动 force re-extract 默认进入待人工审核。
      </p>

      <div style={{ display: "flex", gap: "8px", alignItems: "center", marginBottom: "12px", flexWrap: "wrap" }}>
        <label>
          状态
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
                {option === "all"
                  ? "全部"
                  : option === "pending"
                    ? "待处理"
                    : option === "approved"
                      ? "已通过"
                      : option === "rejected"
                        ? "已拒绝"
                        : "入库"}
              </option>
            ))}
          </select>
        </label>
        <label>
          关键词
          <input
            value={q}
            onChange={(event) => {
              setQ(event.target.value);
              setOffset(0);
            }}
            placeholder="请输入关键词"
            style={{ marginLeft: "8px" }}
          />
        </label>
        <button type="button" onClick={() => void recomputeStatuses()} disabled={loading || recomputeBusy}>
          {recomputeBusy ? "执行中..." : "刷新状态（按新规则）"}
        </button>
        {status === "pending" && (
          <>
            <button type="button" onClick={() => void retryAllPending()} disabled={loading || retryPendingBusy}>
              {retryPendingBusy ? "提交中..." : "重新全部上传 AI 解析"}
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
          </>
        )}
      </div>

      {message && <p style={{ color: "green" }}>{message}</p>}
      {error && <p style={{ color: "crimson" }}>{error}</p>}
      {progress && (
        <p style={{ color: "#555" }}>
          进度[{progress.scope}] AI处理成功={Math.max(0, progress.total_raw_posts - progress.pending_count)}, 已通过=
          {progress.extracted_success_count}, 待处理={progress.pending_count}, 已拒绝={progress.failed_count}, 已入库=
          {progress.no_extraction_count}
        </p>
      )}
      {stats && (
        <p style={{ color: "#555" }}>
          当前库：原始贴文={stats.raw_posts_count}，AI解析结果={stats.post_extractions_count}，存在重复版本的原始贴文=
          {stats.duplicate_raw_post_count}
        </p>
      )}

      {!loading && items.length === 0 && <p>暂无抽取记录。</p>}

      <div style={{ display: "grid", gap: "10px" }}>
        {items.map((item, idx) => {
          const rejectedReason = item.status === "rejected" ? buildRejectReason(item.extracted_json) : null;
          const summaryText = summarizeExtraction(item.extracted_json);
          const serialNo = offset + idx + 1;
          const publicId = buildPublicId(item);
          return (
            <article key={item.id} style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "10px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", gap: "8px", marginBottom: "6px" }}>
                <strong>第 {serialNo} 条 [{statusText(item.status)}] public_id={publicId}</strong>
                <small>创建时间: {stripTimezoneSuffix(item.created_at)}</small>
              </div>
              <div>
                <small>抽取 ID={item.id}</small>
              </div>
              <div>
                {item.raw_post.platform} / @{item.raw_post.author_handle}
              </div>
              <div>
                <a href={item.raw_post.url} target="_blank" rel="noreferrer">
                  {item.raw_post.url}
                </a>
              </div>
              <div>发布时间: {displayPostedTime(item.raw_post)}</div>
              {item.status === "approved" && <div style={{ marginTop: "6px" }}>摘要: {summaryText}</div>}
              {item.status === "rejected" && rejectedReason && (
                <div style={{ marginTop: "4px", color: "#8a5800" }}>
                  异常已拒绝：{rejectedReason}
                </div>
              )}
              <div style={{ display: "flex", gap: "8px", marginTop: "8px" }}>
                <Link href={`/extractions/${item.id}`}>查看详情</Link>
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
        <small>偏移量={offset}</small>
      </div>
    </main>
  );
}
