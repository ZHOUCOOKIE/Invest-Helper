"use client";

import Link from "next/link";
import type { DragEvent, FormEvent } from "react";
import { useEffect, useMemo, useRef, useState } from "react";

type ManualIngestResponse = {
  raw_post: { id: number };
  extraction: {
    id: number;
    raw_post_id: number;
    extractor_name: string;
    last_error: string | null;
  };
  extraction_id: number;
};

type ExtractorStatus = {
  mode: "auto" | "dummy" | "openai" | string;
  has_api_key: boolean;
  default_model: string;
  base_url: string;
  call_budget_remaining: number | null;
  max_output_tokens: number;
};

type Kol = {
  id: number;
  platform: string;
  handle: string;
  display_name: string | null;
  enabled: boolean;
};

type AssetItem = {
  id: number;
  symbol: string;
  name: string | null;
  market: string | null;
};

type XImportItem = {
  external_id: string;
  author_handle: string;
  resolved_author_handle?: string | null;
  url: string;
  posted_at: string;
  content_text: string;
  kol_id?: number;
  raw_json?: Record<string, unknown> | null;
};

type XConvertError = {
  row_index: number;
  external_id?: string | null;
  url?: string | null;
  reason: string;
};

type SkippedNotFollowed = {
  row_index: number;
  author_handle?: string | null;
  external_id?: string | null;
  reason: string;
};

type HandleSummary = {
  author_handle: string;
  count: number;
  earliest_posted_at: string | null;
  latest_posted_at: string | null;
  will_create_kol: boolean;
};

type XConvertResult = {
  converted_rows: number;
  converted_ok: number;
  converted_failed: number;
  errors: XConvertError[];
  items: XImportItem[];
  handles_summary: HandleSummary[];
  resolved_author_handle: string | null;
  resolved_kol_id: number | null;
  kol_created: boolean;
  skipped_not_followed_count: number;
  skipped_not_followed_samples: SkippedNotFollowed[];
};

type ImportByHandleStats = {
  received: number;
  inserted: number;
  dedup: number;
  warnings: number;
  raw_post_ids: number[];
  extract_success: number;
  extract_failed: number;
  skipped_already_extracted: number;
};

type CreatedKol = {
  id: number;
  handle: string;
  name: string | null;
};

type XImportStats = {
  received_count: number;
  inserted_raw_posts_count: number;
  inserted_raw_post_ids: number[];
  dedup_existing_raw_post_ids: number[];
  dedup_skipped_count: number;
  extract_success_count: number;
  extract_failed_count: number;
  skipped_already_extracted_count: number;
  warnings_count: number;
  warnings: string[];
  imported_by_handle: Record<string, ImportByHandleStats>;
  created_kols: CreatedKol[];
  resolved_author_handle: string | null;
  resolved_kol_id: number | null;
  kol_created: boolean;
  skipped_not_followed_count: number;
  skipped_not_followed_samples: SkippedNotFollowed[];
};

type FollowingImportError = {
  row_index: number;
  reason: string;
  raw_snippet: string;
};

type FollowingImportKol = {
  id: number;
  handle: string;
};

type FollowingImportStats = {
  received_rows: number;
  following_true_rows: number;
  created_kols_count: number;
  updated_kols_count: number;
  skipped_count: number;
  created_kols: FollowingImportKol[];
  updated_kols: FollowingImportKol[];
  errors: FollowingImportError[];
};

type AdminClearPendingResponse = {
  deleted_extractions_count: number;
  deleted_raw_posts_count: number;
  scoped_author_handle: string | null;
};

type AdminHardDeleteResponse = {
  operation: string;
  target: string;
  derived_only: boolean;
  enable_cascade: boolean;
  also_delete_raw_posts: boolean;
  counts: Record<string, number>;
};

type ExtractBatchStats = {
  requested_count: number;
  success_count: number;
  skipped_count: number;
  skipped_already_extracted_count: number;
  skipped_already_pending_count: number;
  skipped_already_success_count: number;
  skipped_already_has_result_count: number;
  skipped_already_rejected_count: number;
  skipped_already_approved_count: number;
  skipped_due_to_import_limit_count: number;
  skipped_not_followed_count: number;
  failed_count: number;
  auto_rejected_count: number;
  resumed_requested_count: number;
  resumed_success: number;
  resumed_failed: number;
  resumed_skipped: number;
};

type ExtractJobCreateResponse = {
  job_id: string;
};

type ExtractJob = ExtractBatchStats & {
  job_id: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled" | "timeout" | "done";
  is_terminal: boolean;
  mode: "pending_only" | "pending_or_failed" | "force";
  batch_size: number;
  batch_sleep_ms: number;
  ai_call_limit_total: number | null;
  ai_call_used: number;
  last_error_summary: string | null;
  created_at: string;
  last_updated_at: string;
  started_at: string | null;
  finished_at: string | null;
};

type ApiErrorBody = {
  request_id?: string;
  error_code?: string;
  message?: string;
  detail?: unknown;
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

type FormState = {
  platform: "x" | "reddit" | "other";
  author_handle: string;
  url: string;
  content_text: string;
  posted_at: string;
};

const STORAGE_START_DATE = "x_import_start_date";
const STORAGE_END_DATE = "x_import_end_date";
const DEFAULT_DIRECT_API_BASE_URL = "http://localhost:8000";
const DIRECT_CONVERT_PATH = "/ingest/x/convert";
const DIRECT_IMPORT_PATH = "/ingest/x/import";
const DIRECT_EXTRACT_JOBS_PATH = "/extract-jobs";

function getDirectApiBaseUrl(): string {
  const raw = process.env.NEXT_PUBLIC_API_BASE_URL?.trim();
  const base = raw && raw.length > 0 ? raw : DEFAULT_DIRECT_API_BASE_URL;
  return base.replace(/\/+$/, "");
}

function buildDirectApiUrl(path: string, params: URLSearchParams): string {
  const query = params.toString();
  const suffix = query ? `?${query}` : "";
  return `${getDirectApiBaseUrl()}${path}${suffix}`;
}

function buildDirectConvertUrl(params: URLSearchParams): string {
  return buildDirectApiUrl(DIRECT_CONVERT_PATH, params);
}

function formatConvertNetworkError(err: unknown): string {
  const baseUrl = getDirectApiBaseUrl();
  const rawMessage = err instanceof Error ? err.message : String(err ?? "unknown error");
  return [
    `Convert failed: 无法连接后端 ${baseUrl}`,
    "请确认后端服务已启动，或检查是否被代理层的大文件限制中断（例如 10MB 限制导致 ECONNRESET/socket hang up）。",
    `原始错误: ${rawMessage}`,
  ].join(" ");
}

function formatImportNetworkError(err: unknown): string {
  const baseUrl = getDirectApiBaseUrl();
  const rawMessage = err instanceof Error ? err.message : String(err ?? "unknown error");
  return [
    `Import failed: 无法连接后端 ${baseUrl}`,
    "请检查后端是否启动、CORS 是否允许当前前端域名，或网络是否中断。",
    `原始错误: ${rawMessage}`,
  ].join(" ");
}

function formatExtractJobNetworkError(stage: "create" | "poll", err: unknown): string {
  const baseUrl = getDirectApiBaseUrl();
  const rawMessage = err instanceof Error ? err.message : String(err ?? "unknown error");
  const action = stage === "create" ? "创建抽取任务" : "轮询抽取任务";
  return [
    `Extract job failed: ${action}时无法连接后端 ${baseUrl}`,
    "请检查后端是否启动、CORS 是否允许当前前端域名，或网络是否中断。",
    `原始错误: ${rawMessage}`,
  ].join(" ");
}

function isStandardImportJson(value: unknown): value is XImportItem[] {
  if (!Array.isArray(value)) return false;
  return value.every((item) => {
    if (typeof item !== "object" || item === null) return false;
    const row = item as Record<string, unknown>;
    return (
      typeof row.external_id === "string" &&
      typeof row.author_handle === "string" &&
      typeof row.url === "string" &&
      typeof row.posted_at === "string" &&
      typeof row.content_text === "string"
    );
  });
}

function toDateKeyFromIso(value: string): string | null {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return null;
  return parsed.toISOString().slice(0, 10);
}

export default function IngestPage() {
  const [form, setForm] = useState<FormState>({
    platform: "x",
    author_handle: "",
    url: "",
    content_text: "",
    posted_at: "",
  });
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ManualIngestResponse | null>(null);
  const [extractorStatus, setExtractorStatus] = useState<ExtractorStatus | null>(null);
  const [extractBatchSize, setExtractBatchSize] = useState(20);
  const [statusError, setStatusError] = useState<string | null>(null);

  const [kols, setKols] = useState<Kol[]>([]);
  const [assets, setAssets] = useState<AssetItem[]>([]);
  const [assetsError, setAssetsError] = useState<string | null>(null);
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [onlyFollowedKols, setOnlyFollowedKols] = useState(true);
  const [autoExtract, setAutoExtract] = useState(true);
  const [autoGenerateDigest, setAutoGenerateDigest] = useState(false);
  const [digestDate, setDigestDate] = useState(new Date().toISOString().slice(0, 10));
  const [dragActive, setDragActive] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [workflowBusy, setWorkflowBusy] = useState(false);
  const [workflowStep, setWorkflowStep] = useState<string | null>(null);
  const [workflowError, setWorkflowError] = useState<string | null>(null);
  const [workflowRequestId, setWorkflowRequestId] = useState<string | null>(null);
  const [convertedCount, setConvertedCount] = useState<number | null>(null);
  const [convertResult, setConvertResult] = useState<XConvertResult | null>(null);
  const [importStats, setImportStats] = useState<XImportStats | null>(null);
  const [extractStats, setExtractStats] = useState<ExtractBatchStats | null>(null);
  const [selectedHandles, setSelectedHandles] = useState<string[]>([]);
  const [extractByHandle, setExtractByHandle] = useState<Record<string, ExtractBatchStats>>({});
  const [clearAlsoDeleteRawPosts, setClearAlsoDeleteRawPosts] = useState(false);
  const [progress, setProgress] = useState<XProgress | null>(null);
  const [generatedDigestDate, setGeneratedDigestDate] = useState<string | null>(null);
  const [cleanupKolId, setCleanupKolId] = useState("");
  const [cleanupKolCascade, setCleanupKolCascade] = useState(false);
  const [cleanupKolDeleteRawPosts, setCleanupKolDeleteRawPosts] = useState(false);
  const [cleanupAssetId, setCleanupAssetId] = useState("");
  const [cleanupAssetCascade, setCleanupAssetCascade] = useState(false);
  const [cleanupDigestDate, setCleanupDigestDate] = useState(new Date().toISOString().slice(0, 10));
  const [cleanupDigestProfileId, setCleanupDigestProfileId] = useState("1");
  const [followingFile, setFollowingFile] = useState<File | null>(null);
  const [followingImportBusy, setFollowingImportBusy] = useState(false);
  const [followingImportError, setFollowingImportError] = useState<string | null>(null);
  const [followingImportStats, setFollowingImportStats] = useState<FollowingImportStats | null>(null);
  const extractPollControllerRef = useRef<AbortController | null>(null);
  const extractPollRunIdRef = useRef(0);
  const cancelExtractPolling = (reason?: string) => {
    extractPollRunIdRef.current += 1;
    extractPollControllerRef.current?.abort();
    extractPollControllerRef.current = null;
    if (reason) setWorkflowStep(reason);
  };

  useEffect(() => {
    const cachedStart = localStorage.getItem(STORAGE_START_DATE);
    const cachedEnd = localStorage.getItem(STORAGE_END_DATE);
    if (cachedStart) setStartDate(cachedStart);
    if (cachedEnd) setEndDate(cachedEnd);
  }, []);

  useEffect(() => {
    if (startDate) localStorage.setItem(STORAGE_START_DATE, startDate);
  }, [startDate]);

  useEffect(() => {
    return () => {
      cancelExtractPolling();
    };
  }, []);

  useEffect(() => {
    if (endDate) localStorage.setItem(STORAGE_END_DATE, endDate);
  }, [endDate]);

  useEffect(() => {
    const loadStatus = async () => {
      try {
        const extractorRes = await fetch("/api/extractor-status", { cache: "no-store" });
        const extractorParsed = await parseApiResponse<ExtractorStatus>(extractorRes);
        if (!extractorRes.ok || !extractorParsed.data) {
          throw new Error(formatApiError("Load extractor status failed", extractorParsed.data, extractorParsed.textBody, extractorParsed.requestId));
        }
        setExtractorStatus(extractorParsed.data as ExtractorStatus);
      } catch (err) {
        setStatusError(err instanceof Error ? err.message : "Load extractor status failed");
      }
    };
    const loadKols = async () => {
      try {
        const res = await fetch("/api/kols?enabled=true", { cache: "no-store" });
        const parsed = await parseApiResponse<Kol[]>(res);
        if (!res.ok || !Array.isArray(parsed.data)) {
          throw new Error(formatApiError("Load kols failed", parsed.data, parsed.textBody, parsed.requestId));
        }
        const xKols = (parsed.data as Kol[]).filter((item) => item.platform === "x");
        setKols(xKols);
      } catch (err) {
        setWorkflowError(err instanceof Error ? err.message : "Load kols failed");
      }
    };
    const loadAssets = async () => {
      try {
        const res = await fetch("/api/assets", { cache: "no-store" });
        const parsed = await parseApiResponse<AssetItem[]>(res);
        if (!res.ok || !Array.isArray(parsed.data)) {
          throw new Error(formatApiError("Load assets failed", parsed.data, parsed.textBody, parsed.requestId));
        }
        const sorted = [...(parsed.data as AssetItem[])].sort((a, b) => a.symbol.localeCompare(b.symbol));
        setAssets(sorted);
      } catch (err) {
        setAssetsError(err instanceof Error ? err.message : "Load assets failed");
      }
    };
    void loadStatus();
    void loadKols();
    void loadAssets();
  }, []);

  const formatApiError = (
    fallbackMessage: string,
    body: unknown,
    textBody: string,
    requestId: string | null,
    statusCode?: number,
    requestPath?: string,
  ): string => {
    let message = fallbackMessage;
    if (body && typeof body === "object") {
      const obj = body as Record<string, unknown>;
      if (typeof obj.message === "string" && obj.message) message = obj.message;
      else if (typeof obj.detail === "string" && obj.detail) message = obj.detail;
    } else if (textBody) {
      const statusPart = typeof statusCode === "number" ? `status=${statusCode}` : "status=unknown";
      const pathPart = requestPath ? `path=${requestPath}` : "path=unknown";
      message = `非 JSON 错误文本 (${statusPart}, ${pathPart}): ${textBody.slice(0, 300)}`;
    }
    return requestId ? `${message} (request_id=${requestId})` : message;
  };

  const parseApiResponse = async <T,>(
    res: Response,
  ): Promise<{
    data: T | ApiErrorBody | null;
    textBody: string;
    requestId: string | null;
    statusCode: number;
    requestPath: string;
  }> => {
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
  };

  const progressHandle = useMemo(() => {
    if (selectedHandles.length === 1) return selectedHandles[0];
    return "";
  }, [selectedHandles]);

  const refreshProgress = async () => {
    const query = progressHandle ? `?author_handle=${encodeURIComponent(progressHandle)}` : "";
    const res = await fetch(`/api/ingest/x/progress${query}`, { cache: "no-store" });
    const parsed = await parseApiResponse<XProgress>(res);
    if (!res.ok || !parsed.data) {
      throw new Error(formatApiError("Load progress failed", parsed.data, parsed.textBody, parsed.requestId));
    }
    setProgress(parsed.data as XProgress);
  };

  const onManualSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    setResult(null);
    try {
      const payload: Record<string, string | null> = {
        platform: form.platform,
        author_handle: form.author_handle.trim(),
        url: form.url.trim(),
        content_text: form.content_text.trim(),
      };
      if (form.posted_at) payload.posted_at = new Date(form.posted_at).toISOString();

      const res = await fetch("/api/ingest/manual", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const parsed = await parseApiResponse<ManualIngestResponse>(res);
      if (!res.ok || !parsed.data) throw new Error(formatApiError("Submit failed", parsed.data, parsed.textBody, parsed.requestId));
      setResult(parsed.data as ManualIngestResponse);
      setForm((prev) => ({ ...prev, url: "", content_text: "" }));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setSubmitting(false);
    }
  };

  const applyOverridesForStandardRows = (rows: XImportItem[]): XImportItem[] => {
    return rows.filter((item) => {
      const key = toDateKeyFromIso(item.posted_at);
      if (!key) return false;
      if (startDate && key < startDate) return false;
      if (endDate && key > endDate) return false;
      return true;
    });
  };

  const convertViaApi = async (targetFile: File): Promise<XConvertResult> => {
    const params = new URLSearchParams();
    params.set("filename", targetFile.name);
    if (startDate) params.set("start_date", startDate);
    if (endDate) params.set("end_date", endDate);
    params.set("include_raw_json", "true");
    params.set("only_followed", onlyFollowedKols ? "true" : "false");
    params.set("allow_unknown_handles", onlyFollowedKols ? "false" : "true");
    let res: Response;
    try {
      res = await fetch(buildDirectConvertUrl(params), {
        method: "POST",
        headers: { "Content-Type": targetFile.type || "application/octet-stream" },
        body: targetFile,
      });
    } catch (err) {
      throw new Error(formatConvertNetworkError(err));
    }
    const parsed = await parseApiResponse<XConvertResult>(res);
    if (!res.ok || !parsed.data || !("items" in (parsed.data as Record<string, unknown>))) {
      setWorkflowRequestId(parsed.requestId);
      throw new Error(formatApiError("Convert failed", parsed.data, parsed.textBody, parsed.requestId));
    }
    return parsed.data as XConvertResult;
  };

  const importRows = async (rows: XImportItem[]): Promise<XImportStats> => {
    const params = new URLSearchParams();
    params.set("only_followed", onlyFollowedKols ? "true" : "false");
    params.set("allow_unknown_handles", onlyFollowedKols ? "false" : "true");
    let res: Response;
    try {
      res = await fetch(buildDirectApiUrl(DIRECT_IMPORT_PATH, params), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(rows),
      });
    } catch (err) {
      throw new Error(formatImportNetworkError(err));
    }
    const parsed = await parseApiResponse<XImportStats>(res);
    setWorkflowRequestId(parsed.requestId);
    if (!res.ok || !parsed.data) {
      throw new Error(formatApiError("Import failed", parsed.data, parsed.textBody, parsed.requestId));
    }
    return parsed.data as XImportStats;
  };

  const runExtractJob = async (
    ids: number[],
    batchSize: number,
    mode: "pending_only" | "pending_or_failed" | "force",
    aiCallLimitTotal?: number,
  ): Promise<ExtractBatchStats> => {
    const stableIds = Array.from(new Set(ids)).sort((a, b) => a - b);
    const normalizedAiLimit =
      typeof aiCallLimitTotal === "number" && Number.isFinite(aiCallLimitTotal)
        ? Math.max(0, Math.floor(aiCallLimitTotal))
        : null;
    const idempotencyKey = `ingest:${mode}:${Math.max(1, batchSize)}:${normalizedAiLimit ?? "none"}:${stableIds.join(",")}`.slice(0, 256);
    cancelExtractPolling();
    const pollController = new AbortController();
    extractPollControllerRef.current = pollController;
    const runId = extractPollRunIdRef.current + 1;
    extractPollRunIdRef.current = runId;
    let createRes: Response;
    try {
      createRes = await fetch(buildDirectApiUrl(DIRECT_EXTRACT_JOBS_PATH, new URLSearchParams()), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: pollController.signal,
        body: JSON.stringify({
          raw_post_ids: stableIds,
          mode,
          batch_size: Math.max(1, batchSize),
          ai_call_limit_total: normalizedAiLimit,
          idempotency_key: idempotencyKey,
        }),
      });
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        throw new Error("Extract job polling cancelled");
      }
      throw new Error(formatExtractJobNetworkError("create", err));
    }
    const createParsed = await parseApiResponse<ExtractJobCreateResponse>(createRes);
    setWorkflowRequestId(createParsed.requestId);
    if (!createRes.ok || !createParsed.data || !("job_id" in (createParsed.data as Record<string, unknown>))) {
      throw new Error(
        formatApiError(
          "Create extract job failed",
          createParsed.data,
          createParsed.textBody,
          createParsed.requestId,
          createParsed.statusCode,
          createParsed.requestPath,
        ),
      );
    }
    const jobId = (createParsed.data as ExtractJobCreateResponse).job_id;

    try {
      for (let attempt = 0; attempt < 1800; attempt += 1) {
        if (extractPollRunIdRef.current !== runId) {
          throw new Error("Extract job polling superseded by a newer run");
        }
        let res: Response;
        try {
          res = await fetch(buildDirectApiUrl(`${DIRECT_EXTRACT_JOBS_PATH}/${encodeURIComponent(jobId)}`, new URLSearchParams()), {
            cache: "no-store",
            signal: pollController.signal,
          });
        } catch (err) {
          if (err instanceof DOMException && err.name === "AbortError") {
            throw new Error("Extract job polling cancelled");
          }
          throw new Error(formatExtractJobNetworkError("poll", err));
        }
        const parsed = await parseApiResponse<ExtractJob>(res);
        if (!res.ok || !parsed.data) {
          setWorkflowRequestId(parsed.requestId);
          throw new Error(
            formatApiError("Load extract job failed", parsed.data, parsed.textBody, parsed.requestId, parsed.statusCode, parsed.requestPath),
          );
        }
        const job = parsed.data as ExtractJob;
        setWorkflowRequestId(parsed.requestId);
        setWorkflowStep(
          `Extract job ${job.status}: success=${job.success_count}, failed=${job.failed_count}, skipped=${job.skipped_count}, requested=${job.requested_count}, ai_used=${job.ai_call_used}${job.ai_call_limit_total !== null ? `/${job.ai_call_limit_total}` : ""}`,
        );
        if (job.is_terminal) {
          if (job.status === "failed" || job.status === "cancelled" || job.status === "timeout") {
            const summary = job.last_error_summary ? `: ${job.last_error_summary}` : "";
            throw new Error(`Extract job failed${summary}${parsed.requestId ? ` (request_id=${parsed.requestId})` : ""}`);
          }
          return job;
        }
        await new Promise((resolve, reject) => {
          const timer = window.setTimeout(() => resolve(undefined), 1200);
          pollController.signal.addEventListener(
            "abort",
            () => {
              window.clearTimeout(timer);
              reject(new DOMException("Aborted", "AbortError"));
            },
            { once: true },
          );
        });
      }
      throw new Error("Extract job polling timeout");
    } finally {
      if (extractPollControllerRef.current === pollController) {
        extractPollControllerRef.current = null;
      }
    }
  };

  const generateDigest = async (dateStr: string) => {
    setWorkflowStep(`Generating digest for ${dateStr}...`);
    const res = await fetch(`/api/digests/generate?date=${dateStr}&days=7&profile_id=1`, { method: "POST" });
    const body = (await res.json()) as { detail?: string };
    if (!res.ok) throw new Error(body.detail ?? "Generate digest failed");
    setGeneratedDigestDate(dateStr);
  };

  const onSelectFile = (targetFile: File | null) => {
    setFile(targetFile);
    setWorkflowError(null);
    setImportStats(null);
    setExtractStats(null);
    setExtractByHandle({});
    setConvertedCount(null);
    setConvertResult(null);
    setSelectedHandles([]);
  };

  const syncSelectedHandles = (handles: string[]) => {
    if (handles.length === 0) return;
    setSelectedHandles((prev) => {
      if (prev.length === 0) return handles;
      const kept = prev.filter((item) => handles.includes(item));
      return kept.length > 0 ? kept : handles;
    });
  };

  const onDropFile = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragActive(false);
    const dropped = event.dataTransfer.files?.[0] ?? null;
    onSelectFile(dropped);
  };

  const clearPending = async () => {
    const confirmText = window.prompt(
      "Dangerous operation: this will hard delete pending/failed extractions. Type YES to continue.",
      "",
    );
    if (confirmText !== "YES") return;
    cancelExtractPolling("Extract polling cancelled before cleanup.");
    setWorkflowError(null);
    try {
      const params = new URLSearchParams({
        confirm: "YES",
        also_delete_raw_posts: clearAlsoDeleteRawPosts ? "true" : "false",
        enable_cascade: clearAlsoDeleteRawPosts ? "true" : "false",
      });
      if (progressHandle) params.set("author_handle", progressHandle);
      const res = await fetch(`/api/admin/extractions/pending?${params.toString()}`, { method: "DELETE" });
      const body = (await res.json()) as AdminClearPendingResponse | { detail?: string };
      if (!res.ok) {
        throw new Error("detail" in body ? (body.detail ?? "Clear pending failed") : "Clear pending failed");
      }
      await refreshProgress();
      setWorkflowStep(
        `Deleted pending/failed extractions=${(body as AdminClearPendingResponse).deleted_extractions_count}, raw_posts=${(body as AdminClearPendingResponse).deleted_raw_posts_count}.`,
      );
    } catch (err) {
      setWorkflowError(err instanceof Error ? err.message : "Clear pending failed");
    }
  };

  const askYes = (message: string): boolean => {
    const confirmText = window.prompt(message, "");
    return confirmText === "YES";
  };

  const refreshCleanupScope = async () => {
    await refreshProgress();
    const [kolsRes, assetsRes] = await Promise.all([
      fetch("/api/kols?enabled=true", { cache: "no-store" }),
      fetch("/api/assets", { cache: "no-store" }),
    ]);
    if (kolsRes.ok) {
      const body = (await kolsRes.json()) as Kol[];
      const xKols = body.filter((item) => item.platform === "x");
      setKols(xKols);
    }
    if (assetsRes.ok) {
      const body = (await assetsRes.json()) as AssetItem[];
      setAssets([...body].sort((a, b) => a.symbol.localeCompare(b.symbol)));
    }
  };

  const deleteKolCleanup = async () => {
    const kolId = Number(cleanupKolId);
    if (!Number.isFinite(kolId) || kolId <= 0) {
      setWorkflowError("请输入合法的 KOL ID");
      return;
    }
    if (!askYes("Dangerous operation: delete KOL derived data. Type YES to continue.")) return;
    if ((cleanupKolCascade || cleanupKolDeleteRawPosts) && !askYes("Cascade/raw delete requested. Type YES again to continue.")) {
      return;
    }
    setWorkflowError(null);
    try {
      const params = new URLSearchParams({
        confirm: "YES",
        enable_cascade: cleanupKolCascade ? "true" : "false",
        also_delete_raw_posts: cleanupKolDeleteRawPosts ? "true" : "false",
      });
      const res = await fetch(`/api/admin/kols/${kolId}?${params.toString()}`, { method: "DELETE" });
      const body = (await res.json()) as AdminHardDeleteResponse | { detail?: string };
      if (!res.ok) {
        throw new Error("detail" in body ? (body.detail ?? "Delete KOL failed") : "Delete KOL failed");
      }
      const done = body as AdminHardDeleteResponse;
      await refreshCleanupScope();
      setWorkflowStep(`KOL cleanup done: ${JSON.stringify(done.counts)}`);
    } catch (err) {
      setWorkflowError(err instanceof Error ? err.message : "Delete KOL failed");
    }
  };

  const deleteAssetCleanup = async () => {
    const assetId = Number(cleanupAssetId);
    if (!Number.isFinite(assetId) || assetId <= 0) {
      setWorkflowError("请输入合法的 Asset ID");
      return;
    }
    if (!askYes("Dangerous operation: delete asset derived data. Type YES to continue.")) return;
    if (cleanupAssetCascade && !askYes("Cascade asset base deletion requested. Type YES again to continue.")) {
      return;
    }
    setWorkflowError(null);
    try {
      const params = new URLSearchParams({
        confirm: "YES",
        enable_cascade: cleanupAssetCascade ? "true" : "false",
      });
      const res = await fetch(`/api/admin/assets/${assetId}?${params.toString()}`, { method: "DELETE" });
      const body = (await res.json()) as AdminHardDeleteResponse | { detail?: string };
      if (!res.ok) {
        throw new Error("detail" in body ? (body.detail ?? "Delete asset failed") : "Delete asset failed");
      }
      const done = body as AdminHardDeleteResponse;
      await refreshCleanupScope();
      setWorkflowStep(`Asset cleanup done: ${JSON.stringify(done.counts)}`);
    } catch (err) {
      setWorkflowError(err instanceof Error ? err.message : "Delete asset failed");
    }
  };

  const deleteDigestByDateCleanup = async () => {
    const profileId = Number(cleanupDigestProfileId);
    if (!cleanupDigestDate) {
      setWorkflowError("请选择 digest_date");
      return;
    }
    if (!Number.isFinite(profileId) || profileId <= 0) {
      setWorkflowError("请输入合法 profile_id");
      return;
    }
    if (!askYes("Dangerous operation: delete digests by date/profile. Type YES to continue.")) return;
    setWorkflowError(null);
    try {
      const params = new URLSearchParams({
        confirm: "YES",
        digest_date: cleanupDigestDate,
        profile_id: String(profileId),
      });
      const res = await fetch(`/api/admin/digests?${params.toString()}`, { method: "DELETE" });
      const body = (await res.json()) as AdminHardDeleteResponse | { detail?: string };
      if (!res.ok) {
        throw new Error("detail" in body ? (body.detail ?? "Delete digest failed") : "Delete digest failed");
      }
      const done = body as AdminHardDeleteResponse;
      await refreshCleanupScope();
      setGeneratedDigestDate(cleanupDigestDate);
      setWorkflowStep(`Digest cleanup done: ${JSON.stringify(done.counts)}`);
    } catch (err) {
      setWorkflowError(err instanceof Error ? err.message : "Delete digest failed");
    }
  };

  const handleImportWorkflow = async () => {
    if (workflowBusy) return;
    if (!file) {
      setWorkflowError("Please choose a file first.");
      return;
    }
    setWorkflowBusy(true);
    setWorkflowError(null);
    setWorkflowRequestId(null);
    setWorkflowStep("Reading file...");
    setConvertResult(null);
    setExtractByHandle({});
    setGeneratedDigestDate(null);
    try {
      let rows: XImportItem[] = [];
      let detectedHandles: string[] = [];
      let convertedRowsTotal = 0;
      const maybeJson = file.name.toLowerCase().endsWith(".json");
      if (maybeJson) {
        const text = await file.text();
        try {
          const parsed = JSON.parse(text) as unknown;
          if (isStandardImportJson(parsed)) {
            setWorkflowStep("Detected standard x_import.json, importing directly...");
            rows = applyOverridesForStandardRows(parsed);
            setConvertResult({
              converted_rows: parsed.length,
              converted_ok: rows.length,
              converted_failed: parsed.length - rows.length,
              errors: [],
              items: rows,
              handles_summary: [],
              resolved_author_handle: null,
              resolved_kol_id: null,
              kol_created: false,
              skipped_not_followed_count: 0,
              skipped_not_followed_samples: [],
            });
            detectedHandles = Array.from(
              new Set(rows.map((item) => (item.author_handle || "").trim().toLowerCase()).filter(Boolean)),
            );
            convertedRowsTotal = parsed.length;
          } else {
            setWorkflowStep("Detected raw export JSON, converting...");
            const converted = await convertViaApi(file);
            setConvertResult(converted);
            detectedHandles = converted.handles_summary.map((item) => item.author_handle);
            syncSelectedHandles(detectedHandles);
            rows = converted.items;
            convertedRowsTotal = converted.converted_rows;
          }
        } catch {
          setWorkflowStep("JSON parse failed in browser, converting via backend...");
          const converted = await convertViaApi(file);
          setConvertResult(converted);
          detectedHandles = converted.handles_summary.map((item) => item.author_handle);
          syncSelectedHandles(detectedHandles);
          rows = converted.items;
          convertedRowsTotal = converted.converted_rows;
        }
      } else {
        setWorkflowStep("Converting upload via backend...");
        const converted = await convertViaApi(file);
        setConvertResult(converted);
        detectedHandles = converted.handles_summary.map((item) => item.author_handle);
        syncSelectedHandles(detectedHandles);
        rows = converted.items;
        convertedRowsTotal = converted.converted_rows;
      }

      const activeHandleSet = new Set(
        (selectedHandles.length > 0 ? selectedHandles : detectedHandles).map((item) => item.trim().toLowerCase()),
      );
      if (activeHandleSet && activeHandleSet.size > 0) {
        rows = rows.filter((item) =>
          activeHandleSet.has((item.resolved_author_handle || item.author_handle).trim().toLowerCase()),
        );
      }

      setConvertedCount(convertedRowsTotal || rows.length);
      if (rows.length === 0) throw new Error("No rows left after conversion/filtering.");

      setWorkflowStep("Importing into raw_posts...");
      const imported = await importRows(rows);
      setImportStats(imported);

      if (autoExtract && Object.keys(imported.imported_by_handle).length > 0) {
        setWorkflowStep("Resuming extraction for pending/failed/no_extraction posts...");
        const totals: ExtractBatchStats = {
          requested_count: 0,
          success_count: 0,
          skipped_count: 0,
          skipped_already_extracted_count: 0,
          skipped_already_pending_count: 0,
          skipped_already_success_count: 0,
          skipped_already_has_result_count: 0,
          skipped_already_rejected_count: 0,
          skipped_already_approved_count: 0,
          skipped_due_to_import_limit_count: 0,
          skipped_not_followed_count: 0,
          failed_count: 0,
          auto_rejected_count: 0,
          resumed_requested_count: 0,
          resumed_success: 0,
          resumed_failed: 0,
          resumed_skipped: 0,
        };
        const allIds = Array.from(
          new Set(
            Object.values(imported.imported_by_handle).flatMap((byHandle) => byHandle.raw_post_ids ?? []),
          ),
        );
        if (allIds.length > 0) {
          const extracted = await runExtractJob(allIds, extractBatchSize, "pending_or_failed", allIds.length);
          totals.requested_count = extracted.requested_count;
          totals.success_count = extracted.success_count;
          totals.skipped_count = extracted.skipped_count;
          totals.skipped_already_extracted_count = extracted.skipped_already_extracted_count;
          totals.skipped_already_pending_count = extracted.skipped_already_pending_count;
          totals.skipped_already_success_count = extracted.skipped_already_success_count;
          totals.skipped_already_has_result_count = extracted.skipped_already_has_result_count;
          totals.skipped_already_rejected_count = extracted.skipped_already_rejected_count;
          totals.skipped_already_approved_count = extracted.skipped_already_approved_count;
          totals.skipped_due_to_import_limit_count = extracted.skipped_due_to_import_limit_count;
          totals.skipped_not_followed_count = extracted.skipped_not_followed_count;
          totals.failed_count = extracted.failed_count;
          totals.auto_rejected_count = extracted.auto_rejected_count;
          totals.resumed_requested_count = extracted.resumed_requested_count;
          totals.resumed_success = extracted.resumed_success;
          totals.resumed_failed = extracted.resumed_failed;
          totals.resumed_skipped = extracted.resumed_skipped;
        }
        setExtractByHandle({});
        setExtractStats(totals);
      }

      setWorkflowStep("Refreshing progress...");
      await refreshProgress();
      if (autoExtract) {
        await new Promise((resolve) => window.setTimeout(resolve, 300));
        await refreshProgress();
      }
      if (autoGenerateDigest) await generateDigest(digestDate);
      setWorkflowStep("Completed.");
    } catch (err) {
      setWorkflowError(err instanceof Error ? err.message : "Workflow failed");
      setWorkflowStep(null);
    } finally {
      setWorkflowBusy(false);
    }
  };

  const handleFollowingImport = async () => {
    if (!followingFile) {
      setFollowingImportError("Please choose a following JSON file first.");
      return;
    }
    setFollowingImportBusy(true);
    setFollowingImportError(null);
    setFollowingImportStats(null);
    try {
      const params = new URLSearchParams({ filename: followingFile.name });
      const res = await fetch(`/api/ingest/x/following/import?${params.toString()}`, {
        method: "POST",
        headers: { "Content-Type": followingFile.type || "application/octet-stream" },
        body: followingFile,
      });
      const body = (await res.json()) as FollowingImportStats | { detail?: string };
      if (!res.ok) {
        throw new Error("detail" in body ? (body.detail ?? "Import following failed") : "Import following failed");
      }
      setFollowingImportStats(body as FollowingImportStats);
      await refreshCleanupScope();
    } catch (err) {
      setFollowingImportError(err instanceof Error ? err.message : "Import following failed");
    } finally {
      setFollowingImportBusy(false);
    }
  };

  return (
    <main style={{ padding: "24px", fontFamily: "monospace", display: "grid", gap: "12px" }}>
      <h1>Ingest</h1>
      <p>
        手动单条入口 + X 文件拖拽导入。<Link href="/dashboard">返回 Dashboard</Link>
      </p>

      <section style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "10px", maxWidth: "900px" }}>
        <h2 style={{ marginTop: 0 }}>Drag & Drop X Import</h2>
        <p style={{ marginTop: 0 }}>
          支持原始导出 JSON/CSV 与标准 x_import.json。标准 JSON 会走直导入路径；否则先调用转换接口。
        </p>
        <label>
          <input
            type="checkbox"
            checked={onlyFollowedKols}
            onChange={(event) => setOnlyFollowedKols(event.target.checked)}
          />{" "}
          Only import/analyze followed KOLs (enabled only)
        </label>
        <p style={{ margin: "4px 0 0 0", color: "#555" }}>
          服务端默认也会开启该保护；关闭时会允许未知 handle 入库并自动建 KOL。
        </p>

        <div style={{ display: "grid", gap: "8px", maxWidth: "700px" }}>
          {convertResult && convertResult.handles_summary.length > 1 && (
            <div style={{ border: "1px solid #eee", borderRadius: "8px", padding: "8px", display: "grid", gap: "6px" }}>
              <strong>Import all handles（默认）</strong>
              {convertResult.handles_summary.map((item) => {
                const checked = selectedHandles.includes(item.author_handle);
                return (
                  <label key={item.author_handle} style={{ display: "inline-flex", gap: "6px", alignItems: "center" }}>
                    <input
                      type="checkbox"
                      checked={checked}
                      onChange={(event) => {
                        setSelectedHandles((prev) => {
                          if (event.target.checked) return Array.from(new Set([...prev, item.author_handle]));
                          return prev.filter((handle) => handle !== item.author_handle);
                        });
                      }}
                    />
                    @{item.author_handle} count={item.count} {item.will_create_kol ? "（将自动创建 KOL）" : ""}
                  </label>
                );
              })}
            </div>
          )}

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px" }}>
            <label>
              start_date
              <input type="date" value={startDate} onChange={(event) => setStartDate(event.target.value)} style={{ display: "block", width: "100%" }} />
            </label>
            <label>
              end_date
              <input type="date" value={endDate} onChange={(event) => setEndDate(event.target.value)} style={{ display: "block", width: "100%" }} />
            </label>
          </div>
          <p style={{ margin: 0, fontSize: "12px", color: "#666" }}>
            时间范围按 UTC 日期过滤，start_date / end_date 都是包含边界（inclusive）。
          </p>

          <label>
            <input type="checkbox" checked={autoExtract} onChange={(event) => setAutoExtract(event.target.checked)} /> 导入后自动批量抽取（按 batch size 分批）
          </label>
          <label>
            extract batch size
            <select
              value={extractBatchSize}
              onChange={(event) => setExtractBatchSize(Number(event.target.value))}
              style={{ display: "block", width: "100%" }}
            >
              <option value={10}>10</option>
              <option value={20}>20</option>
              <option value={50}>50</option>
            </select>
          </label>
          <label>
            <input type="checkbox" checked={autoGenerateDigest} onChange={(event) => setAutoGenerateDigest(event.target.checked)} /> 导入后自动生成 digest
          </label>
          <label>
            digest date
            <input type="date" value={digestDate} onChange={(event) => setDigestDate(event.target.value)} style={{ display: "block", width: "100%" }} />
          </label>
        </div>

        <div
          onDragOver={(event) => {
            event.preventDefault();
            setDragActive(true);
          }}
          onDragLeave={(event) => {
            event.preventDefault();
            setDragActive(false);
          }}
          onDrop={onDropFile}
          style={{
            marginTop: "10px",
            border: `2px dashed ${dragActive ? "#2684ff" : "#999"}`,
            borderRadius: "8px",
            padding: "16px",
            background: dragActive ? "#f2f8ff" : "#fafafa",
          }}
        >
          <p style={{ marginTop: 0 }}>Drop JSON/CSV here, or choose file:</p>
          <input
            type="file"
            accept=".json,.csv,application/json,text/csv"
            onChange={(event) => onSelectFile(event.target.files?.[0] ?? null)}
          />
          <p style={{ marginBottom: 0 }}>Selected: {file ? file.name : "none"}</p>
        </div>

        <div style={{ marginTop: "10px", display: "flex", gap: "8px", flexWrap: "wrap" }}>
          <button type="button" onClick={() => void handleImportWorkflow()} disabled={workflowBusy}>
            {workflowBusy ? "Running..." : "Convert + Import"}
          </button>
          <button
            type="button"
            onClick={() => void generateDigest(digestDate)}
            disabled={workflowBusy}
          >
            Generate Digest Now
          </button>
          <button type="button" onClick={() => void refreshProgress()} disabled={workflowBusy}>
            刷新进度
          </button>
          <button type="button" onClick={() => cancelExtractPolling("Extract polling cancelled by user.")}>
            停止轮询
          </button>
          <label style={{ display: "inline-flex", alignItems: "center", gap: "4px" }}>
            <input
              type="checkbox"
              checked={clearAlsoDeleteRawPosts}
              onChange={(event) => setClearAlsoDeleteRawPosts(event.target.checked)}
              disabled={workflowBusy}
            />
            also delete raw posts
          </label>
          <button type="button" onClick={() => void clearPending()} disabled={workflowBusy}>
            Clear Pending (Delete)
          </button>
          <Link href={`/digests/${generatedDigestDate ?? digestDate}`}>Open Digest</Link>
        </div>

        <section style={{ border: "1px solid #eee", borderRadius: "8px", padding: "10px", marginTop: "10px" }}>
          <h3 style={{ marginTop: 0 }}>Admin Cleanup</h3>
          <p style={{ marginTop: 0 }}>
            所有操作都需要输入 YES；raw_posts 删除必须同时满足 enable_cascade=true。
          </p>

          <div style={{ display: "grid", gap: "10px" }}>
            <div style={{ border: "1px solid #f0f0f0", borderRadius: "6px", padding: "8px", display: "grid", gap: "6px" }}>
              <strong>Delete KOL</strong>
              <label>
                KOL ID
                <input
                  list="cleanup-kol-list"
                  value={cleanupKolId}
                  onChange={(event) => setCleanupKolId(event.target.value)}
                  style={{ display: "block", width: "100%" }}
                />
                <datalist id="cleanup-kol-list">
                  {kols.map((item) => (
                    <option key={item.id} value={item.id}>
                      @{item.handle}
                    </option>
                  ))}
                </datalist>
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={cleanupKolCascade}
                  onChange={(event) => setCleanupKolCascade(event.target.checked)}
                />{" "}
                enable_cascade
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={cleanupKolDeleteRawPosts}
                  onChange={(event) => setCleanupKolDeleteRawPosts(event.target.checked)}
                />{" "}
                also_delete_raw_posts
              </label>
              <button type="button" onClick={() => void deleteKolCleanup()} disabled={workflowBusy}>
                Delete KOL (Hard)
              </button>
            </div>

            <div style={{ border: "1px solid #f0f0f0", borderRadius: "6px", padding: "8px", display: "grid", gap: "6px" }}>
              <strong>Delete Asset</strong>
              <label>
                Asset ID
                <input
                  list="cleanup-asset-list"
                  value={cleanupAssetId}
                  onChange={(event) => setCleanupAssetId(event.target.value)}
                  style={{ display: "block", width: "100%" }}
                />
                <datalist id="cleanup-asset-list">
                  {assets.map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.symbol}
                    </option>
                  ))}
                </datalist>
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={cleanupAssetCascade}
                  onChange={(event) => setCleanupAssetCascade(event.target.checked)}
                />{" "}
                enable_cascade (delete asset + aliases)
              </label>
              <button type="button" onClick={() => void deleteAssetCleanup()} disabled={workflowBusy}>
                Delete Asset (Hard)
              </button>
            </div>

            <div style={{ border: "1px solid #f0f0f0", borderRadius: "6px", padding: "8px", display: "grid", gap: "6px" }}>
              <strong>Delete Digest By Date/Profile</strong>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px" }}>
                <label>
                  profile_id
                  <input
                    type="number"
                    min={1}
                    value={cleanupDigestProfileId}
                    onChange={(event) => setCleanupDigestProfileId(event.target.value)}
                    style={{ display: "block", width: "100%" }}
                  />
                </label>
                <label>
                  digest_date
                  <input
                    type="date"
                    value={cleanupDigestDate}
                    onChange={(event) => setCleanupDigestDate(event.target.value)}
                    style={{ display: "block", width: "100%" }}
                  />
                </label>
              </div>
              <button type="button" onClick={() => void deleteDigestByDateCleanup()} disabled={workflowBusy}>
                Delete Digest By Date
              </button>
            </div>
          </div>

          {assetsError && <p style={{ color: "crimson", marginBottom: 0 }}>{assetsError}</p>}
        </section>

        {workflowStep && <p style={{ marginBottom: 0 }}>step: {workflowStep}</p>}
        {workflowError && <p style={{ color: "crimson", marginBottom: 0 }}>{workflowError}</p>}
        {workflowRequestId && <p style={{ color: "#666", marginTop: "4px", marginBottom: 0 }}>request_id: {workflowRequestId}</p>}

        {convertedCount !== null && <p style={{ marginBottom: 0 }}>converted_rows={convertedCount}</p>}
        {convertResult && (
          <p style={{ margin: 0 }}>
            converted_ok={convertResult.converted_ok}, converted_failed={convertResult.converted_failed},
            skipped_not_followed={convertResult.skipped_not_followed_count}
          </p>
        )}
        {convertResult && convertResult.skipped_not_followed_samples.length > 0 && (
          <details>
            <summary>
              skipped_not_followed_samples ({convertResult.skipped_not_followed_count}) - showing first{" "}
              {Math.min(20, convertResult.skipped_not_followed_samples.length)}
            </summary>
            <ul>
              {convertResult.skipped_not_followed_samples.slice(0, 20).map((item) => (
                <li key={`${item.row_index}-${item.author_handle ?? "-"}-${item.external_id ?? "-"}`}>
                  row={item.row_index}, handle={item.author_handle ?? "-"}, external_id={item.external_id ?? "-"}, reason=
                  {item.reason}
                </li>
              ))}
            </ul>
          </details>
        )}
        {convertResult && convertResult.handles_summary.length > 0 && (
          <div>
            <strong>handles_summary</strong>
            <ul>
              {convertResult.handles_summary.map((item) => (
                <li key={item.author_handle}>
                  @{item.author_handle}: count={item.count}, will_create_kol={item.will_create_kol ? "true" : "false"}
                </li>
              ))}
            </ul>
          </div>
        )}
        {convertResult && convertResult.resolved_author_handle && (
          <p style={{ margin: 0 }}>
            Detected handle: @{convertResult.resolved_author_handle} (Create/Use KOL on import)
          </p>
        )}
        {convertResult && convertResult.errors.length > 0 && (
          <details>
            <summary>
              convert errors ({convertResult.errors.length}) - showing first{" "}
              {Math.min(20, convertResult.errors.length)}
            </summary>
            <ul>
              {convertResult.errors.slice(0, 20).map((item) => (
                <li key={`${item.row_index}-${item.external_id ?? "none"}-${item.reason}`}>
                  row={item.row_index}, external_id={item.external_id ?? "-"}, url={item.url ?? "-"}, reason=
                  {item.reason}
                </li>
              ))}
            </ul>
            <button
              type="button"
              onClick={() => {
                const blob = new Blob([JSON.stringify(convertResult.errors, null, 2)], { type: "application/json" });
                const href = URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = href;
                a.download = "x_convert_errors.json";
                document.body.appendChild(a);
                a.click();
                a.remove();
                URL.revokeObjectURL(href);
              }}
            >
              Download errors JSON
            </button>
          </details>
        )}
        {importStats && (
          <div style={{ marginBottom: 0 }}>
            <p style={{ margin: 0 }}>
              imported_rows={importStats.inserted_raw_posts_count}, dedup_skipped_count={importStats.dedup_skipped_count},
              dedup_existing_ids={importStats.dedup_existing_raw_post_ids.length}, warnings={importStats.warnings_count}
            </p>
            <p style={{ margin: 0 }}>
              import-trigger-extract: success={importStats.extract_success_count}, failed={importStats.extract_failed_count},
              skipped_already_extracted={importStats.skipped_already_extracted_count}, skipped_not_followed=
              {importStats.skipped_not_followed_count}
            </p>
            {(importStats.resolved_author_handle || importStats.resolved_kol_id) && (
              <p style={{ margin: 0 }}>
                resolved_author_handle={importStats.resolved_author_handle ?? "-"}, resolved_kol_id=
                {importStats.resolved_kol_id ?? "-"}, kol_created={importStats.kol_created ? "true" : "false"}
              </p>
            )}
            {Object.keys(importStats.imported_by_handle).length > 0 && (
              <div>
                <strong>imported_by_handle</strong>
                <ul>
                  {Object.entries(importStats.imported_by_handle).map(([handle, stats]) => (
                    <li key={handle}>
                      @{handle}: received={stats.received}, inserted={stats.inserted}, dedup={stats.dedup}, warnings=
                      {stats.warnings}
                      {extractByHandle[handle]
                        ? `, extracted=${extractByHandle[handle].success_count}, resumed=${extractByHandle[handle].resumed_success}, skipped=${extractByHandle[handle].resumed_skipped}`
                        : ""}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {importStats.created_kols.length > 0 && (
              <div>
                <strong>created_kols</strong>
                <ul>
                  {importStats.created_kols.map((item) => (
                    <li key={item.id}>
                      #{item.id} @{item.handle} {item.name ?? ""}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
        {importStats && importStats.skipped_not_followed_samples.length > 0 && (
          <details>
            <summary>
              import skipped_not_followed_samples ({importStats.skipped_not_followed_count}) - showing first{" "}
              {Math.min(20, importStats.skipped_not_followed_samples.length)}
            </summary>
            <ul>
              {importStats.skipped_not_followed_samples.slice(0, 20).map((item) => (
                <li key={`${item.row_index}-${item.author_handle ?? "-"}-${item.external_id ?? "-"}`}>
                  row={item.row_index}, handle={item.author_handle ?? "-"}, external_id={item.external_id ?? "-"}, reason=
                  {item.reason}
                </li>
              ))}
            </ul>
          </details>
        )}
        {extractStats && (
          <div style={{ marginBottom: 0 }}>
            <p style={{ margin: 0 }}>
              extracted_rows={extractStats.success_count}, extract_requested={extractStats.requested_count}, extract_failed=
              {extractStats.failed_count}, extract_skipped={extractStats.skipped_count}, skipped_already_extracted=
              {extractStats.skipped_already_extracted_count}
            </p>
            <p style={{ margin: 0 }}>
              skipped_already_pending={extractStats.skipped_already_pending_count}, skipped_already_success=
              {extractStats.skipped_already_success_count}
            </p>
            <p style={{ margin: 0 }}>
              skipped_already_has_result={extractStats.skipped_already_has_result_count}, auto_rejected=
              {extractStats.auto_rejected_count}
            </p>
            <p style={{ margin: 0 }}>
              skipped_already_rejected={extractStats.skipped_already_rejected_count}, skipped_already_approved=
              {extractStats.skipped_already_approved_count}, skipped_due_to_import_limit=
              {extractStats.skipped_due_to_import_limit_count}, skipped_not_followed={extractStats.skipped_not_followed_count}
            </p>
            <p style={{ margin: 0 }}>
              resumed_requested_count={extractStats.resumed_requested_count}, resumed_success={extractStats.resumed_success},
              resumed_failed={extractStats.resumed_failed}, resumed_skipped={extractStats.resumed_skipped}
            </p>
          </div>
        )}
        {!workflowBusy && extractStats && (
          <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
            <Link href="/extractions?status=pending">Open pending list</Link>
            <button type="button" onClick={() => void refreshProgress()}>
              刷新进度
            </button>
            <Link href={`/extractions?status=pending&t=${Date.now()}`}>刷新 extractions</Link>
          </div>
        )}
        {importStats && importStats.warnings.length > 0 && (
          <div>
            <strong>warnings</strong>
            <ul>
              {importStats.warnings.slice(0, 10).map((item, idx) => (
                <li key={`${idx}-${item}`}>{item}</li>
              ))}
            </ul>
          </div>
        )}
        {progress && (
          <p style={{ marginBottom: 0 }}>
            progress[{progress.scope}] total={progress.total_raw_posts}, success={progress.extracted_success_count},
            pending={progress.pending_count}, failed={progress.failed_count}, no_extraction={progress.no_extraction_count}
          </p>
        )}
      </section>

      <section style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "10px", maxWidth: "900px" }}>
        <h2 style={{ marginTop: 0 }}>Import Following → KOLs</h2>
        <p style={{ marginTop: 0 }}>上传 twitter-正在关注-*.json，自动同步 following=true 的 KOL 到数据库并启用。</p>
        <input
          type="file"
          accept=".json,application/json"
          onChange={(event) => setFollowingFile(event.target.files?.[0] ?? null)}
        />
        <p style={{ margin: "8px 0 0 0" }}>Selected: {followingFile ? followingFile.name : "none"}</p>
        <button type="button" onClick={() => void handleFollowingImport()} disabled={followingImportBusy} style={{ marginTop: "8px" }}>
          {followingImportBusy ? "Importing..." : "Import Following"}
        </button>
        {followingImportError && <p style={{ color: "crimson", marginBottom: 0 }}>{followingImportError}</p>}
        {followingImportStats && (
          <div style={{ marginTop: "8px" }}>
            <p style={{ margin: 0 }}>
              received_rows={followingImportStats.received_rows}, following_true_rows={followingImportStats.following_true_rows},
              created={followingImportStats.created_kols_count}, updated={followingImportStats.updated_kols_count},
              skipped={followingImportStats.skipped_count}
            </p>
            {followingImportStats.created_kols.length > 0 && (
              <p style={{ margin: "4px 0 0 0" }}>
                created: {followingImportStats.created_kols.map((item) => `#${item.id}@${item.handle}`).join(", ")}
              </p>
            )}
            {followingImportStats.updated_kols.length > 0 && (
              <p style={{ margin: "4px 0 0 0" }}>
                updated: {followingImportStats.updated_kols.map((item) => `#${item.id}@${item.handle}`).join(", ")}
              </p>
            )}
            {followingImportStats.errors.length > 0 && (
              <details>
                <summary>
                  errors ({followingImportStats.errors.length}) - showing first {Math.min(20, followingImportStats.errors.length)}
                </summary>
                <ul>
                  {followingImportStats.errors.slice(0, 20).map((item) => (
                    <li key={`${item.row_index}-${item.reason}`}>
                      row={item.row_index}, reason={item.reason}, raw={item.raw_snippet}
                    </li>
                  ))}
                </ul>
              </details>
            )}
          </div>
        )}
      </section>

      <section style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "10px", maxWidth: "760px" }}>
        <h2 style={{ marginTop: 0 }}>Extractor Status</h2>
        {statusError && <p style={{ color: "crimson" }}>{statusError}</p>}
        {!statusError && !extractorStatus && <p>Loading extractor status...</p>}
        {extractorStatus && (
          <div style={{ display: "grid", gap: "4px" }}>
            <p style={{ margin: 0 }}>
              EXTRACTOR_MODE: <strong>{extractorStatus.mode}</strong> | base_url: {extractorStatus.base_url}
            </p>
            <p style={{ margin: 0 }}>
              default_model: {extractorStatus.default_model} | has_api_key: {extractorStatus.has_api_key ? "yes" : "no"} |
              call_budget_remaining: {extractorStatus.call_budget_remaining ?? "unlimited"} | max_output_tokens: {extractorStatus.max_output_tokens}
            </p>
          </div>
        )}
      </section>

      <form onSubmit={onManualSubmit} style={{ display: "grid", gap: "8px", maxWidth: "760px" }}>
        <h2 style={{ marginBottom: 0 }}>Manual Single Ingest</h2>
        <label>
          platform
          <select
            value={form.platform}
            onChange={(event) => setForm((prev) => ({ ...prev, platform: event.target.value as FormState["platform"] }))}
            style={{ display: "block", width: "100%" }}
          >
            <option value="x">x</option>
            <option value="reddit">reddit</option>
            <option value="other">other</option>
          </select>
        </label>

        <label>
          author_handle
          <input value={form.author_handle} onChange={(event) => setForm((prev) => ({ ...prev, author_handle: event.target.value }))} style={{ display: "block", width: "100%" }} required />
        </label>

        <label>
          url
          <input type="url" value={form.url} onChange={(event) => setForm((prev) => ({ ...prev, url: event.target.value }))} style={{ display: "block", width: "100%" }} required />
        </label>

        <label>
          content_text
          <textarea value={form.content_text} onChange={(event) => setForm((prev) => ({ ...prev, content_text: event.target.value }))} rows={5} style={{ display: "block", width: "100%" }} required />
        </label>

        <label>
          posted_at (optional)
          <input type="datetime-local" value={form.posted_at} onChange={(event) => setForm((prev) => ({ ...prev, posted_at: event.target.value }))} style={{ display: "block", width: "100%" }} />
        </label>

        <button type="submit" disabled={submitting}>
          {submitting ? "Submitting..." : "Create Raw Post + Pending Extraction"}
        </button>
      </form>

      {error && <p style={{ color: "crimson" }}>{error}</p>}
      {result && (
        <div style={{ border: "1px solid #96d29a", borderRadius: "8px", padding: "10px", background: "#f4fff5" }}>
          <p style={{ margin: 0 }}>
            提交成功。raw_post_id={result.extraction.raw_post_id}，extraction_id={result.extraction_id}，extractor_name={result.extraction.extractor_name}
          </p>
          {result.extraction.last_error && <p style={{ color: "crimson", marginBottom: 0 }}>last_error: {result.extraction.last_error}</p>}
          <p style={{ marginBottom: 0 }}>
            <Link href={`/extractions/${result.extraction_id}`}>去审核该 extraction</Link>
          </p>
        </div>
      )}
    </main>
  );
}
