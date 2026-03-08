"use client";

import type { DragEvent } from "react";
import { useEffect, useMemo, useRef, useState } from "react";
import { useImportProgressHub } from "../components/import-progress-hub";

type ExtractorStatus = {
  mode: "auto" | "dummy" | "openai" | string;
  has_api_key: boolean;
  default_model: string;
  base_url: string;
  call_budget_remaining: number | null;
  max_output_tokens: number;
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
  pending_failed_dedup_count: number;
  pending_failed_dedup_ids: number[];
  pending_failed_reason_breakdown: Record<string, number>;
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

type RunExtractJobHooks = {
  onJobCreated?: (jobId: string) => void;
  onJobProgress?: (job: ExtractJob) => void;
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

type HandlePick = {
  author_handle: string;
  count: number;
  will_create_kol: boolean;
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
  const rawMessage = err instanceof Error ? err.message : String(err ?? "未知错误");
  return [
    `转换失败：无法连接后端 ${baseUrl}`,
    "请确认后端服务已启动，或检查是否被代理层的大文件限制中断（例如 10MB 限制导致 ECONNRESET/socket hang up）。",
    `原始错误: ${rawMessage}`,
  ].join(" ");
}

function formatImportNetworkError(err: unknown): string {
  const baseUrl = getDirectApiBaseUrl();
  const rawMessage = err instanceof Error ? err.message : String(err ?? "未知错误");
  return [
    `导入失败：无法连接后端 ${baseUrl}`,
    "请检查后端是否启动、CORS 是否允许当前前端域名，或网络是否中断。",
    `原始错误: ${rawMessage}`,
  ].join(" ");
}

function formatExtractJobNetworkError(stage: "create" | "poll", err: unknown): string {
  const baseUrl = getDirectApiBaseUrl();
  const rawMessage = err instanceof Error ? err.message : String(err ?? "未知错误");
  const action = stage === "create" ? "创建抽取任务" : "轮询抽取任务";
  return [
    `抽取任务失败：${action}时无法连接后端 ${baseUrl}`,
    "请检查后端是否启动、CORS 是否允许当前前端域名，或网络是否中断。",
    `原始错误: ${rawMessage}`,
  ].join(" ");
}

function classifyExtractFailureCategory(message: string): string {
  const lower = message.toLowerCase();
  if (lower.includes("手动停止")) return "手动停止";
  if (lower.includes("无法连接后端") || lower.includes("network") || lower.includes("econnreset")) return "网络连接失败";
  if (lower.includes("status=429") || lower.includes("rate_limit") || lower.includes("rate limited")) return "限流";
  if (lower.includes("timeout")) return "超时";
  if (lower.includes("status=503") || lower.includes("service unavailable")) return "服务不可用";
  if (lower.includes("parse") || lower.includes("json")) return "解析失败";
  if (lower.includes("budget_exhausted") || lower.includes("import_limit_reached")) return "额度不足";
  return "处理失败";
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
  const trimmed = value.trim();
  const matched = trimmed.match(/^(\d{4}-\d{2}-\d{2})/);
  return matched ? matched[1] : null;
}

export default function IngestPage() {
  const importProgressHub = useImportProgressHub();
  const [extractorStatus, setExtractorStatus] = useState<ExtractorStatus | null>(null);
  const [extractBatchSize, setExtractBatchSize] = useState(20);
  const [statusError, setStatusError] = useState<string | null>(null);

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
  const [includeAllHandles, setIncludeAllHandles] = useState(true);
  const [convertedRowsForConfirm, setConvertedRowsForConfirm] = useState<XImportItem[] | null>(null);
  const [awaitingImportConfirm, setAwaitingImportConfirm] = useState(false);
  const [extractByHandle, setExtractByHandle] = useState<Record<string, ExtractBatchStats>>({});
  const [progress, setProgress] = useState<XProgress | null>(null);
  const [followingFile, setFollowingFile] = useState<File | null>(null);
  const [followingDragActive, setFollowingDragActive] = useState(false);
  const [followingImportBusy, setFollowingImportBusy] = useState(false);
  const [followingImportError, setFollowingImportError] = useState<string | null>(null);
  const [followingImportStats, setFollowingImportStats] = useState<FollowingImportStats | null>(null);
  const extractPollControllerRef = useRef<AbortController | null>(null);
  const extractPollRunIdRef = useRef(0);
  const extractPollAbortSourceRef = useRef<"manual" | "unmount" | "replace" | "none">("none");
  const cancelExtractPolling = (reason?: string, source: "manual" | "unmount" | "replace" = "manual") => {
    extractPollAbortSourceRef.current = source;
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
      cancelExtractPolling(undefined, "unmount");
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
          throw new Error(formatApiError("加载抽取器状态失败", extractorParsed.data, extractorParsed.textBody, extractorParsed.requestId));
        }
        setExtractorStatus(extractorParsed.data as ExtractorStatus);
      } catch (err) {
        setStatusError(err instanceof Error ? err.message : "加载抽取器状态失败");
      }
    };
    void loadStatus();
  }, []);

  useEffect(() => {
    if (!importProgressHub.state.visible) return;
    if (importProgressHub.state.message) {
      setWorkflowStep(importProgressHub.state.message);
    }
    if (importProgressHub.state.status === "error" && importProgressHub.state.error) {
      setWorkflowError(importProgressHub.state.error);
    }
  }, [
    importProgressHub.state.visible,
    importProgressHub.state.message,
    importProgressHub.state.status,
    importProgressHub.state.error,
  ]);

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
      const statusPart = typeof statusCode === "number" ? `状态=${statusCode}` : "状态=未知";
      const pathPart = requestPath ? `路径=${requestPath}` : "路径=未知";
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
      throw new Error(formatApiError("加载进度失败", parsed.data, parsed.textBody, parsed.requestId));
    }
    setProgress(parsed.data as XProgress);
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
      throw new Error(formatApiError("转换失败", parsed.data, parsed.textBody, parsed.requestId));
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
      throw new Error(formatApiError("导入失败", parsed.data, parsed.textBody, parsed.requestId));
    }
    return parsed.data as XImportStats;
  };

  const runExtractJob = async (
    ids: number[],
    batchSize: number,
    mode: "pending_only" | "pending_or_failed" | "force",
    aiCallLimitTotal?: number,
    hooks?: RunExtractJobHooks,
  ): Promise<ExtractBatchStats> => {
    const stableIds = Array.from(new Set(ids)).sort((a, b) => a - b);
    const normalizedAiLimit =
      typeof aiCallLimitTotal === "number" && Number.isFinite(aiCallLimitTotal)
        ? Math.max(0, Math.floor(aiCallLimitTotal))
        : null;
    const idempotencyKey = `ingest:${mode}:${Math.max(1, batchSize)}:${normalizedAiLimit ?? "none"}:${stableIds.join(",")}`.slice(0, 256);
    cancelExtractPolling(undefined, "replace");
    const pollController = new AbortController();
    extractPollControllerRef.current = pollController;
    extractPollAbortSourceRef.current = "none";
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
        throw new Error("抽取任务轮询已取消");
      }
      throw new Error(formatExtractJobNetworkError("create", err));
    }
    const createParsed = await parseApiResponse<ExtractJobCreateResponse>(createRes);
    setWorkflowRequestId(createParsed.requestId);
    if (!createRes.ok || !createParsed.data || !("job_id" in (createParsed.data as Record<string, unknown>))) {
      throw new Error(
        formatApiError(
          "创建抽取任务失败",
          createParsed.data,
          createParsed.textBody,
          createParsed.requestId,
          createParsed.statusCode,
          createParsed.requestPath,
        ),
      );
    }
    const jobId = (createParsed.data as ExtractJobCreateResponse).job_id;
    hooks?.onJobCreated?.(jobId);

    try {
      for (let attempt = 0; attempt < 1800; attempt += 1) {
        if (extractPollRunIdRef.current !== runId) {
          throw new Error("抽取任务轮询已被新的任务替代");
        }
        let res: Response;
        try {
          res = await fetch(buildDirectApiUrl(`${DIRECT_EXTRACT_JOBS_PATH}/${encodeURIComponent(jobId)}`, new URLSearchParams()), {
            cache: "no-store",
            signal: pollController.signal,
          });
        } catch (err) {
          if (err instanceof DOMException && err.name === "AbortError") {
            throw new Error("抽取任务轮询已取消");
          }
          throw new Error(formatExtractJobNetworkError("poll", err));
        }
        const parsed = await parseApiResponse<ExtractJob>(res);
        if (!res.ok || !parsed.data) {
          setWorkflowRequestId(parsed.requestId);
          throw new Error(
            formatApiError("加载抽取任务失败", parsed.data, parsed.textBody, parsed.requestId, parsed.statusCode, parsed.requestPath),
          );
        }
        const job = parsed.data as ExtractJob;
        hooks?.onJobProgress?.(job);
        setWorkflowRequestId(parsed.requestId);
        setWorkflowStep(
          `抽取任务 ${job.status}：成功=${job.success_count}，失败=${job.failed_count}，跳过=${job.skipped_count}，请求=${job.requested_count}，AI调用=${job.ai_call_used}${job.ai_call_limit_total !== null ? `/${job.ai_call_limit_total}` : ""}`,
        );
        if (job.is_terminal) {
          if (job.status === "failed" || job.status === "cancelled" || job.status === "timeout") {
            const summary = job.last_error_summary ? `: ${job.last_error_summary}` : "";
            throw new Error(`抽取任务失败${summary}${parsed.requestId ? ` (request_id=${parsed.requestId})` : ""}`);
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
      throw new Error("抽取任务轮询超时");
    } finally {
      if (extractPollControllerRef.current === pollController) {
        extractPollControllerRef.current = null;
      }
    }
  };

  const generateDigest = async (dateStr: string) => {
    setWorkflowStep(`正在生成 ${dateStr} 的日报...`);
    importProgressHub.updateRunning(`正在生成 ${dateStr} 的日报...`, "refreshing");
    const res = await fetch(`/api/digests/generate?date=${dateStr}`, { method: "POST" });
    const body = (await res.json()) as { detail?: string };
    if (!res.ok) throw new Error(body.detail ?? "生成日报失败");
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
    setIncludeAllHandles(true);
    setConvertedRowsForConfirm(null);
    setAwaitingImportConfirm(false);
  };

  const syncSelectedHandles = (handles: string[]) => {
    if (handles.length === 0) {
      setSelectedHandles([]);
      return;
    }
    setSelectedHandles(Array.from(new Set(handles.map((item) => item.trim().toLowerCase()).filter(Boolean))));
  };

  const summarizeHandlesFromRows = (rows: XImportItem[]): HandlePick[] => {
    const map = new Map<string, number>();
    rows.forEach((item) => {
      const key = (item.resolved_author_handle || item.author_handle || "").trim().toLowerCase();
      if (!key) return;
      map.set(key, (map.get(key) ?? 0) + 1);
    });
    return Array.from(map.entries())
      .map(([author_handle, count]) => ({ author_handle, count, will_create_kol: false }))
      .sort((a, b) => b.count - a.count || a.author_handle.localeCompare(b.author_handle));
  };

  const getCurrentHandlePicks = (): HandlePick[] => {
    if (!convertResult) return [];
    if (convertResult.handles_summary.length > 0) {
      return convertResult.handles_summary.map((item) => ({
        author_handle: item.author_handle.trim().toLowerCase(),
        count: item.count,
        will_create_kol: item.will_create_kol,
      }));
    }
    return summarizeHandlesFromRows(convertResult.items);
  };

  const onDropFile = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragActive(false);
    const dropped = event.dataTransfer.files?.[0] ?? null;
    onSelectFile(dropped);
  };

  const onSelectFollowingFile = (targetFile: File | null) => {
    setFollowingFile(targetFile);
    setFollowingImportError(null);
    setFollowingImportStats(null);
  };

  const onDropFollowingFile = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setFollowingDragActive(false);
    const dropped = event.dataTransfer.files?.[0] ?? null;
    onSelectFollowingFile(dropped);
  };

  const handleImportWorkflow = async () => {
    if (workflowBusy) return;
    if (!file) {
      setWorkflowError("请先选择文件。");
      return;
    }
    setWorkflowBusy(true);
    importProgressHub.begin("正在转换文件...");
    setWorkflowError(null);
    setWorkflowRequestId(null);
    setWorkflowStep("正在读取文件...");
    setConvertResult(null);
    setImportStats(null);
    setExtractStats(null);
    setProgress(null);
    setConvertedRowsForConfirm(null);
    setAwaitingImportConfirm(false);
    setIncludeAllHandles(true);
    setExtractByHandle({});
    try {
      let rows: XImportItem[] = [];
      let convertedRowsTotal = 0;
      const maybeJson = file.name.toLowerCase().endsWith(".json");
      if (maybeJson) {
        const text = await file.text();
        try {
          const parsed = JSON.parse(text) as unknown;
          if (isStandardImportJson(parsed)) {
            setWorkflowStep("检测到标准 x_import.json，直接导入...");
            importProgressHub.updateRunning("检测到标准 x_import.json，正在准备导入...", "converting");
            rows = applyOverridesForStandardRows(parsed);
            setConvertResult({
              converted_rows: parsed.length,
              converted_ok: rows.length,
              converted_failed: parsed.length - rows.length,
              errors: [],
              items: rows,
              handles_summary: summarizeHandlesFromRows(rows),
              resolved_author_handle: null,
              resolved_kol_id: null,
              kol_created: false,
              skipped_not_followed_count: 0,
              skipped_not_followed_samples: [],
            });
            convertedRowsTotal = parsed.length;
          } else {
            setWorkflowStep("检测到原始导出 JSON，正在转换...");
            importProgressHub.updateRunning("检测到原始导出 JSON，正在转换...", "converting");
            const converted = await convertViaApi(file);
            setConvertResult(converted);
            rows = converted.items;
            convertedRowsTotal = converted.converted_rows;
          }
        } catch {
          setWorkflowStep("浏览器解析 JSON 失败，改用后端转换...");
          importProgressHub.updateRunning("浏览器解析 JSON 失败，改用后端转换...", "converting");
          const converted = await convertViaApi(file);
          setConvertResult(converted);
          rows = converted.items;
          convertedRowsTotal = converted.converted_rows;
        }
      } else {
        setWorkflowStep("正在通过后端转换上传文件...");
        importProgressHub.updateRunning("正在通过后端转换上传文件...", "converting");
        const converted = await convertViaApi(file);
        setConvertResult(converted);
        rows = converted.items;
        convertedRowsTotal = converted.converted_rows;
      }

      const handles = Array.from(
        new Set(rows.map((item) => (item.resolved_author_handle || item.author_handle || "").trim().toLowerCase()).filter(Boolean)),
      );
      syncSelectedHandles(handles);
      setConvertedRowsForConfirm(rows);
      setConvertedCount(convertedRowsTotal || rows.length);
      if (rows.length === 0) throw new Error("转换后无可导入数据。");
      setAwaitingImportConfirm(true);
      setWorkflowStep("转换完成。请确认账号选择后点击“确认并导入”。");
      importProgressHub.markPendingConfirm("转换完成，等待你确认并导入。");
    } catch (err) {
      const message = err instanceof Error ? err.message : "流程执行失败";
      setWorkflowError(message);
      setWorkflowStep(null);
      importProgressHub.markError(message);
    } finally {
      setWorkflowBusy(false);
    }
  };

  const handleConfirmImport = async () => {
    if (workflowBusy) return;
    if (!convertedRowsForConfirm || convertedRowsForConfirm.length === 0) {
      setWorkflowError("请先执行转换。");
      return;
    }
    setWorkflowBusy(true);
    setWorkflowError(null);
    setWorkflowRequestId(null);
    importProgressHub.updateRunning("确认完成，准备导入数据...", "importing");
    try {
      let rows = convertedRowsForConfirm;
      if (!includeAllHandles) {
        const activeHandleSet = new Set(selectedHandles.map((item) => item.trim().toLowerCase()).filter(Boolean));
        rows = rows.filter((item) => activeHandleSet.has((item.resolved_author_handle || item.author_handle).trim().toLowerCase()));
      }
      if (rows.length === 0) throw new Error("当前勾选账号下无可导入数据。");

      setWorkflowStep("正在导入到 raw_posts...");
      importProgressHub.updateRunning("正在导入到 raw_posts...", "importing");
      const imported = await importRows(rows);
      setImportStats(imported);
      if (imported.pending_failed_dedup_count > 0) {
        setWorkflowStep(
          `检测到 ${imported.pending_failed_dedup_count} 条待处理失败贴文，${
            autoExtract ? "将重新上传AI模型进行解析..." : "请开启自动抽取或到待处理列表执行一键重传。"
          }`,
        );
      }

      if (autoExtract && Object.keys(imported.imported_by_handle).length > 0) {
        setWorkflowStep("正在为 pending/failed/no_extraction 贴文续跑抽取...");
        importProgressHub.updateRunning("正在上传到AI模型处理...", "extracting");
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
        const allIds = Array.from(new Set(Object.values(imported.imported_by_handle).flatMap((byHandle) => byHandle.raw_post_ids ?? [])));
        if (allIds.length > 0) {
          const extracted = await runExtractJob(allIds, extractBatchSize, "pending_or_failed", allIds.length, {
            onJobCreated: (jobId) => importProgressHub.attachExtractJob(jobId),
            onJobProgress: (job) => {
              importProgressHub.applyExtractStats({
                requested: job.requested_count,
                success: job.success_count,
                failed: job.failed_count,
                skipped: job.skipped_count,
              });
              importProgressHub.updateRunning(
                `AI处理中：成功=${job.success_count}，失败=${job.failed_count}，完成=${job.success_count + job.failed_count + job.skipped_count}/${job.requested_count}`,
                "extracting",
              );
            },
          });
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
        importProgressHub.applyExtractStats({
          requested: totals.requested_count,
          success: totals.success_count,
          failed: totals.failed_count,
          skipped: totals.skipped_count,
        });
      }

      setWorkflowStep("正在刷新进度...");
      importProgressHub.updateRunning("正在刷新进度...", "refreshing");
      await refreshProgress();
      if (autoExtract) {
        await new Promise((resolve) => window.setTimeout(resolve, 300));
        await refreshProgress();
      }
      if (autoGenerateDigest) await generateDigest(digestDate);
      setAwaitingImportConfirm(false);
      setWorkflowStep("已完成。");
      importProgressHub.markSuccess("导入与AI处理已完成。");
    } catch (err) {
      const message = err instanceof Error ? err.message : "流程执行失败";
      const suppressByUnmountAbort = message === "抽取任务轮询已取消" && extractPollAbortSourceRef.current === "unmount";
      if (suppressByUnmountAbort) return;
      if (message === "抽取任务轮询已取消" && extractPollAbortSourceRef.current === "manual") {
        const manualStopped = "手动停止：已停止抽取轮询，未完成贴文保留在待处理列表。";
        setWorkflowError(manualStopped);
        setWorkflowStep(null);
        importProgressHub.markError(manualStopped);
        return;
      }
      const category = classifyExtractFailureCategory(message);
      const labeledMessage = `[${category}] ${message}`;
      setWorkflowError(labeledMessage);
      setWorkflowStep(null);
      importProgressHub.markError(labeledMessage);
    } finally {
      setWorkflowBusy(false);
    }
  };

  const handleFollowingImport = async () => {
    if (!followingFile) {
      setFollowingImportError("请先选择 following JSON 文件。");
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
        throw new Error("detail" in body ? (body.detail ?? "导入 following 失败") : "导入 following 失败");
      }
      setFollowingImportStats(body as FollowingImportStats);
    } catch (err) {
      setFollowingImportError(err instanceof Error ? err.message : "导入 following 失败");
    } finally {
      setFollowingImportBusy(false);
    }
  };

  return (
    <main style={{ padding: "24px", fontFamily: "monospace", display: "grid", gap: "12px" }}>
      <h1>导入中心</h1>
      <p>X 文件拖拽导入。</p>

      <section style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "10px", maxWidth: "900px" }}>
        <h2 style={{ marginTop: 0 }}>拖拽导入 X 数据</h2>
        <p style={{ marginTop: 0 }}>
          支持原始导出 JSON/CSV 与标准 x_import.json。标准 JSON 会走直导入路径；否则先调用转换接口。
        </p>
        <label>
          <input
            type="checkbox"
            checked={onlyFollowedKols}
            onChange={(event) => setOnlyFollowedKols(event.target.checked)}
          />{" "}
          仅导入/分析已关注的 KOL（仅启用状态）
        </label>
        <p style={{ margin: "4px 0 0 0", color: "#555" }}>
          服务端默认也会开启该保护；关闭时会允许未知 handle 入库并自动建 KOL。
        </p>

        <div style={{ display: "grid", gap: "8px", maxWidth: "700px" }}>
          {awaitingImportConfirm && (
            <div style={{ border: "1px solid #eee", borderRadius: "8px", padding: "8px", display: "grid", gap: "6px" }}>
              <strong>导入账号确认</strong>
              <label style={{ display: "inline-flex", gap: "6px", alignItems: "center" }}>
                <input
                  type="checkbox"
                  checked={includeAllHandles}
                  onChange={(event) => setIncludeAllHandles(event.target.checked)}
                />
                导入全部账号（默认）
              </label>
              {!includeAllHandles && (
                <div style={{ display: "grid", gap: "6px" }}>
                  {getCurrentHandlePicks().map((item) => {
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
                        @{item.author_handle} 数量={item.count} {item.will_create_kol ? "（将自动创建 KOL）" : ""}
                      </label>
                    );
                  })}
                </div>
              )}
            </div>
          )}

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px" }}>
            <label>
              开始日期
              <input type="date" value={startDate} onChange={(event) => setStartDate(event.target.value)} style={{ display: "block", width: "100%" }} />
            </label>
            <label>
              结束日期
              <input type="date" value={endDate} onChange={(event) => setEndDate(event.target.value)} style={{ display: "block", width: "100%" }} />
            </label>
          </div>
          <p style={{ margin: 0, fontSize: "12px", color: "#666" }}>
            时间范围按 UTC 日期过滤，start_date / end_date 都是包含边界（inclusive）。
          </p>

          <label>
            <input type="checkbox" checked={autoExtract} onChange={(event) => setAutoExtract(event.target.checked)} /> 导入后自动批量抽取（按批次大小分批）
          </label>
          <label>
            抽取批次大小
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
            <input type="checkbox" checked={autoGenerateDigest} onChange={(event) => setAutoGenerateDigest(event.target.checked)} /> 导入后自动生成日报
          </label>
          <label>
            日报日期
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
            border: `2px dashed ${dragActive ? "var(--accent)" : "var(--line-strong)"}`,
            borderRadius: "8px",
            padding: "16px",
            background: dragActive
              ? "color-mix(in srgb, var(--accent-soft) 70%, var(--bg-elev-strong) 30%)"
              : "color-mix(in srgb, var(--bg-elev-strong) 88%, transparent 12%)",
          }}
        >
          <p style={{ marginTop: 0 }}>将 JSON/CSV 拖拽到这里，或选择文件：</p>
          <input
            type="file"
            accept=".json,.csv,application/json,text/csv"
            onChange={(event) => onSelectFile(event.target.files?.[0] ?? null)}
          />
          <p style={{ marginBottom: 0 }}>已选择：{file ? file.name : "无"}</p>
        </div>

        <div style={{ marginTop: "10px", display: "flex", gap: "8px", flexWrap: "wrap" }}>
          <button type="button" onClick={() => void handleImportWorkflow()} disabled={workflowBusy}>
            {workflowBusy ? "执行中..." : "转换并准备导入"}
          </button>
          <button
            type="button"
            onClick={() => void handleConfirmImport()}
            disabled={workflowBusy || !awaitingImportConfirm}
          >
            {workflowBusy ? "执行中..." : "确认并导入"}
          </button>
          <button type="button" onClick={() => void refreshProgress()} disabled={workflowBusy}>
            刷新进度
          </button>
          <button type="button" onClick={() => cancelExtractPolling("用户已取消抽取任务轮询。", "manual")}>
            停止轮询
          </button>
        </div>
      </section>

        {workflowStep && <p style={{ marginBottom: 0 }}>步骤：{workflowStep}</p>}
        {workflowError && <p style={{ color: "crimson", marginBottom: 0 }}>{workflowError}</p>}
        {workflowRequestId && <p style={{ color: "#666", marginTop: "4px", marginBottom: 0 }}>请求 ID：{workflowRequestId}</p>}

        {convertedCount !== null && <p style={{ marginBottom: 0 }}>文件数据总数={convertedCount}</p>}
        {convertResult && <p style={{ margin: 0 }}>未关注排除（转换阶段）={convertResult.skipped_not_followed_count}</p>}
        {importStats && <p style={{ margin: 0 }}>导入接收={importStats.received_count}, 导入成功={importStats.inserted_raw_posts_count}</p>}
        {importStats && <p style={{ margin: 0 }}>去重排除={importStats.dedup_skipped_count}, 未关注排除（导入阶段）={importStats.skipped_not_followed_count}</p>}
        {importStats && importStats.pending_failed_dedup_count > 0 && (
          <p style={{ margin: 0, color: "#8a5800" }}>
            本次文件命中待处理失败贴文={importStats.pending_failed_dedup_count}（将/需重新上传AI模型解析）
          </p>
        )}
        {extractStats && (
          <p style={{ margin: 0 }}>
            上传到AI模型={extractStats.success_count + extractStats.failed_count}, AI处理成功={extractStats.success_count}
          </p>
        )}
        {extractStats && <p style={{ margin: 0 }}>AI处理失败={extractStats.failed_count}</p>}
        {convertResult && convertResult.skipped_not_followed_samples.length > 0 && (
          <details>
            <summary>
              非关注跳过样本（{convertResult.skipped_not_followed_count}）- 仅显示前{" "}
              {Math.min(20, convertResult.skipped_not_followed_samples.length)}
            </summary>
            <ul>
              {convertResult.skipped_not_followed_samples.slice(0, 20).map((item) => (
                <li key={`${item.row_index}-${item.author_handle ?? "-"}-${item.external_id ?? "-"}`}>
                  行={item.row_index}, handle={item.author_handle ?? "-"}, external_id={item.external_id ?? "-"}, 原因=
                  {item.reason}
                </li>
              ))}
            </ul>
          </details>
        )}
        {convertResult && getCurrentHandlePicks().length > 0 && (
          <details>
            <summary>账号汇总（默认折叠）</summary>
            <strong>账号汇总</strong>
            <ul>
              {getCurrentHandlePicks().map((item) => (
                <li key={item.author_handle}>
                  @{item.author_handle}: 数量={item.count}, 将创建KOL={item.will_create_kol ? "是" : "否"}
                </li>
              ))}
            </ul>
          </details>
        )}
        {convertResult && convertResult.resolved_author_handle && (
          <p style={{ margin: 0 }}>
            检测到账号：@{convertResult.resolved_author_handle}（导入时自动创建/复用 KOL）
          </p>
        )}
        {convertResult && convertResult.errors.length > 0 && (
          <details>
            <summary>
              转换错误（{convertResult.errors.length}）- 仅显示前{" "}
              {Math.min(20, convertResult.errors.length)}
            </summary>
            <ul>
              {convertResult.errors.slice(0, 20).map((item) => (
                <li key={`${item.row_index}-${item.external_id ?? "none"}-${item.reason}`}>
                  行={item.row_index}, external_id={item.external_id ?? "-"}, url={item.url ?? "-"}, 原因=
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
              下载错误 JSON
            </button>
          </details>
        )}
        {importStats && Object.keys(importStats.imported_by_handle).length > 0 && (
          <details>
            <summary>按账号导入统计（默认折叠）</summary>
            <ul>
              {Object.entries(importStats.imported_by_handle).map(([handle, stats]) => (
                <li key={handle}>
                  @{handle}: 接收={stats.received}, 插入={stats.inserted}, 去重={stats.dedup}, 警告={stats.warnings}
                  {extractByHandle[handle]
                    ? `, 抽取成功=${extractByHandle[handle].success_count}, 续跑成功=${extractByHandle[handle].resumed_success}, 跳过=${extractByHandle[handle].resumed_skipped}`
                    : ""}
                </li>
              ))}
            </ul>
          </details>
        )}
        {importStats && importStats.skipped_not_followed_samples.length > 0 && (
          <details>
            <summary>
              导入时非关注跳过样本（{importStats.skipped_not_followed_count}）- 仅显示前{" "}
              {Math.min(20, importStats.skipped_not_followed_samples.length)}
            </summary>
            <ul>
              {importStats.skipped_not_followed_samples.slice(0, 20).map((item) => (
                <li key={`${item.row_index}-${item.author_handle ?? "-"}-${item.external_id ?? "-"}`}>
                  行={item.row_index}, handle={item.author_handle ?? "-"}, external_id={item.external_id ?? "-"}, 原因=
                  {item.reason}
                </li>
              ))}
            </ul>
          </details>
        )}
        {importStats && importStats.created_kols.length > 0 && (
          <details>
            <summary>新建 KOL（默认折叠）</summary>
            <ul>
              {importStats.created_kols.map((item) => (
                <li key={item.id}>
                  #{item.id} @{item.handle} {item.name ?? ""}
                </li>
              ))}
            </ul>
          </details>
        )}
        {importStats && importStats.warnings.length > 0 && (
          <div>
            <strong>警告</strong>
            <ul>
              {importStats.warnings.slice(0, 10).map((item, idx) => (
                <li key={`${idx}-${item}`}>{item}</li>
              ))}
            </ul>
          </div>
        )}
        {progress && (
          <p style={{ marginBottom: 0 }}>
            进度[{progress.scope}] AI处理成功={Math.max(0, progress.total_raw_posts - progress.pending_count)}, 已通过=
            {progress.extracted_success_count}, 待处理={progress.pending_count}, 已拒绝={progress.failed_count}, 已入库=
            {progress.no_extraction_count}
          </p>
        )}
      <section style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "10px", maxWidth: "900px" }}>
        <h2 style={{ marginTop: 0 }}>导入关注列表 → KOL</h2>
        <p style={{ marginTop: 0 }}>上传 twitter-正在关注-*.json，自动同步 following=true 的 KOL 到数据库并启用。</p>
        <div
          onDragOver={(event) => {
            event.preventDefault();
            setFollowingDragActive(true);
          }}
          onDragLeave={(event) => {
            event.preventDefault();
            setFollowingDragActive(false);
          }}
          onDrop={onDropFollowingFile}
          style={{
            marginTop: "8px",
            border: `2px dashed ${followingDragActive ? "var(--accent)" : "var(--line-strong)"}`,
            borderRadius: "8px",
            padding: "16px",
            background: followingDragActive
              ? "color-mix(in srgb, var(--accent-soft) 70%, var(--bg-elev-strong) 30%)"
              : "color-mix(in srgb, var(--bg-elev-strong) 88%, transparent 12%)",
          }}
        >
          <p style={{ marginTop: 0 }}>将 following JSON 拖拽到这里，或选择文件：</p>
          <input
            type="file"
            accept=".json,application/json"
            onChange={(event) => onSelectFollowingFile(event.target.files?.[0] ?? null)}
          />
          <p style={{ margin: "8px 0 0 0" }}>已选择：{followingFile ? followingFile.name : "无"}</p>
        </div>
        <button type="button" onClick={() => void handleFollowingImport()} disabled={followingImportBusy} style={{ marginTop: "8px" }}>
          {followingImportBusy ? "导入中..." : "导入关注列表"}
        </button>
        {followingImportError && <p style={{ color: "crimson", marginBottom: 0 }}>{followingImportError}</p>}
        {followingImportStats && (
          <div style={{ marginTop: "8px" }}>
            <p style={{ margin: 0 }}>
              接收行数={followingImportStats.received_rows}, following=true 行数={followingImportStats.following_true_rows},
              新建={followingImportStats.created_kols_count}, 更新={followingImportStats.updated_kols_count},
              跳过={followingImportStats.skipped_count}
            </p>
            {followingImportStats.created_kols.length > 0 && (
              <p style={{ margin: "4px 0 0 0" }}>
                新建：{followingImportStats.created_kols.map((item) => `#${item.id}@${item.handle}`).join(", ")}
              </p>
            )}
            {followingImportStats.updated_kols.length > 0 && (
              <p style={{ margin: "4px 0 0 0" }}>
                更新：{followingImportStats.updated_kols.map((item) => `#${item.id}@${item.handle}`).join(", ")}
              </p>
            )}
            {followingImportStats.errors.length > 0 && (
              <details>
                <summary>
                  错误（{followingImportStats.errors.length}）- 仅显示前 {Math.min(20, followingImportStats.errors.length)}
                </summary>
                <ul>
                  {followingImportStats.errors.slice(0, 20).map((item) => (
                    <li key={`${item.row_index}-${item.reason}`}>
                      行={item.row_index}, 原因={item.reason}, 原文片段={item.raw_snippet}
                    </li>
                  ))}
                </ul>
              </details>
            )}
          </div>
        )}
      </section>

      <section style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "10px", maxWidth: "760px" }}>
        <h2 style={{ marginTop: 0 }}>抽取器状态</h2>
        {statusError && <p style={{ color: "crimson" }}>{statusError}</p>}
        {!statusError && !extractorStatus && <p>正在加载抽取器状态...</p>}
        {extractorStatus && (
          <div style={{ display: "grid", gap: "4px" }}>
            <p style={{ margin: 0 }}>
              抽取模式：<strong>{extractorStatus.mode}</strong> | 服务地址：{extractorStatus.base_url}
            </p>
            <p style={{ margin: 0 }}>
              默认模型：{extractorStatus.default_model} | 已配置 API Key：{extractorStatus.has_api_key ? "是" : "否"} |
              剩余额度：{extractorStatus.call_budget_remaining ?? "无限制"} | 最大输出令牌：{extractorStatus.max_output_tokens}
            </p>
          </div>
        )}
      </section>

    </main>
  );
}
