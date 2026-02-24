"use client";

import Link from "next/link";
import type { DragEvent, FormEvent } from "react";
import { useEffect, useMemo, useState } from "react";

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
  status: "queued" | "running" | "done" | "failed";
  mode: "pending_only" | "pending_or_failed" | "force";
  batch_size: number;
  batch_sleep_ms: number;
  last_error_summary: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
};

type ApiErrorBody = {
  request_id?: string;
  error_code?: string;
  message?: string;
  detail?: unknown;
};

type RuntimeSettings = {
  extractor_mode: string;
  provider_detected: string;
  extraction_output_mode: string;
  model: string;
  has_api_key: boolean;
  base_url: string;
  budget_remaining: number | null;
  budget_total: number;
  default_budget_total: number;
  call_budget_override_enabled: boolean;
  call_budget_override_value: number | null;
  window_start: string;
  window_end: string;
  max_output_tokens: number;
  auto_reject_confidence_threshold: number;
  throttle: {
    max_concurrency: number;
    max_rpm: number;
    batch_size: number;
    batch_sleep_ms: number;
  };
  burst: {
    enabled: boolean;
    mode: string | null;
    expires_at: string | null;
  };
  runtime_overrides: {
    call_budget: boolean;
    burst: boolean;
    throttle: boolean;
  };
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
  const [runtimeSettings, setRuntimeSettings] = useState<RuntimeSettings | null>(null);
  const [runtimeBudgetInput, setRuntimeBudgetInput] = useState("30");
  const [runtimeBudgetSaving, setRuntimeBudgetSaving] = useState(false);
  const [burstEnabledInput, setBurstEnabledInput] = useState(false);
  const [burstCallBudgetInput, setBurstCallBudgetInput] = useState("1000");
  const [burstDurationInput, setBurstDurationInput] = useState("30");
  const [burstSaving, setBurstSaving] = useState(false);
  const [throttleInput, setThrottleInput] = useState({
    max_concurrency: "2",
    max_rpm: "30",
    batch_size: "20",
    batch_sleep_ms: "250",
  });
  const [throttleSaving, setThrottleSaving] = useState(false);
  const [extractBatchSize, setExtractBatchSize] = useState(20);
  const [statusError, setStatusError] = useState<string | null>(null);

  const [kols, setKols] = useState<Kol[]>([]);
  const [kolsError, setKolsError] = useState<string | null>(null);
  const [assets, setAssets] = useState<AssetItem[]>([]);
  const [assetsError, setAssetsError] = useState<string | null>(null);
  const [selectedKolId, setSelectedKolId] = useState<number | null>(null);
  const [authorHandle, setAuthorHandle] = useState("");
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
  const [nowTs, setNowTs] = useState(() => Date.now());
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
    if (endDate) localStorage.setItem(STORAGE_END_DATE, endDate);
  }, [endDate]);

  useEffect(() => {
    const timer = window.setInterval(() => setNowTs(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    const loadStatus = async () => {
      try {
        const [extractorRes, runtimeRes] = await Promise.all([
          fetch("/api/extractor-status", { cache: "no-store" }),
          fetch("/api/settings/runtime", { cache: "no-store" }),
        ]);
        const extractorParsed = await parseApiResponse<ExtractorStatus>(extractorRes);
        if (!extractorRes.ok || !extractorParsed.data) {
          throw new Error(formatApiError("Load extractor status failed", extractorParsed.data, extractorParsed.textBody, extractorParsed.requestId));
        }
        setExtractorStatus(extractorParsed.data as ExtractorStatus);

        const runtimeParsed = await parseApiResponse<RuntimeSettings>(runtimeRes);
        if (!runtimeRes.ok || !runtimeParsed.data) {
          throw new Error(formatApiError("Load runtime settings failed", runtimeParsed.data, runtimeParsed.textBody, runtimeParsed.requestId));
        }
        const runtime = runtimeParsed.data as RuntimeSettings;
        setRuntimeSettings(runtime);
        setRuntimeBudgetInput(String(runtime.budget_total));
        setBurstEnabledInput(runtime.burst.enabled);
        setThrottleInput({
          max_concurrency: String(runtime.throttle.max_concurrency),
          max_rpm: String(runtime.throttle.max_rpm),
          batch_size: String(runtime.throttle.batch_size),
          batch_sleep_ms: String(runtime.throttle.batch_sleep_ms),
        });
        setExtractBatchSize([10, 20, 50].includes(runtime.throttle.batch_size) ? runtime.throttle.batch_size : 20);
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
        setKolsError(err instanceof Error ? err.message : "Load kols failed");
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

  const selectedKol = useMemo(() => kols.find((item) => item.id === selectedKolId) ?? null, [kols, selectedKolId]);

  const formatApiError = (fallbackMessage: string, body: unknown, textBody: string, requestId: string | null): string => {
    let message = fallbackMessage;
    if (body && typeof body === "object") {
      const obj = body as Record<string, unknown>;
      if (typeof obj.message === "string" && obj.message) message = obj.message;
      else if (typeof obj.detail === "string" && obj.detail) message = obj.detail;
    } else if (textBody) {
      message = `非 JSON 错误文本: ${textBody.slice(0, 300)}`;
    }
    return requestId ? `${message} (request_id=${requestId})` : message;
  };

  const parseApiResponse = async <T,>(res: Response): Promise<{ data: T | ApiErrorBody | null; textBody: string; requestId: string | null }> => {
    const requestId = res.headers.get("x-request-id");
    const textBody = await res.text();
    try {
      const parsed = JSON.parse(textBody) as T | ApiErrorBody;
      if (parsed && typeof parsed === "object") {
        const bodyRequestId = (parsed as ApiErrorBody).request_id;
        return { data: parsed, textBody: "", requestId: bodyRequestId ?? requestId };
      }
      return { data: parsed, textBody: "", requestId };
    } catch {
      return { data: null, textBody, requestId };
    }
  };

  const progressHandle = useMemo(() => {
    if (authorHandle.trim()) return authorHandle.trim();
    if (selectedKol) return selectedKol.handle;
    if (selectedHandles.length === 1) return selectedHandles[0];
    return "";
  }, [authorHandle, selectedHandles, selectedKol]);

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
    const effectiveAuthorOverride = authorHandle.trim().replace(/^@+/, "");
    const withOverrides = rows.map((item) => {
      const next: XImportItem = { ...item };
      if (effectiveAuthorOverride) next.author_handle = effectiveAuthorOverride;
      if (selectedKolId !== null) next.kol_id = selectedKolId;
      return next;
    });
    return withOverrides.filter((item) => {
      const key = toDateKeyFromIso(item.posted_at);
      if (!key) return false;
      if (startDate && key < startDate) return false;
      if (endDate && key > endDate) return false;
      return true;
    });
  };

  const convertViaApi = async (targetFile: File, forcedAuthorHandle?: string): Promise<XConvertResult> => {
    const params = new URLSearchParams();
    params.set("filename", targetFile.name);
    const effectiveAuthorHandle = (forcedAuthorHandle || authorHandle).trim();
    if (effectiveAuthorHandle) params.set("author_handle", effectiveAuthorHandle.replace(/^@+/, ""));
    if (selectedKolId !== null) params.set("kol_id", String(selectedKolId));
    if (startDate) params.set("start_date", startDate);
    if (endDate) params.set("end_date", endDate);
    params.set("include_raw_json", "true");
    params.set("only_followed", onlyFollowedKols ? "true" : "false");
    params.set("allow_unknown_handles", onlyFollowedKols ? "false" : "true");
    const res = await fetch(`/api/ingest/x/convert?${params.toString()}`, {
      method: "POST",
      headers: { "Content-Type": targetFile.type || "application/octet-stream" },
      body: targetFile,
    });
    const parsed = await parseApiResponse<XConvertResult>(res);
    if (!res.ok || !parsed.data || !("items" in (parsed.data as Record<string, unknown>))) {
      setWorkflowRequestId(parsed.requestId);
      throw new Error(formatApiError("Convert failed", parsed.data, parsed.textBody, parsed.requestId));
    }
    return parsed.data as XConvertResult;
  };

  const importRows = async (rows: XImportItem[], authorHandleOverride?: string): Promise<XImportStats> => {
    const params = new URLSearchParams();
    const normalizedOverride = (authorHandleOverride || "").trim().replace(/^@+/, "");
    if (normalizedOverride) params.set("author_handle_override", normalizedOverride);
    params.set("only_followed", onlyFollowedKols ? "true" : "false");
    params.set("allow_unknown_handles", onlyFollowedKols ? "false" : "true");
    const query = params.toString() ? `?${params.toString()}` : "";
    const res = await fetch(`/api/ingest/x/import${query}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(rows),
    });
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
  ): Promise<ExtractBatchStats> => {
    const stableIds = Array.from(new Set(ids)).sort((a, b) => a - b);
    const idempotencyKey = `ingest:${mode}:${Math.max(1, batchSize)}:${stableIds.join(",")}`.slice(0, 256);
    const createRes = await fetch("/api/extract-jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        raw_post_ids: stableIds,
        mode,
        batch_size: Math.max(1, batchSize),
        idempotency_key: idempotencyKey,
      }),
    });
    const createParsed = await parseApiResponse<ExtractJobCreateResponse>(createRes);
    setWorkflowRequestId(createParsed.requestId);
    if (!createRes.ok || !createParsed.data || !("job_id" in (createParsed.data as Record<string, unknown>))) {
      throw new Error(formatApiError("Create extract job failed", createParsed.data, createParsed.textBody, createParsed.requestId));
    }
    const jobId = (createParsed.data as ExtractJobCreateResponse).job_id;

    for (let attempt = 0; attempt < 1800; attempt += 1) {
      const res = await fetch(`/api/extract-jobs/${jobId}`, { cache: "no-store" });
      const parsed = await parseApiResponse<ExtractJob>(res);
      if (!res.ok || !parsed.data) {
        setWorkflowRequestId(parsed.requestId);
        throw new Error(formatApiError("Load extract job failed", parsed.data, parsed.textBody, parsed.requestId));
      }
      const job = parsed.data as ExtractJob;
      setWorkflowRequestId(parsed.requestId);
      setWorkflowStep(
        `Extract job ${job.status}: success=${job.success_count}, failed=${job.failed_count}, skipped=${job.skipped_count}, requested=${job.requested_count}`,
      );
      if (job.status === "done") return job;
      if (job.status === "failed") {
        const summary = job.last_error_summary ? `: ${job.last_error_summary}` : "";
        throw new Error(`Extract job failed${summary}${parsed.requestId ? ` (request_id=${parsed.requestId})` : ""}`);
      }
      await new Promise((resolve) => window.setTimeout(resolve, 1200));
    }
    throw new Error("Extract job polling timeout");
  };

  const updateRuntimeCallBudget = async () => {
    const next = Number(runtimeBudgetInput);
    if (!Number.isFinite(next) || next < 0) {
      setStatusError("call budget 必须是 >= 0 的数字");
      return;
    }
    setRuntimeBudgetSaving(true);
    setStatusError(null);
    try {
      const res = await fetch("/api/settings/runtime/call-budget", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ call_budget: Math.floor(next) }),
      });
      const body = (await res.json()) as RuntimeSettings | { detail?: string };
      if (!res.ok) {
        throw new Error("detail" in body ? (body.detail ?? "Update runtime budget failed") : "Update failed");
      }
      const updated = body as RuntimeSettings;
      setRuntimeSettings(updated);
      setRuntimeBudgetInput(String(updated.budget_total));
    } catch (err) {
      setStatusError(err instanceof Error ? err.message : "Update runtime budget failed");
    } finally {
      setRuntimeBudgetSaving(false);
    }
  };

  const clearRuntimeCallBudgetOverride = async () => {
    setRuntimeBudgetSaving(true);
    setStatusError(null);
    try {
      const res = await fetch("/api/settings/runtime/call-budget/clear", {
        method: "POST",
      });
      const body = (await res.json()) as RuntimeSettings | { detail?: string };
      if (!res.ok) {
        throw new Error("detail" in body ? (body.detail ?? "Clear runtime budget override failed") : "Clear failed");
      }
      const updated = body as RuntimeSettings;
      setRuntimeSettings(updated);
      setRuntimeBudgetInput(String(updated.default_budget_total));
    } catch (err) {
      setStatusError(err instanceof Error ? err.message : "Clear runtime budget override failed");
    } finally {
      setRuntimeBudgetSaving(false);
    }
  };

  const updateRuntimeBurst = async () => {
    const callBudget = Math.floor(Number(burstCallBudgetInput));
    const durationMinutes = Math.floor(Number(burstDurationInput));
    if (!burstEnabledInput) {
      setBurstSaving(true);
      setStatusError(null);
      try {
        const res = await fetch("/api/settings/runtime/burst", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ enabled: false, mode: "normal", call_budget: 0, duration_minutes: 1 }),
        });
        const body = (await res.json()) as RuntimeSettings | { detail?: string };
        if (!res.ok) throw new Error("detail" in body ? (body.detail ?? "Disable burst failed") : "Disable burst failed");
        setRuntimeSettings(body as RuntimeSettings);
      } catch (err) {
        setStatusError(err instanceof Error ? err.message : "Disable burst failed");
      } finally {
        setBurstSaving(false);
      }
      return;
    }

    if (!Number.isFinite(callBudget) || callBudget < 0) {
      setStatusError("burst call_budget 必须是 >= 0 的数字");
      return;
    }
    if (!Number.isFinite(durationMinutes) || durationMinutes < 1) {
      setStatusError("burst duration_minutes 必须是 >= 1 的整数");
      return;
    }
    if (durationMinutes > 120) {
      setStatusError("burst duration_minutes 上限是 120");
      return;
    }
    setBurstSaving(true);
    setStatusError(null);
    try {
      const res = await fetch("/api/settings/runtime/burst", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          enabled: true,
          mode: "normal",
          call_budget: callBudget,
          duration_minutes: durationMinutes,
        }),
      });
      const body = (await res.json()) as RuntimeSettings | { detail?: string };
      if (!res.ok) throw new Error("detail" in body ? (body.detail ?? "Enable burst failed") : "Enable burst failed");
      setRuntimeSettings(body as RuntimeSettings);
    } catch (err) {
      setStatusError(err instanceof Error ? err.message : "Enable burst failed");
    } finally {
      setBurstSaving(false);
    }
  };

  const enableUnlimitedSafeBurst = async () => {
    const durationMinutes = Math.floor(Number(burstDurationInput));
    if (!Number.isFinite(durationMinutes) || durationMinutes < 1 || durationMinutes > 120) {
      setStatusError("Unlimited (safe) 需要 1-120 分钟 duration");
      return;
    }
    setBurstSaving(true);
    setStatusError(null);
    try {
      const res = await fetch("/api/settings/runtime/burst", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          enabled: true,
          mode: "unlimited_safe",
          call_budget: 100000,
          duration_minutes: durationMinutes,
        }),
      });
      const body = (await res.json()) as RuntimeSettings | { detail?: string };
      if (!res.ok) throw new Error("detail" in body ? (body.detail ?? "Enable unlimited safe burst failed") : "Enable failed");
      const updated = body as RuntimeSettings;
      setRuntimeSettings(updated);
      setBurstEnabledInput(updated.burst.enabled);
      setBurstCallBudgetInput(String(updated.budget_total));
    } catch (err) {
      setStatusError(err instanceof Error ? err.message : "Enable unlimited safe burst failed");
    } finally {
      setBurstSaving(false);
    }
  };

  const updateRuntimeThrottle = async () => {
    const payload = {
      max_concurrency: Math.floor(Number(throttleInput.max_concurrency)),
      max_rpm: Math.floor(Number(throttleInput.max_rpm)),
      batch_size: Math.floor(Number(throttleInput.batch_size)),
      batch_sleep_ms: Math.floor(Number(throttleInput.batch_sleep_ms)),
    };
    if (Object.values(payload).some((value) => !Number.isFinite(value))) {
      setStatusError("throttle 参数必须都是数字");
      return;
    }
    setThrottleSaving(true);
    setStatusError(null);
    try {
      const res = await fetch("/api/settings/runtime/throttle", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const body = (await res.json()) as RuntimeSettings | { detail?: string };
      if (!res.ok) throw new Error("detail" in body ? (body.detail ?? "Update throttle failed") : "Update throttle failed");
      const updated = body as RuntimeSettings;
      setRuntimeSettings(updated);
      setThrottleInput({
        max_concurrency: String(updated.throttle.max_concurrency),
        max_rpm: String(updated.throttle.max_rpm),
        batch_size: String(updated.throttle.batch_size),
        batch_sleep_ms: String(updated.throttle.batch_sleep_ms),
      });
      if ([10, 20, 50].includes(updated.throttle.batch_size)) {
        setExtractBatchSize(updated.throttle.batch_size);
      }
    } catch (err) {
      setStatusError(err instanceof Error ? err.message : "Update throttle failed");
    } finally {
      setThrottleSaving(false);
    }
  };

  const windowRemainingText = useMemo(() => {
    if (!runtimeSettings) return "-";
    const end = new Date(runtimeSettings.window_end).getTime();
    if (Number.isNaN(end)) return "-";
    const diffMs = Math.max(0, end - nowTs);
    const totalSeconds = Math.floor(diffMs / 1000);
    const mins = Math.floor(totalSeconds / 60);
    const secs = totalSeconds % 60;
    return `${mins}m ${secs}s`;
  }, [runtimeSettings, nowTs]);

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
      const effectiveAuthorOverride = authorHandle.trim().replace(/^@+/, "");
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
            const converted = await convertViaApi(file, effectiveAuthorOverride);
            setConvertResult(converted);
            detectedHandles = converted.handles_summary.map((item) => item.author_handle);
            syncSelectedHandles(detectedHandles);
            rows = converted.items;
            convertedRowsTotal = converted.converted_rows;
          }
        } catch {
          setWorkflowStep("JSON parse failed in browser, converting via backend...");
          const converted = await convertViaApi(file, effectiveAuthorOverride);
          setConvertResult(converted);
          detectedHandles = converted.handles_summary.map((item) => item.author_handle);
          syncSelectedHandles(detectedHandles);
          rows = converted.items;
          convertedRowsTotal = converted.converted_rows;
        }
      } else {
        setWorkflowStep("Converting upload via backend...");
        const converted = await convertViaApi(file, effectiveAuthorOverride);
        setConvertResult(converted);
        detectedHandles = converted.handles_summary.map((item) => item.author_handle);
        syncSelectedHandles(detectedHandles);
        rows = converted.items;
        convertedRowsTotal = converted.converted_rows;
      }

      const activeHandleSet =
        selectedKolId !== null || effectiveAuthorOverride
          ? null
          : new Set(
              (selectedHandles.length > 0
                ? selectedHandles
                : detectedHandles
              ).map((item) => item.trim().toLowerCase()),
            );
      if (activeHandleSet && activeHandleSet.size > 0) {
        rows = rows.filter((item) =>
          activeHandleSet.has((item.resolved_author_handle || item.author_handle).trim().toLowerCase()),
        );
      }

      setConvertedCount(convertedRowsTotal || rows.length);
      if (rows.length === 0) throw new Error("No rows left after conversion/filtering.");

      setWorkflowStep("Importing into raw_posts...");
      const imported = await importRows(rows, effectiveAuthorOverride);
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
          skipped_not_followed_count: 0,
          failed_count: 0,
          auto_rejected_count: 0,
          resumed_requested_count: 0,
          resumed_success: 0,
          resumed_failed: 0,
          resumed_skipped: 0,
        };
        const perHandle: Record<string, ExtractBatchStats> = {};
        for (const [handle, byHandle] of Object.entries(imported.imported_by_handle)) {
          const ids = byHandle.raw_post_ids ?? [];
          if (ids.length === 0) continue;
          const extracted = await runExtractJob(ids, extractBatchSize, "pending_or_failed");
          perHandle[handle] = extracted;
          totals.requested_count += extracted.requested_count;
          totals.success_count += extracted.success_count;
          totals.skipped_count += extracted.skipped_count;
          totals.skipped_already_extracted_count += extracted.skipped_already_extracted_count;
          totals.skipped_already_pending_count += extracted.skipped_already_pending_count;
          totals.skipped_already_success_count += extracted.skipped_already_success_count;
          totals.skipped_already_has_result_count += extracted.skipped_already_has_result_count;
          totals.skipped_already_rejected_count += extracted.skipped_already_rejected_count;
          totals.skipped_already_approved_count += extracted.skipped_already_approved_count;
          totals.skipped_not_followed_count += extracted.skipped_not_followed_count;
          totals.failed_count += extracted.failed_count;
          totals.auto_rejected_count += extracted.auto_rejected_count;
          totals.resumed_requested_count += extracted.resumed_requested_count;
          totals.resumed_success += extracted.resumed_success;
          totals.resumed_failed += extracted.resumed_failed;
          totals.resumed_skipped += extracted.resumed_skipped;
        }
        setExtractByHandle(perHandle);
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
          <label>
            KOL (recommended)
            <select
              value={selectedKolId ?? ""}
              onChange={(event) => {
                const next = event.target.value ? Number(event.target.value) : null;
                setSelectedKolId(next);
                const kol = kols.find((item) => item.id === next);
                if (kol) setAuthorHandle(kol.handle);
              }}
              style={{ display: "block", width: "100%" }}
            >
              <option value="">(none)</option>
              {kols.map((kol) => (
                <option key={kol.id} value={kol.id}>
                  #{kol.id} {kol.display_name ?? kol.handle} (@{kol.handle})
                </option>
              ))}
            </select>
          </label>
          {kolsError && <p style={{ color: "crimson", margin: 0 }}>{kolsError}</p>}

          <label>
            author_handle 覆盖（可选）
            <input value={authorHandle} onChange={(event) => setAuthorHandle(event.target.value)} style={{ display: "block", width: "100%" }} />
          </label>
          {convertResult && convertResult.handles_summary.length > 1 && selectedKolId === null && !authorHandle.trim() && (
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
            Refresh Progress
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
        {convertResult && selectedKolId === null && convertResult.resolved_author_handle && (
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
              {extractStats.skipped_already_approved_count}, skipped_not_followed={extractStats.skipped_not_followed_count}
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
              Refresh progress
            </button>
            <Link href={`/extractions?status=pending&t=${Date.now()}`}>Refresh extractions</Link>
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
        <h2 style={{ marginTop: 0 }}>Runtime Call Budget</h2>
        {runtimeSettings ? (
          <div style={{ display: "grid", gap: "8px" }}>
            <p style={{ margin: 0 }}>
              Current effective budget_total={runtimeSettings.budget_total}, budget_remaining=
              {runtimeSettings.budget_remaining ?? "unlimited"}
            </p>
            <p style={{ margin: 0 }}>
              Default (env/settings)={runtimeSettings.default_budget_total} (expected 30/hour reset)
            </p>
            {runtimeSettings.call_budget_override_enabled && (
              <p style={{ margin: 0, color: "crimson", fontWeight: 700 }}>
                Override active: {runtimeSettings.call_budget_override_value} (click Clear to restore default{" "}
                {runtimeSettings.default_budget_total})
              </p>
            )}
            <p style={{ margin: 0 }}>
              window_start: {runtimeSettings.window_start} | window_end: {runtimeSettings.window_end} (remaining {windowRemainingText})
            </p>
            <p style={{ margin: 0 }}>
              provider_detected: {runtimeSettings.provider_detected} | extraction_output_mode: {runtimeSettings.extraction_output_mode}
            </p>
            <p style={{ margin: 0 }}>
              burst: {runtimeSettings.burst.enabled ? "enabled" : "disabled"} | mode: {runtimeSettings.burst.mode ?? "-"} | expires_at: {runtimeSettings.burst.expires_at ?? "-"}
            </p>
            <p style={{ margin: 0, color: "#8a5800" }}>
              Update Runtime Budget 是进程内临时覆盖，重启后会恢复到 env/default。
            </p>
            <div style={{ display: "flex", gap: "8px", alignItems: "center", flexWrap: "wrap" }}>
              <input
                type="number"
                min={0}
                step={1}
                value={runtimeBudgetInput}
                onChange={(event) => setRuntimeBudgetInput(event.target.value)}
                style={{ width: "120px" }}
              />
              <button type="button" onClick={() => void updateRuntimeCallBudget()} disabled={runtimeBudgetSaving}>
                {runtimeBudgetSaving ? "Saving..." : "Update Runtime Budget"}
              </button>
              <button
                type="button"
                onClick={() => void clearRuntimeCallBudgetOverride()}
                disabled={runtimeBudgetSaving || !runtimeSettings.call_budget_override_enabled}
              >
                {runtimeBudgetSaving ? "Saving..." : "Clear Override"}
              </button>
              <span>override: {runtimeSettings.call_budget_override_enabled ? "yes" : "no"}</span>
            </div>
            <div style={{ display: "flex", gap: "8px", alignItems: "center", flexWrap: "wrap" }}>
              <label>
                <input
                  type="checkbox"
                  checked={burstEnabledInput}
                  onChange={(event) => setBurstEnabledInput(event.target.checked)}
                />{" "}
                Enable Burst
              </label>
              <input
                type="number"
                min={0}
                step={1}
                value={burstCallBudgetInput}
                onChange={(event) => setBurstCallBudgetInput(event.target.value)}
                style={{ width: "120px" }}
              />
              <input
                type="number"
                min={1}
                step={1}
                value={burstDurationInput}
                onChange={(event) => setBurstDurationInput(event.target.value)}
                style={{ width: "120px" }}
              />
              <button type="button" onClick={() => void updateRuntimeBurst()} disabled={burstSaving}>
                {burstSaving ? "Saving..." : burstEnabledInput ? "Enable Burst" : "Disable Burst"}
              </button>
              <button type="button" onClick={() => void enableUnlimitedSafeBurst()} disabled={burstSaving}>
                {burstSaving ? "Saving..." : "Unlimited (safe)"}
              </button>
            </div>
          </div>
        ) : (
          <p style={{ margin: 0 }}>Loading runtime settings...</p>
        )}
      </section>

      <section style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "10px", maxWidth: "760px" }}>
        <h2 style={{ marginTop: 0 }}>Runtime Throttle</h2>
        {runtimeSettings ? (
          <div style={{ display: "grid", gap: "8px" }}>
            <p style={{ margin: 0 }}>
              max_rpm={runtimeSettings.throttle.max_rpm}, max_concurrency={runtimeSettings.throttle.max_concurrency},
              batch_size={runtimeSettings.throttle.batch_size}, batch_sleep_ms={runtimeSettings.throttle.batch_sleep_ms}
            </p>
            <p style={{ margin: 0, color: "#8a5800" }}>
              默认是付费模型的保守配置；若 429 增多，建议降低 rpm/并发或增大 sleep。
            </p>
            <div style={{ display: "flex", gap: "8px", alignItems: "center", flexWrap: "wrap" }}>
              <input
                type="number"
                min={1}
                value={throttleInput.max_rpm}
                onChange={(event) => setThrottleInput((prev) => ({ ...prev, max_rpm: event.target.value }))}
                style={{ width: "100px" }}
              />
              <input
                type="number"
                min={1}
                value={throttleInput.max_concurrency}
                onChange={(event) => setThrottleInput((prev) => ({ ...prev, max_concurrency: event.target.value }))}
                style={{ width: "100px" }}
              />
              <input
                type="number"
                min={1}
                value={throttleInput.batch_size}
                onChange={(event) => setThrottleInput((prev) => ({ ...prev, batch_size: event.target.value }))}
                style={{ width: "100px" }}
              />
              <input
                type="number"
                min={100}
                value={throttleInput.batch_sleep_ms}
                onChange={(event) => setThrottleInput((prev) => ({ ...prev, batch_sleep_ms: event.target.value }))}
                style={{ width: "120px" }}
              />
              <button type="button" onClick={() => void updateRuntimeThrottle()} disabled={throttleSaving}>
                {throttleSaving ? "Saving..." : "Update Throttle"}
              </button>
            </div>
          </div>
        ) : (
          <p style={{ margin: 0 }}>Loading throttle settings...</p>
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
