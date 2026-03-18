"use client";

import { createContext, useContext, useEffect, useMemo, useRef, useState } from "react";
import { getAiUploadCount } from "../ingest/import-stats.js";

type ProgressPhase = "idle" | "confirm" | "converting" | "importing" | "extracting" | "refreshing" | "done";
type ProgressStatus = "idle" | "running" | "success" | "error" | "pending_confirm";

type ImportProgressState = {
  visible: boolean;
  minimized: boolean;
  status: ProgressStatus;
  phase: ProgressPhase;
  message: string | null;
  error: string | null;
  extractJobId: string | null;
  extractRequested: number;
  extractUploaded: number;
  extractSuccess: number;
  extractFailed: number;
  extractSkipped: number;
  startedAt: number | null;
  finishedAt: number | null;
  noticePending: boolean;
  lastUpdatedAt: number | null;
};

type ImportProgressContextValue = {
  state: ImportProgressState;
  begin: (message: string) => void;
  markPendingConfirm: (message: string) => void;
  updateRunning: (message: string, phase?: ProgressPhase) => void;
  attachExtractJob: (jobId: string) => void;
  applyExtractStats: (stats: { requested: number; uploaded: number; success: number; failed: number; skipped: number }) => void;
  markSuccess: (message: string) => void;
  markError: (message: string) => void;
  toggleMinimized: () => void;
  closePanel: () => void;
  clearDone: () => void;
};

type ExtractJobPayload = {
  status?: string;
  is_terminal?: boolean;
  requested_count?: number;
  success_count?: number;
  failed_count?: number;
  skipped_count?: number;
  ai_call_used?: number;
  openai_call_attempted_count?: number;
  last_error_summary?: string | null;
};

const STORAGE_KEY = "ip_import_progress_v1";
const DEFAULT_DIRECT_API_BASE_URL = "http://localhost:8000";

const initialState: ImportProgressState = {
  visible: false,
  minimized: true,
  status: "idle",
  phase: "idle",
  message: null,
  error: null,
  extractJobId: null,
  extractRequested: 0,
  extractUploaded: 0,
  extractSuccess: 0,
  extractFailed: 0,
  extractSkipped: 0,
  startedAt: null,
  finishedAt: null,
  noticePending: false,
  lastUpdatedAt: null,
};

const ImportProgressContext = createContext<ImportProgressContextValue | null>(null);

function getDirectApiBaseUrl(): string {
  const raw = process.env.NEXT_PUBLIC_API_BASE_URL?.trim();
  const base = raw && raw.length > 0 ? raw : DEFAULT_DIRECT_API_BASE_URL;
  return base.replace(/\/+$/, "");
}

function mergeState(prev: ImportProgressState, partial: Partial<ImportProgressState>): ImportProgressState {
  return { ...prev, ...partial, lastUpdatedAt: Date.now() };
}

function toNumber(input: unknown): number {
  return typeof input === "number" && Number.isFinite(input) ? input : 0;
}

export function ImportProgressProvider({ children }: { children: React.ReactNode }) {
  const [state, setState] = useState<ImportProgressState>(initialState);
  const [hydrated, setHydrated] = useState(false);
  const mountedRef = useRef(false);

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(STORAGE_KEY);
      if (raw) {
        const parsed = JSON.parse(raw) as Partial<ImportProgressState>;
        setState((prev) => ({
          ...prev,
          ...parsed,
          status: (parsed.status as ProgressStatus) ?? prev.status,
          phase: (parsed.phase as ProgressPhase) ?? prev.phase,
        }));
      }
    } catch {
      // ignore corrupted local cache
    } finally {
      setHydrated(true);
      mountedRef.current = true;
    }
  }, []);

  useEffect(() => {
    if (!hydrated || !mountedRef.current) return;
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  }, [hydrated, state]);

  useEffect(() => {
    if (!hydrated || state.status !== "running" || !state.extractJobId) return;
    let cancelled = false;

    const poll = async () => {
      try {
        const res = await fetch(`${getDirectApiBaseUrl()}/extract-jobs/${encodeURIComponent(state.extractJobId ?? "")}`, {
          cache: "no-store",
        });
        if (!res.ok) return;
        const job = (await res.json()) as ExtractJobPayload;
        if (cancelled) return;
        const requested = toNumber(job.requested_count);
        const success = toNumber(job.success_count);
        const failed = toNumber(job.failed_count);
        const skipped = toNumber(job.skipped_count);
        const uploaded = getAiUploadCount(job);
        const terminal = Boolean(job.is_terminal);
        const status = typeof job.status === "string" ? job.status : "";
        const isFailed = status === "failed" || status === "cancelled" || status === "timeout";
        setState((prev) =>
          mergeState(prev, {
            phase: "extracting",
            message: `AI处理中：成功=${success}，失败=${failed}${requested > 0 ? `，完成=${success + failed + skipped}/${requested}` : ""}`,
            extractRequested: requested,
            extractUploaded: uploaded,
            extractSuccess: success,
            extractFailed: failed,
            extractSkipped: skipped,
            status: isFailed ? "error" : terminal ? "success" : "running",
            finishedAt: terminal ? Date.now() : null,
            error: isFailed ? job.last_error_summary ?? "抽取任务失败" : null,
            noticePending: !isFailed && terminal ? true : prev.noticePending,
          }),
        );
      } catch {
        // keep previous state; user can retry later
      }
    };

    void poll();
    const timer = window.setInterval(() => {
      void poll();
    }, 1200);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [hydrated, state.status, state.extractJobId]);

  const value = useMemo<ImportProgressContextValue>(
    () => ({
      state,
      begin: (message) => {
        setState(
          mergeState(initialState, {
            visible: true,
            minimized: true,
            status: "running",
            phase: "converting",
            message,
            startedAt: Date.now(),
          }),
        );
      },
      markPendingConfirm: (message) => {
        setState((prev) =>
          mergeState(prev, {
            visible: true,
            minimized: true,
            status: "pending_confirm",
            phase: "confirm",
            message,
            error: null,
          }),
        );
      },
      updateRunning: (message, phase = "importing") => {
        setState((prev) =>
          mergeState(prev, {
            visible: true,
            status: "running",
            phase,
            message,
            error: null,
          }),
        );
      },
      attachExtractJob: (jobId) => {
        setState((prev) =>
          mergeState(prev, {
            visible: true,
            status: "running",
            phase: "extracting",
            extractJobId: jobId,
            message: prev.message ?? "AI处理中...",
          }),
        );
      },
      applyExtractStats: ({ requested, uploaded, success, failed, skipped }) => {
        setState((prev) =>
          mergeState(prev, {
            extractRequested: requested,
            extractUploaded: uploaded,
            extractSuccess: success,
            extractFailed: failed,
            extractSkipped: skipped,
          }),
        );
      },
      markSuccess: (message) => {
        setState((prev) =>
          mergeState(prev, {
            visible: true,
            status: "success",
            phase: "done",
            message,
            error: null,
            finishedAt: Date.now(),
            noticePending: true,
          }),
        );
      },
      markError: (message) => {
        setState((prev) =>
          mergeState(prev, {
            visible: true,
            status: "error",
            message,
            error: message,
            finishedAt: Date.now(),
          }),
        );
      },
      toggleMinimized: () => {
        setState((prev) => mergeState(prev, { minimized: !prev.minimized }));
      },
      closePanel: () => {
        setState((prev) => mergeState(prev, { visible: false, noticePending: false }));
      },
      clearDone: () => {
        setState(initialState);
      },
    }),
    [state],
  );

  const totalHandled = state.extractSuccess + state.extractFailed + state.extractSkipped;
  const percent = state.extractRequested > 0 ? Math.min(100, Math.round((totalHandled / state.extractRequested) * 100)) : null;
  const uploadCount = state.extractUploaded;
  const compactPercent = percent ?? (state.status === "running" ? 35 : state.status === "success" ? 100 : 0);
  const compactTitle = state.status === "success" ? "导入完成" : state.status === "error" ? "导入失败" : "导入进度";

  return (
    <ImportProgressContext.Provider value={value}>
      {children}
      {state.visible && (
        <aside className={`import-progress-fab ${state.minimized ? "min" : ""}`} aria-live="polite">
          {state.minimized ? (
            <button
              type="button"
              className="import-progress-compact"
              onClick={() => value.toggleMinimized()}
              aria-label="展开导入进度"
            >
              <span className="import-progress-compact-title">{compactTitle}</span>
              <span className={`import-progress-compact-track ${percent === null && state.status === "running" ? "indeterminate" : ""}`}>
                <span className="import-progress-compact-fill" style={{ width: `${compactPercent}%` }} />
              </span>
            </button>
          ) : (
            <>
              <div className="import-progress-head">
                <strong>{compactTitle}</strong>
                <div className="import-progress-actions">
                  <button type="button" onClick={() => value.toggleMinimized()}>收起</button>
                  {state.status !== "running" && (
                    <button type="button" onClick={() => value.closePanel()}>
                      关闭
                    </button>
                  )}
                </div>
              </div>
            <div className="import-progress-body">
              <p>{state.message ?? "正在处理..."}</p>
              {(state.phase === "extracting" || state.extractRequested > 0 || uploadCount > 0) && (
                <>
                  <p>上传到AI模型={uploadCount}，成功={state.extractSuccess}，失败={state.extractFailed}</p>
                  {percent !== null && (
                    <>
                      <div className="import-progress-track">
                        <div className="import-progress-fill" style={{ width: `${percent}%` }} />
                      </div>
                      <p>当前进度={percent}%（{totalHandled}/{state.extractRequested}）</p>
                    </>
                  )}
                </>
              )}
              {state.noticePending && state.status === "success" && <p className="ok">导入与AI处理已完成，可关闭窗口。</p>}
              {state.status === "error" && state.error && <p className="err">{state.error}</p>}
            </div>
            </>
          )}
        </aside>
      )}
    </ImportProgressContext.Provider>
  );
}

export function useImportProgressHub(): ImportProgressContextValue {
  const ctx = useContext(ImportProgressContext);
  if (!ctx) throw new Error("useImportProgressHub must be used within ImportProgressProvider");
  return ctx;
}
