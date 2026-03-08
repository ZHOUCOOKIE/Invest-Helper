"use client";

import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { pickAssetViews } from "./review-inference.js";

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

type Extraction = {
  id: number;
  raw_post_id: number;
  status: "pending" | "approved" | "rejected";
  extracted_json: Record<string, unknown>;
  model_name: string;
  extractor_name: string;
  prompt_version: string | null;
  prompt_text: string | null;
  prompt_hash: string | null;
  raw_model_output: string | null;
  parsed_model_output: Record<string, unknown> | null;
  model_latency_ms: number | null;
  model_input_tokens: number | null;
  model_output_tokens: number | null;
  last_error: string | null;
  reviewed_at: string | null;
  reviewed_by: string | null;
  review_note: string | null;
  applied_kol_view_id: number | null;
  auto_applied_count: number | null;
  auto_policy: "threshold" | "top1_fallback" | null;
  auto_applied_kol_view_ids: number[] | null;
  auto_approve_confidence_threshold: number | null;
  auto_reject_confidence_threshold: number | null;
  approve_inserted_count: number | null;
  approve_skipped_count: number | null;
  auto_applied_asset_view_keys: string[] | null;
  auto_applied_views:
    | {
        kol_view_id: number;
        symbol: string;
        asset_id: number;
        stance: "bull" | "bear" | "neutral";
        horizon: "intraday" | "1w" | "1m" | "3m" | "1y";
        as_of: string;
        confidence: number;
      }[]
    | null;
  created_at: string;
  raw_post: RawPost;
};

type AssetViewItem = {
  symbol: string;
  stance: "bull" | "bear" | "neutral";
  horizon: "intraday" | "1w" | "1m" | "3m" | "1y";
  confidence: number;
  summary: string | null;
  as_of: string | null;
};

type ExtractorStatus = {
  mode: "auto" | "dummy" | "openai" | string;
  has_api_key: boolean;
  default_model: string;
  base_url: string;
  call_budget_remaining: number | null;
  max_output_tokens: number;
};

function todayIsoDate(): string {
  return new Date().toISOString().slice(0, 10);
}

function normalizeAsOfDate(value: string | null | undefined, fallback: string): string {
  if (!value || !value.trim()) return fallback;
  return value.slice(0, 10);
}

function buildAssetViewKey(item: AssetViewItem, fallbackAsOf: string): string {
  return `${item.symbol}|${item.horizon}|${normalizeAsOfDate(item.as_of, fallbackAsOf)}`;
}

function getHttpErrorMessage(status: number, detail: string | undefined, fallback: string): string {
  if (status === 422) return detail ?? "参数不合法（422），请检查输入内容。";
  if (status === 409) return detail ?? "状态冲突（409），该 extraction 可能已被处理。";
  if (status === 429) return detail ?? "请求过于频繁（429），请稍后再试。";
  return detail ?? fallback;
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

export default function ExtractionDetailPage() {
  const params = useParams<{ id: string }>();
  const router = useRouter();
  const extractionId = useMemo(() => Number(params.id), [params.id]);

  const [extraction, setExtraction] = useState<Extraction | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [errorStatus, setErrorStatus] = useState<number | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [actionInfo, setActionInfo] = useState<string | null>(null);
  const [reExtracting, setReExtracting] = useState(false);
  const [extractorStatus, setExtractorStatus] = useState<ExtractorStatus | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [copyMessage, setCopyMessage] = useState<string | null>(null);

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      setError(null);
      setErrorStatus(null);
      setActionError(null);
      setStatusError(null);

      if (Number.isNaN(extractionId)) {
        setError("无效的 extraction id");
        setLoading(false);
        return;
      }

      try {
        const [extractionRes, statusRes] = await Promise.all([
          fetch(`/api/extractions/${extractionId}`, { cache: "no-store" }),
          fetch("/api/extractor-status", { cache: "no-store" }),
        ]);

        const extractionBody = (await extractionRes.json()) as Extraction | { detail?: string };
        if (!extractionRes.ok) {
          setErrorStatus(extractionRes.status);
          throw new Error(
            getHttpErrorMessage(
              extractionRes.status,
              "detail" in extractionBody ? extractionBody.detail : undefined,
              "加载抽取记录失败",
            ),
          );
        }

        const statusBody = (await statusRes.json()) as ExtractorStatus | { detail?: string };
        if (!statusRes.ok) {
          setStatusError("detail" in statusBody ? (statusBody.detail ?? "加载抽取器状态失败") : "加载失败");
        } else {
          setExtractorStatus(statusBody as ExtractorStatus);
        }

        const extractionData = extractionBody as Extraction;
        setExtraction(extractionData);
      } catch (err) {
        setError(err instanceof Error ? err.message : "未知错误");
      } finally {
        setLoading(false);
      }
    };

    void load();
  }, [extractionId]);

  const reExtract = async () => {
    if (!extraction) {
      return;
    }
    if (!window.confirm("确认强制重新提取？这会创建新的 extraction 版本并消耗预算（不会覆盖历史）。")) {
      return;
    }
    setActionError(null);
    setActionInfo(null);
    setReExtracting(true);
    try {
      const res = await fetch(`/api/extractions/${extraction.id}/re-extract`, { method: "POST" });
      const body = (await res.json()) as { id?: number; detail?: string };
      if (!res.ok || typeof body.id !== "number") {
        throw new Error(getHttpErrorMessage(res.status, body.detail, `重新抽取失败: ${res.status}`));
      }
      setActionInfo(`已创建新 extraction #${body.id}，正在跳转...`);
      router.push(`/extractions/${body.id}`);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "重新抽取失败");
    } finally {
      setReExtracting(false);
    }
  };

  const budgetExhaustedFallback = useMemo(() => {
    if (!extraction) return false;
    if (extraction.last_error?.includes("budget_exhausted")) return true;
    const meta = extraction.extracted_json["meta"];
    if (!meta || typeof meta !== "object") return false;
    return (meta as Record<string, unknown>)["fallback_reason"] === "budget_exhausted";
  }, [extraction]);

  const reviewFailure = useMemo(() => {
    if (!extraction) return null;
    const rawMeta = extraction.extracted_json["meta"];
    const meta = rawMeta && typeof rawMeta === "object" ? (rawMeta as Record<string, unknown>) : null;
    const hasAssetViews = pickAssetViews(extraction.extracted_json).length > 0;
    const hasview = extraction.extracted_json["hasview"];
    if (meta?.["auto_rejected"] === true) {
      const code = String(meta["auto_review_reason"] ?? "-");
      const threshold = String(meta["auto_review_threshold"] ?? extraction.auto_reject_confidence_threshold ?? "-");
      const modelConfidence = String(meta["model_confidence"] ?? "-");
      if (code === "hasview_zero") {
        return "不通过：未识别到可审核资产观点（hasview=0 或 asset_views 为空）。";
      }
      if (code === "confidence_below_threshold") {
        return `不通过：模型置信度不足（${modelConfidence} < ${threshold}）。`;
      }
      return `不通过：${code}。`;
    }
    if (hasview === 0 || !hasAssetViews) {
      return "不通过：未识别到可审核资产观点（hasview=0 或 asset_views 为空）。";
    }
    return null;
  }, [extraction]);

  const extractedAssetViews = useMemo(() => {
    if (!extraction) return [];
    return pickAssetViews(extraction.extracted_json);
  }, [extraction]);

  const extractedAsOf = useMemo(() => {
    if (!extraction) return todayIsoDate();
    const rawAsOf = extraction.extracted_json["as_of"];
    return normalizeAsOfDate(
      typeof rawAsOf === "string" ? rawAsOf : null,
      extraction.raw_post.posted_at.slice(0, 10),
    );
  }, [extraction]);

  const autoAppliedKeySet = useMemo(() => {
    if (!extraction) return new Set<string>();
    const fallbackAsOf = extractedAsOf;
    const keys = (extraction.auto_applied_views || []).map(
      (item) => `${item.symbol.toUpperCase()}|${item.horizon}|${normalizeAsOfDate(item.as_of, fallbackAsOf)}`,
    );
    return new Set(keys);
  }, [extractedAsOf, extraction]);

  const copyRawOutput = async () => {
    if (!extraction?.raw_model_output) {
      return;
    }
    try {
      await navigator.clipboard.writeText(extraction.raw_model_output);
      setCopyMessage("已复制原始输出");
      window.setTimeout(() => setCopyMessage(null), 1200);
    } catch {
      setCopyMessage("复制失败");
      window.setTimeout(() => setCopyMessage(null), 1200);
    }
  };

  return (
    <main style={{ padding: "24px", fontFamily: "monospace" }}>
      <h1>抽取详情 #{Number.isNaN(extractionId) ? "?" : extractionId}</h1>
      <p>
        <Link href="/extractions">返回审核列表</Link>
      </p>

      {loading && <p>加载中...</p>}
      {error && (
        <section style={{ border: "1px solid #f0bcbc", background: "#fff6f6", borderRadius: "8px", padding: "10px" }}>
          <p style={{ color: "crimson", margin: 0 }}>{error}</p>
          {errorStatus === 422 && <p style={{ marginBottom: 0 }}>请检查抽取 ID 或请求参数。</p>}
          {errorStatus === 409 && <p style={{ marginBottom: 0 }}>资源状态冲突，请返回列表刷新后重试。</p>}
          {errorStatus && errorStatus >= 500 && <p style={{ marginBottom: 0 }}>服务暂时不可用，请稍后重试。</p>}
          <p style={{ marginBottom: 0 }}>
            <Link href="/extractions">返回审核列表</Link>
          </p>
        </section>
      )}
      {!loading && !error && !extraction && <p>空状态：未找到抽取数据。</p>}

      {!loading && !error && extraction && (
        <div style={{ display: "grid", gap: "12px" }}>
          <section style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "10px" }}>
            <h2 style={{ marginTop: 0 }}>原始贴文</h2>
            <div>平台: {extraction.raw_post.platform}</div>
            <div>作者: @{extraction.raw_post.author_handle}</div>
            <div>发布时间: {displayPostedTime(extraction.raw_post)}</div>
            <div>
              链接:{" "}
              <a href={extraction.raw_post.url} target="_blank" rel="noreferrer">
                {extraction.raw_post.url}
              </a>
            </div>
            <pre
              style={{
                whiteSpace: "pre-wrap",
                border: "1px solid #eee",
                borderRadius: "6px",
                padding: "8px",
                marginTop: "8px",
              }}
            >
              {extraction.raw_post.content_text}
            </pre>
          </section>

          <section style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "10px" }}>
            <h2 style={{ marginTop: 0 }}>抽取结果 JSON</h2>
            <div>抽取器: {extraction.extractor_name}</div>
            <div>模型: {extraction.model_name}</div>
            {statusError && <div style={{ color: "crimson" }}>抽取器状态错误: {statusError}</div>}
            {extractorStatus && (
              <div style={{ marginTop: "4px" }}>
                状态: mode={extractorStatus.mode}, base_url={extractorStatus.base_url}, default_model=
                {extractorStatus.default_model}, has_api_key={extractorStatus.has_api_key ? "是" : "否"}, budget=
                {extractorStatus.call_budget_remaining ?? "无限制"}
              </div>
            )}
            {extraction.last_error && <div style={{ color: "crimson" }}>最后错误: {extraction.last_error}</div>}
            {reviewFailure && <div style={{ color: "#8a5800", marginTop: "4px" }}>{reviewFailure}</div>}
            {budgetExhaustedFallback && (
              <div style={{ color: "#b35c00" }}>已自动降级 Dummy，避免过度消耗额度。</div>
            )}
            <pre style={{ whiteSpace: "pre-wrap", margin: 0 }}>{JSON.stringify(extraction.extracted_json, null, 2)}</pre>
          </section>

          <section style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "10px" }}>
            <h2 style={{ marginTop: 0 }}>按资产观点</h2>
            {extractedAssetViews.length === 0 && <div>（无）</div>}
            {extractedAssetViews.length > 0 && (
              <div style={{ display: "grid", gap: "8px" }}>
                {extractedAssetViews.map((item, index) => (
                  <div key={`${index}:${item.symbol}:${item.horizon}`} style={{ border: "1px solid #eee", padding: "8px" }}>
                    <div>
                      {item.symbol} | {item.stance} | {item.horizon} | 置信度={item.confidence}
                      {autoAppliedKeySet.has(buildAssetViewKey(item, extractedAsOf)) && (
                        <span
                          style={{
                            marginLeft: "6px",
                            padding: "1px 6px",
                            borderRadius: "10px",
                            fontSize: "12px",
                            color: "#666",
                            background: "#f0f0f0",
                          }}
                        >
                          自动通过
                        </span>
                      )}
                    </div>
                    <div>摘要: {item.summary || "（无）"}</div>
                  </div>
                ))}
              </div>
            )}
            <div style={{ marginTop: "8px" }}>
              自动应用数量={extraction.auto_applied_count}, 自动策略={extraction.auto_policy || "null"},
              自动应用观点ID=
              {extraction.auto_applied_kol_view_ids ? extraction.auto_applied_kol_view_ids.join(",") : "[]"}
            </div>
            {extraction.auto_policy === "top1_fallback" && (
              <div style={{ color: "#8a5800" }}>自动审核未达到阈值，已按 top1_fallback 仅落最高分一条。</div>
            )}
          </section>

          <section style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "10px" }}>
            <details>
              <summary style={{ cursor: "pointer", fontWeight: 700 }}>调试 / 模型输出</summary>
              <div style={{ marginTop: "10px", display: "grid", gap: "6px" }}>
                <div>提示词版本: {extraction.prompt_version || "（无）"}</div>
                <div>提示词哈希: {extraction.prompt_hash || "（无）"}</div>
                <div>模型名: {extraction.model_name}</div>
                <div>抽取器名: {extraction.extractor_name}</div>
                <div>延迟(ms): {extraction.model_latency_ms ?? "（无）"}</div>
                <div>
                  tokens: 输入={extraction.model_input_tokens ?? "（无）"} / 输出=
                  {extraction.model_output_tokens ?? "（无）"}
                </div>
                <div>
                  <button type="button" onClick={() => void copyRawOutput()} disabled={!extraction.raw_model_output}>
                    复制原始输出
                  </button>
                  {copyMessage && <span style={{ marginLeft: "8px" }}>{copyMessage}</span>}
                </div>
                <div>原始模型输出:</div>
                <pre
                  style={{
                    whiteSpace: "pre-wrap",
                    maxHeight: "280px",
                    overflow: "auto",
                    border: "1px solid #eee",
                    borderRadius: "6px",
                    padding: "8px",
                    margin: 0,
                  }}
                >
                  {extraction.raw_model_output || "（空）"}
                </pre>
                <div>解析后模型输出:</div>
                <pre
                  style={{
                    whiteSpace: "pre-wrap",
                    maxHeight: "280px",
                    overflow: "auto",
                    border: "1px solid #eee",
                    borderRadius: "6px",
                    padding: "8px",
                    margin: 0,
                  }}
                >
                  {extraction.parsed_model_output ? JSON.stringify(extraction.parsed_model_output, null, 2) : "（无）"}
                </pre>
              </div>
            </details>
          </section>

          <section style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "10px" }}>
            <h2 style={{ marginTop: 0 }}>操作</h2>
            <button type="button" onClick={() => void reExtract()} disabled={reExtracting}>
              {reExtracting ? "重新解构中..." : "用 AI 重新解构（强制）"}
            </button>
          </section>

          {actionError && <p style={{ color: "crimson" }}>{actionError}</p>}
          {actionInfo && <p style={{ color: "green" }}>{actionInfo}</p>}
        </div>
      )}
    </main>
  );
}
