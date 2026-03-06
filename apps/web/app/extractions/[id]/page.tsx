"use client";

import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import type { FormEvent } from "react";
import { useEffect, useMemo, useState } from "react";
import { buildMissingInferenceHints, pickAssetSymbols, pickAssetViews, pickDefaults } from "./review-inference.js";

type Asset = {
  id: number;
  symbol: string;
  name: string | null;
  market: string | null;
  created_at: string;
};

type Kol = {
  id: number;
  platform: string;
  handle: string;
  display_name: string | null;
  enabled: boolean;
  created_at: string;
};

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

type FormState = {
  kol_id: string;
  asset_id: string;
  stance: "bull" | "bear" | "neutral";
  horizon: "intraday" | "1w" | "1m" | "3m" | "1y";
  confidence: string;
  summary: string;
  source_url: string;
  as_of: string;
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

function statusText(status: Extraction["status"]): string {
  if (status === "pending") return "待处理";
  if (status === "approved") return "已通过";
  if (status === "rejected") return "已拒绝";
  return status;
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
  const [assets, setAssets] = useState<Asset[]>([]);
  const [kols, setKols] = useState<Kol[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [errorStatus, setErrorStatus] = useState<number | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const [actionInfo, setActionInfo] = useState<string | null>(null);
  const [matchingAssetHint, setMatchingAssetHint] = useState<string | null>(null);
  const [reExtracting, setReExtracting] = useState(false);
  const [extractorStatus, setExtractorStatus] = useState<ExtractorStatus | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [form, setForm] = useState<FormState>({
    kol_id: "",
    asset_id: "",
    stance: "neutral",
    horizon: "1w",
    confidence: "50",
    summary: "",
    source_url: "",
    as_of: todayIsoDate(),
  });
  const [rejectReason, setRejectReason] = useState("");
  const [copyMessage, setCopyMessage] = useState<string | null>(null);
  const [selectedBatchKeys, setSelectedBatchKeys] = useState<string[]>([]);

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
        const [extractionRes, assetsRes, kolsRes, statusRes] = await Promise.all([
          fetch(`/api/extractions/${extractionId}`, { cache: "no-store" }),
          fetch("/api/assets", { cache: "no-store" }),
          fetch("/api/kols", { cache: "no-store" }),
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

        const assetsBody = (await assetsRes.json()) as Asset[] | { detail?: string };
        if (!assetsRes.ok) {
          throw new Error("detail" in assetsBody ? (assetsBody.detail ?? "加载资产失败") : "加载资产失败");
        }

        const kolsBody = (await kolsRes.json()) as Kol[] | { detail?: string };
        if (!kolsRes.ok) {
          throw new Error("detail" in kolsBody ? (kolsBody.detail ?? "加载 KOL 失败") : "加载 KOL 失败");
        }
        const statusBody = (await statusRes.json()) as ExtractorStatus | { detail?: string };
        if (!statusRes.ok) {
          setStatusError("detail" in statusBody ? (statusBody.detail ?? "加载抽取器状态失败") : "加载失败");
        } else {
          setExtractorStatus(statusBody as ExtractorStatus);
        }

        const extractionData = extractionBody as Extraction;
        const assetList = assetsBody as Asset[];
        const kolList = kolsBody as Kol[];

        setExtraction(extractionData);
        setAssets(assetList);
        setKols(kolList);

        const defaults = pickDefaults(extractionData.extracted_json, extractionData.raw_post.url);
        const extractedSymbols = pickAssetSymbols(extractionData.extracted_json);
        const uniqueSymbols = Array.from(new Set(extractedSymbols));
        const matchedAssets = uniqueSymbols
          .map((symbol) => assetList.find((asset) => asset.symbol.toUpperCase() === symbol))
          .filter((asset): asset is Asset => Boolean(asset));

        if (uniqueSymbols.length > 1) {
          setMatchingAssetHint(`提取到多个资产(${uniqueSymbols.join(", ")})，请手动选择目标资产。`);
        } else if (uniqueSymbols.length === 1 && matchedAssets.length === 0) {
          setMatchingAssetHint(`提取到资产 ${uniqueSymbols[0]}，未匹配资产，请手动选择/先创建资产。`);
        } else {
          setMatchingAssetHint(null);
        }

        setForm({
          kol_id: kolList[0] ? String(kolList[0].id) : "",
          asset_id: uniqueSymbols.length === 1 && matchedAssets[0] ? String(matchedAssets[0].id) : "",
          stance: "neutral",
          horizon: "1w",
          confidence: "50",
          summary: "",
          source_url: extractionData.raw_post.url,
          as_of: todayIsoDate(),
          ...defaults,
        });

        const extractedViews = pickAssetViews(extractionData.extracted_json);
        const threshold = extractionData.auto_approve_confidence_threshold ?? 80;
        const extractedAsOf = normalizeAsOfDate(
          typeof extractionData.extracted_json.as_of === "string"
            ? (extractionData.extracted_json.as_of as string)
            : null,
          extractionData.raw_post.posted_at.slice(0, 10),
        );
        const autoAppliedKeys = new Set(
          (extractionData.auto_applied_views || []).map((item) =>
            `${item.symbol.toUpperCase()}|${item.horizon}|${normalizeAsOfDate(item.as_of, extractedAsOf)}`,
          ),
        );
        const selectableViews = extractedViews.filter((item) => !autoAppliedKeys.has(buildAssetViewKey(item, extractedAsOf)));
        const highConfidenceKeys = selectableViews
          .filter((item) => item.confidence >= threshold)
          .map((item) => buildAssetViewKey(item, extractedAsOf));
        if (highConfidenceKeys.length > 0) {
          setSelectedBatchKeys(Array.from(new Set(highConfidenceKeys)));
        } else if (selectableViews[0]) {
          setSelectedBatchKeys([buildAssetViewKey(selectableViews[0], extractedAsOf)]);
        } else {
          setSelectedBatchKeys([]);
        }
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

  const approve = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setActionError(null);

    if (!form.kol_id || !form.asset_id) {
      setActionError("请选择资产和 KOL");
      return;
    }

    setSubmitting(true);
    try {
      const res = await fetch(`/api/extractions/${extractionId}/approve`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          kol_id: Number(form.kol_id),
          asset_id: Number(form.asset_id),
          stance: form.stance,
          horizon: form.horizon,
          confidence: Number(form.confidence),
          summary: form.summary.trim(),
          source_url: form.source_url.trim(),
          as_of: form.as_of,
        }),
      });
      const body = (await res.json()) as { detail?: string };
      if (!res.ok) {
        throw new Error(getHttpErrorMessage(res.status, body.detail, `审核通过失败: ${res.status}`));
      }
      router.push("/extractions?msg=%E5%AE%A1%E6%A0%B8%E9%80%9A%E8%BF%87&status=pending");
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "审核通过失败");
    } finally {
      setSubmitting(false);
    }
  };

  const approveBatch = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setActionError(null);
    if (!form.kol_id) {
      setActionError("请选择 KOL");
      return;
    }
    const selected = extractedAssetViews
      .map((item) => ({ item, key: buildAssetViewKey(item, extractedAsOf) }))
      .filter((item) => selectedBatchKeys.includes(item.key) && !autoAppliedKeySet.has(item.key));
    if (selected.length === 0) {
      setActionError("请至少勾选一条资产观点");
      return;
    }
    const views = selected
      .map(({ item }) => {
        const matchedAsset = assets.find((asset) => asset.symbol.toUpperCase() === item.symbol);
        if (!matchedAsset) {
          return null;
        }
        return {
          asset_id: matchedAsset.id,
          stance: item.stance,
          horizon: item.horizon,
          confidence: item.confidence,
          summary: (item.summary || `${item.symbol} ${item.stance}`).slice(0, 1024),
          source_url: (form.source_url || extraction?.raw_post.url || "").trim(),
          as_of: normalizeAsOfDate(item.as_of, extractedAsOf),
        };
      })
      .filter((item): item is NonNullable<typeof item> => item !== null);
    if (views.length === 0) {
      setActionError("所选观点无法匹配资产，请先创建对应资产或检查 symbol");
      return;
    }

    setSubmitting(true);
    try {
      const res = await fetch(`/api/extractions/${extractionId}/approve-batch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          kol_id: Number(form.kol_id),
          views,
        }),
      });
      const body = (await res.json()) as { detail?: string };
      if (!res.ok) {
        throw new Error(getHttpErrorMessage(res.status, body.detail, `批量审核通过失败: ${res.status}`));
      }
      router.push("/extractions?msg=%E6%89%B9%E9%87%8F%E5%AE%A1%E6%A0%B8%E9%80%9A%E8%BF%87&status=pending");
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "批量审核通过失败");
    } finally {
      setSubmitting(false);
    }
  };

  const reject = async () => {
    const confirmed = window.confirm("确认拒绝该 extraction 吗？该操作将把状态设为 rejected。");
    if (!confirmed) return;
    setActionError(null);
    setSubmitting(true);
    try {
      const res = await fetch(`/api/extractions/${extractionId}/reject`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reason: rejectReason.trim() || null }),
      });
      const body = (await res.json()) as { detail?: string };
      if (!res.ok) {
        throw new Error(getHttpErrorMessage(res.status, body.detail, `审核拒绝失败: ${res.status}`));
      }
      router.push("/extractions?msg=%E5%AE%A1%E6%A0%B8%E6%8B%92%E7%BB%9D&status=pending");
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "审核拒绝失败");
    } finally {
      setSubmitting(false);
    }
  };

  const missingInferenceHints = useMemo(() => {
    if (!extraction) return [];
    return buildMissingInferenceHints(extraction.extracted_json);
  }, [extraction]);

  const budgetExhaustedFallback = useMemo(() => {
    if (!extraction) return false;
    if (extraction.last_error?.includes("budget_exhausted")) return true;
    const meta = extraction.extracted_json["meta"];
    if (!meta || typeof meta !== "object") return false;
    return (meta as Record<string, unknown>)["fallback_reason"] === "budget_exhausted";
  }, [extraction]);

  const autoRejectMeta = useMemo(() => {
    if (!extraction) return null;
    const rawMeta = extraction.extracted_json["meta"];
    if (!rawMeta || typeof rawMeta !== "object") return null;
    const meta = rawMeta as Record<string, unknown>;
    if (meta["auto_rejected"] !== true) return null;
    return {
      reason: String(meta["auto_reject_reason"] ?? "-"),
      threshold: String(meta["auto_reject_threshold"] ?? extraction.auto_reject_confidence_threshold ?? "-"),
      modelConfidence: String(meta["model_confidence"] ?? "-"),
    };
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
            {autoRejectMeta && (
              <div style={{ color: "#8a5800", marginTop: "4px" }}>
                自动拒绝=true, 原因={autoRejectMeta.reason}, 阈值={autoRejectMeta.threshold},
                模型置信度={autoRejectMeta.modelConfidence}
              </div>
            )}
            {typeof extraction.extracted_json["meta"] === "object" && extraction.extracted_json["meta"] !== null && (
              <div style={{ marginTop: "4px" }}>
                meta: provider_detected=
                {String((extraction.extracted_json["meta"] as Record<string, unknown>)["provider_detected"] ?? "-")},
                output_mode_used=
                {String((extraction.extracted_json["meta"] as Record<string, unknown>)["output_mode_used"] ?? "-")},
                parse_strategy_used=
                {String((extraction.extracted_json["meta"] as Record<string, unknown>)["parse_strategy_used"] ?? "-")},
                raw_len={String((extraction.extracted_json["meta"] as Record<string, unknown>)["raw_len"] ?? "-")},
                repaired={String((extraction.extracted_json["meta"] as Record<string, unknown>)["repaired"] ?? "-")}
              </div>
            )}
            {budgetExhaustedFallback && (
              <div style={{ color: "#b35c00" }}>已自动降级 Dummy，避免过度消耗额度。</div>
            )}
            {missingInferenceHints.length > 0 && (
              <div style={{ color: "#8a5800", marginTop: "6px" }}>
                {missingInferenceHints.join("；")}。请在下方人工补齐后再审核。
              </div>
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
            <h2 style={{ marginTop: 0 }}>审核通过</h2>
            <button
              type="button"
              onClick={() => void reExtract()}
              disabled={reExtracting}
              style={{ marginBottom: "10px" }}
            >
              {reExtracting ? "重新解构中..." : "用 AI 重新解构（强制）"}
            </button>
            <p style={{ marginTop: 0, color: "#555" }}>
              自动审核规则：`hasview=0` 自动拒绝；`hasview=1` 时按阈值 80 执行（&gt;=80 自动通过，&lt;80 自动拒绝）；手动强制重抽默认进入待人工审核。
            </p>
            {matchingAssetHint && <p style={{ color: "#b35c00", marginTop: 0 }}>{matchingAssetHint}</p>}
            <form
              onSubmit={extractedAssetViews.length > 0 ? approveBatch : approve}
              style={{ display: "grid", gap: "8px", maxWidth: "720px" }}
            >
              {extractedAssetViews.length === 0 && (
                <label>
                  资产
                  <select
                    value={form.asset_id}
                    onChange={(event) => setForm((prev) => ({ ...prev, asset_id: event.target.value }))}
                    style={{ display: "block", width: "100%" }}
                    required
                  >
                    <option value="" disabled>
                      请选择资产
                    </option>
                    {assets.map((asset) => (
                      <option key={asset.id} value={asset.id}>
                        {asset.symbol} {asset.name ? `- ${asset.name}` : ""}
                      </option>
                    ))}
                  </select>
                </label>
              )}
              <label>
                KOL
                <select
                  value={form.kol_id}
                  onChange={(event) => setForm((prev) => ({ ...prev, kol_id: event.target.value }))}
                  style={{ display: "block", width: "100%" }}
                  required
                >
                  <option value="" disabled>
                    请选择 KOL
                  </option>
                  {kols.map((kol) => (
                    <option key={kol.id} value={kol.id}>
                      {kol.display_name || kol.handle} ({kol.platform}/@{kol.handle})
                    </option>
                  ))}
                </select>
              </label>
              {extractedAssetViews.length > 0 && (
                <div>
                  <div style={{ marginBottom: "6px" }}>资产观点（可多选）</div>
                  <div style={{ display: "grid", gap: "4px" }}>
                    {extractedAssetViews.map((item, index) => {
                      const key = buildAssetViewKey(item, extractedAsOf);
                      const checked = selectedBatchKeys.includes(key);
                      const isAutoApproved = autoAppliedKeySet.has(key);
                      return (
                        <label
                          key={`${index}:${key}`}
                          style={{
                            border: "1px solid #eee",
                            padding: "6px",
                            borderRadius: "6px",
                            opacity: isAutoApproved ? 0.6 : 1,
                          }}
                        >
                          <input
                            type="checkbox"
                            checked={checked}
                            disabled={isAutoApproved}
                            onChange={(event) => {
                              setSelectedBatchKeys((prev) => {
                                if (event.target.checked) return Array.from(new Set([...prev, key]));
                                return prev.filter((itemKey) => itemKey !== key);
                              });
                            }}
                          />{" "}
                          {item.symbol} | {item.stance} | {item.horizon} | 置信度={item.confidence} |{" "}
                          {item.summary || "（无）"}
                          {isAutoApproved && (
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
                        </label>
                      );
                    })}
                  </div>
                </div>
              )}

              {extractedAssetViews.length === 0 && (
                <label>
                  方向
                  <select
                    value={form.stance}
                    onChange={(event) => setForm((prev) => ({ ...prev, stance: event.target.value as FormState["stance"] }))}
                    style={{ display: "block", width: "100%" }}
                    required
                  >
                    <option value="bull">看涨</option>
                    <option value="bear">看跌</option>
                    <option value="neutral">中性</option>
                  </select>
                </label>
              )}

              {extractedAssetViews.length === 0 && (
                <label>
                  影响周期
                  <select
                    value={form.horizon}
                    onChange={(event) =>
                      setForm((prev) => ({ ...prev, horizon: event.target.value as FormState["horizon"] }))
                    }
                    style={{ display: "block", width: "100%" }}
                    required
                  >
                    <option value="intraday">日内</option>
                    <option value="1w">1w</option>
                    <option value="1m">1m</option>
                    <option value="3m">3m</option>
                    <option value="1y">1y</option>
                  </select>
                </label>
              )}

              {extractedAssetViews.length === 0 && (
                <label>
                  置信度（0-100）
                  <input
                    type="number"
                    min={0}
                    max={100}
                    value={form.confidence}
                    onChange={(event) => setForm((prev) => ({ ...prev, confidence: event.target.value }))}
                    style={{ display: "block", width: "100%" }}
                    required
                  />
                </label>
              )}

              {extractedAssetViews.length === 0 && (
                <label>
                  观点摘要
                  <textarea
                    rows={3}
                    value={form.summary}
                    onChange={(event) => setForm((prev) => ({ ...prev, summary: event.target.value }))}
                    style={{ display: "block", width: "100%" }}
                    required
                  />
                </label>
              )}

              <label>
                来源链接
                <input
                  type="url"
                  value={form.source_url}
                  onChange={(event) => setForm((prev) => ({ ...prev, source_url: event.target.value }))}
                  style={{ display: "block", width: "100%" }}
                  required
                />
              </label>

              <label>
                观点日期
                <input
                  type="date"
                  value={form.as_of}
                  onChange={(event) => setForm((prev) => ({ ...prev, as_of: event.target.value }))}
                  style={{ display: "block", width: "100%" }}
                  required
                />
              </label>

              <button type="submit" disabled={submitting || extraction.status !== "pending"}>
                {submitting ? "提交中..." : extractedAssetViews.length > 0 ? "批量通过" : "审核通过"}
              </button>
            </form>
          </section>

          <section style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "10px" }}>
            <h2 style={{ marginTop: 0 }}>审核拒绝</h2>
            <label>
              拒绝原因（可选）
              <textarea
                rows={2}
                value={rejectReason}
                onChange={(event) => setRejectReason(event.target.value)}
                style={{ display: "block", width: "100%", maxWidth: "720px" }}
              />
            </label>
            <button
              type="button"
              onClick={() => void reject()}
              disabled={submitting || extraction.status !== "pending"}
              style={{ marginTop: "8px" }}
            >
              拒绝
            </button>
          </section>

          {actionError && <p style={{ color: "crimson" }}>{actionError}</p>}
          {actionInfo && <p style={{ color: "green" }}>{actionInfo}</p>}
          {extraction.status !== "pending" && (
            <p style={{ color: "#666" }}>
              当前状态是 {statusText(extraction.status)}，不可再次审核。reviewed_by={extraction.reviewed_by ?? "N/A"}
            </p>
          )}
        </div>
      )}
    </main>
  );
}
