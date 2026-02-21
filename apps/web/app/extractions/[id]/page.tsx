"use client";

import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import type { FormEvent } from "react";
import { useEffect, useMemo, useState } from "react";

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
  auto_approve_min_display_confidence: number | null;
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
  reasoning: string | null;
  summary: string | null;
  as_of: string | null;
  drivers: string[];
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

const VALID_STANCE = new Set(["bull", "bear", "neutral"]);
const VALID_HORIZON = new Set(["intraday", "1w", "1m", "3m", "1y"]);

function pickDefaults(extracted: Record<string, unknown>, rawPostUrl: string): Partial<FormState> {
  const result: Partial<FormState> = {
    source_url: rawPostUrl,
  };

  if (typeof extracted.stance === "string" && VALID_STANCE.has(extracted.stance)) {
    result.stance = extracted.stance as FormState["stance"];
  }
  if (typeof extracted.horizon === "string" && VALID_HORIZON.has(extracted.horizon)) {
    result.horizon = extracted.horizon as FormState["horizon"];
  }
  if (typeof extracted.confidence === "number") {
    result.confidence = String(Math.max(0, Math.min(100, Math.round(extracted.confidence))));
  }
  if (typeof extracted.summary === "string") result.summary = extracted.summary;
  if (typeof extracted.source_url === "string") result.source_url = extracted.source_url;
  if (typeof extracted.as_of === "string") result.as_of = extracted.as_of;

  const candidates = extracted.candidates;
  if (Array.isArray(candidates) && candidates.length > 0) {
    const first = candidates[0] as Record<string, unknown>;
    if (typeof first.summary === "string") result.summary = first.summary;
    if (typeof first.source_url === "string") result.source_url = first.source_url;
    if (typeof first.as_of === "string") result.as_of = first.as_of;
    if (typeof first.stance === "string" && VALID_STANCE.has(first.stance)) {
      result.stance = first.stance as FormState["stance"];
    }
    if (typeof first.horizon === "string" && VALID_HORIZON.has(first.horizon)) {
      result.horizon = first.horizon as FormState["horizon"];
    }
    if (typeof first.confidence === "number") {
      result.confidence = String(Math.max(0, Math.min(100, Math.round(first.confidence))));
    }
  }

  return result;
}

function pickAssetSymbols(extracted: Record<string, unknown>): string[] {
  const assets = extracted.assets;
  if (!Array.isArray(assets) || assets.length === 0) {
    return [];
  }
  const symbols: string[] = [];
  for (const item of assets) {
    const asset = item as Record<string, unknown>;
    if (typeof asset.symbol === "string" && asset.symbol.trim()) {
      symbols.push(asset.symbol.trim().toUpperCase());
    }
  }
  return symbols;
}

function todayIsoDate(): string {
  return new Date().toISOString().slice(0, 10);
}

function pickAssetViews(extracted: Record<string, unknown>): AssetViewItem[] {
  const value = extracted.asset_views;
  if (!Array.isArray(value)) return [];
  const items: AssetViewItem[] = [];
  for (const raw of value) {
    if (!raw || typeof raw !== "object") continue;
    const item = raw as Record<string, unknown>;
    if (typeof item.symbol !== "string" || !item.symbol.trim()) continue;
    if (typeof item.stance !== "string" || !VALID_STANCE.has(item.stance)) continue;
    if (typeof item.horizon !== "string" || !VALID_HORIZON.has(item.horizon)) continue;
    if (typeof item.confidence !== "number") continue;
    items.push({
      symbol: item.symbol.trim().toUpperCase(),
      stance: item.stance as AssetViewItem["stance"],
      horizon: item.horizon as AssetViewItem["horizon"],
      confidence: Math.max(0, Math.min(100, Math.round(item.confidence))),
      reasoning: typeof item.reasoning === "string" ? item.reasoning : null,
      summary: typeof item.summary === "string" ? item.summary : null,
      as_of: typeof item.as_of === "string" ? item.as_of : null,
      drivers: Array.isArray(item.drivers)
        ? item.drivers.filter((driver): driver is string => typeof driver === "string")
        : [],
    });
  }
  return items.sort((a, b) => b.confidence - a.confidence);
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
  const [matchingAssetHint, setMatchingAssetHint] = useState<string | null>(null);
  const [reExtracting, setReExtracting] = useState(false);
  const [reExtractCooldown, setReExtractCooldown] = useState(false);
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
        setError("Invalid extraction id");
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
              "Load extraction failed",
            ),
          );
        }

        const assetsBody = (await assetsRes.json()) as Asset[] | { detail?: string };
        if (!assetsRes.ok) {
          throw new Error("detail" in assetsBody ? (assetsBody.detail ?? "Load assets failed") : "Load assets failed");
        }

        const kolsBody = (await kolsRes.json()) as Kol[] | { detail?: string };
        if (!kolsRes.ok) {
          throw new Error("detail" in kolsBody ? (kolsBody.detail ?? "Load kols failed") : "Load kols failed");
        }
        const statusBody = (await statusRes.json()) as ExtractorStatus | { detail?: string };
        if (!statusRes.ok) {
          setStatusError("detail" in statusBody ? (statusBody.detail ?? "Load extractor status failed") : "Load failed");
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
        const minDisplayConfidence = extractionData.auto_approve_min_display_confidence ?? 50;
        const threshold = extractionData.auto_approve_confidence_threshold ?? 70;
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
        const selectableViews = extractedViews.filter(
          (item) =>
            item.confidence >= minDisplayConfidence && !autoAppliedKeys.has(buildAssetViewKey(item, extractedAsOf)),
        );
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
        setError(err instanceof Error ? err.message : "Unknown error");
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
    if (reExtractCooldown) {
      return;
    }
    if (!window.confirm("确认重新提取？这会创建一个新的 pending extraction。")) {
      return;
    }
    setActionError(null);
    setReExtractCooldown(true);
    window.setTimeout(() => setReExtractCooldown(false), 3000);
    setReExtracting(true);
    try {
      const res = await fetch(`/api/raw-posts/${extraction.raw_post_id}/extract`, { method: "POST" });
      const body = (await res.json()) as { id?: number; detail?: string };
      if (!res.ok || typeof body.id !== "number") {
        throw new Error(getHttpErrorMessage(res.status, body.detail, `Re-extract failed: ${res.status}`));
      }
      router.push(`/extractions/${body.id}`);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Re-extract failed");
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
        throw new Error(getHttpErrorMessage(res.status, body.detail, `Approve failed: ${res.status}`));
      }
      router.push("/extractions?msg=Approve%20success&status=pending");
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Approve failed");
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
          summary: (item.summary || item.reasoning || `${item.symbol} ${item.stance}`).slice(0, 1024),
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
        throw new Error(getHttpErrorMessage(res.status, body.detail, `Approve batch failed: ${res.status}`));
      }
      router.push("/extractions?msg=Approve%20success&status=pending");
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Approve batch failed");
    } finally {
      setSubmitting(false);
    }
  };

  const reject = async () => {
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
        throw new Error(getHttpErrorMessage(res.status, body.detail, `Reject failed: ${res.status}`));
      }
      router.push("/extractions?msg=Reject%20success&status=pending");
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Reject failed");
    } finally {
      setSubmitting(false);
    }
  };

  const missingInferenceHints = useMemo(() => {
    if (!extraction) return [];
    const extracted = extraction.extracted_json;
    const hints: string[] = [];
    const stanceMissing = typeof extracted.stance !== "string" || !VALID_STANCE.has(extracted.stance);
    const horizonMissing = typeof extracted.horizon !== "string" || !VALID_HORIZON.has(extracted.horizon);
    const assetMissing = pickAssetSymbols(extracted).length === 0;

    if (stanceMissing) hints.push("stance: 模型未判断/信息不足");
    if (horizonMissing) hints.push("horizon: 模型未判断/信息不足");
    if (assetMissing) hints.push("asset: 模型未判断/信息不足");
    return hints;
  }, [extraction]);

  const budgetExhaustedFallback = useMemo(() => {
    if (!extraction) return false;
    if (extraction.last_error?.includes("budget_exhausted")) return true;
    const meta = extraction.extracted_json["meta"];
    if (!meta || typeof meta !== "object") return false;
    return (meta as Record<string, unknown>)["fallback_reason"] === "budget_exhausted";
  }, [extraction]);

  const extractedAssetViews = useMemo(() => {
    if (!extraction) return [];
    const minDisplayConfidence = extraction.auto_approve_min_display_confidence ?? 50;
    return pickAssetViews(extraction.extracted_json).filter((item) => item.confidence >= minDisplayConfidence);
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
      setCopyMessage("已复制 raw output");
      window.setTimeout(() => setCopyMessage(null), 1200);
    } catch {
      setCopyMessage("复制失败");
      window.setTimeout(() => setCopyMessage(null), 1200);
    }
  };

  return (
    <main style={{ padding: "24px", fontFamily: "monospace" }}>
      <h1>Extraction #{Number.isNaN(extractionId) ? "?" : extractionId}</h1>
      <p>
        <Link href="/extractions">返回审核列表</Link>
      </p>

      {loading && <p>Loading...</p>}
      {error && (
        <section style={{ border: "1px solid #f0bcbc", background: "#fff6f6", borderRadius: "8px", padding: "10px" }}>
          <p style={{ color: "crimson", margin: 0 }}>{error}</p>
          {errorStatus === 422 && <p style={{ marginBottom: 0 }}>请检查 extraction id 或请求参数。</p>}
          {errorStatus === 409 && <p style={{ marginBottom: 0 }}>资源状态冲突，请返回列表刷新后重试。</p>}
          {errorStatus && errorStatus >= 500 && <p style={{ marginBottom: 0 }}>服务暂时不可用，请稍后重试。</p>}
          <p style={{ marginBottom: 0 }}>
            <Link href="/extractions">返回审核列表</Link>
          </p>
        </section>
      )}
      {!loading && !error && !extraction && <p>空状态：未找到 extraction 数据。</p>}

      {!loading && !error && extraction && (
        <div style={{ display: "grid", gap: "12px" }}>
          <section style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "10px" }}>
            <h2 style={{ marginTop: 0 }}>Raw Post</h2>
            <div>platform: {extraction.raw_post.platform}</div>
            <div>author: @{extraction.raw_post.author_handle}</div>
            <div>posted_at: {extraction.raw_post.posted_at}</div>
            <div>
              url:{" "}
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
            <h2 style={{ marginTop: 0 }}>Extracted JSON</h2>
            <div>extractor: {extraction.extractor_name}</div>
            <div>model: {extraction.model_name}</div>
            {statusError && <div style={{ color: "crimson" }}>extractor_status_error: {statusError}</div>}
            {extractorStatus && (
              <div style={{ marginTop: "4px" }}>
                status: mode={extractorStatus.mode}, base_url={extractorStatus.base_url}, default_model=
                {extractorStatus.default_model}, has_api_key={extractorStatus.has_api_key ? "yes" : "no"}, budget=
                {extractorStatus.call_budget_remaining ?? "unlimited"}
              </div>
            )}
            {extraction.last_error && <div style={{ color: "crimson" }}>last_error: {extraction.last_error}</div>}
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
            <h2 style={{ marginTop: 0 }}>Per-Asset Views</h2>
            {extractedAssetViews.length === 0 && <div>(none)</div>}
            {extractedAssetViews.length > 0 && (
              <div style={{ display: "grid", gap: "8px" }}>
                {extractedAssetViews.map((item, index) => (
                  <div key={`${index}:${item.symbol}:${item.horizon}`} style={{ border: "1px solid #eee", padding: "8px" }}>
                    <div>
                      {item.symbol} | {item.stance} | {item.horizon} | confidence={item.confidence}
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
                          Auto approved
                        </span>
                      )}
                    </div>
                    <div>summary: {item.summary || "(none)"}</div>
                    <div>reasoning: {item.reasoning || "(none)"}</div>
                    {item.drivers.length > 0 && <div>drivers: {item.drivers.join(", ")}</div>}
                  </div>
                ))}
              </div>
            )}
            <div style={{ marginTop: "8px" }}>
              auto_applied_count={extraction.auto_applied_count}, auto_policy={extraction.auto_policy || "null"},
              auto_applied_kol_view_ids=
              {extraction.auto_applied_kol_view_ids ? extraction.auto_applied_kol_view_ids.join(",") : "[]"}
            </div>
            {extraction.auto_policy === "top1_fallback" && (
              <div style={{ color: "#8a5800" }}>自动审核未达到阈值，已按 top1_fallback 仅落最高分一条。</div>
            )}
          </section>

          <section style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "10px" }}>
            <details>
              <summary style={{ cursor: "pointer", fontWeight: 700 }}>Debug / 模型输出</summary>
              <div style={{ marginTop: "10px", display: "grid", gap: "6px" }}>
                <div>prompt_version: {extraction.prompt_version || "(none)"}</div>
                <div>prompt_hash: {extraction.prompt_hash || "(none)"}</div>
                <div>model_name: {extraction.model_name}</div>
                <div>extractor_name: {extraction.extractor_name}</div>
                <div>latency_ms: {extraction.model_latency_ms ?? "(none)"}</div>
                <div>
                  tokens: in={extraction.model_input_tokens ?? "(none)"} / out=
                  {extraction.model_output_tokens ?? "(none)"}
                </div>
                <div>
                  <button type="button" onClick={() => void copyRawOutput()} disabled={!extraction.raw_model_output}>
                    Copy raw output
                  </button>
                  {copyMessage && <span style={{ marginLeft: "8px" }}>{copyMessage}</span>}
                </div>
                <div>raw_model_output:</div>
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
                  {extraction.raw_model_output || "(empty)"}
                </pre>
                <div>parsed_model_output:</div>
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
                  {extraction.parsed_model_output ? JSON.stringify(extraction.parsed_model_output, null, 2) : "(none)"}
                </pre>
              </div>
            </details>
          </section>

          <section style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "10px" }}>
            <h2 style={{ marginTop: 0 }}>Approve</h2>
            <button
              type="button"
              onClick={() => void reExtract()}
              disabled={reExtracting || reExtractCooldown}
              style={{ marginBottom: "10px" }}
            >
              {reExtracting ? "Re-extracting..." : "Re-extract（需确认）"}
            </button>
            {matchingAssetHint && <p style={{ color: "#b35c00", marginTop: 0 }}>{matchingAssetHint}</p>}
            <form
              onSubmit={extractedAssetViews.length > 0 ? approveBatch : approve}
              style={{ display: "grid", gap: "8px", maxWidth: "720px" }}
            >
              {extractedAssetViews.length === 0 && (
                <label>
                  Asset
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
                  <div style={{ marginBottom: "6px" }}>Asset views（可多选）</div>
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
                          {item.symbol} | {item.stance} | {item.horizon} | confidence={item.confidence} |{" "}
                          {item.summary || item.reasoning || "(none)"}
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
                              Auto approved
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
                  stance
                  <select
                    value={form.stance}
                    onChange={(event) => setForm((prev) => ({ ...prev, stance: event.target.value as FormState["stance"] }))}
                    style={{ display: "block", width: "100%" }}
                    required
                  >
                    <option value="bull">bull</option>
                    <option value="bear">bear</option>
                    <option value="neutral">neutral</option>
                  </select>
                </label>
              )}

              {extractedAssetViews.length === 0 && (
                <label>
                  horizon
                  <select
                    value={form.horizon}
                    onChange={(event) =>
                      setForm((prev) => ({ ...prev, horizon: event.target.value as FormState["horizon"] }))
                    }
                    style={{ display: "block", width: "100%" }}
                    required
                  >
                    <option value="intraday">intraday</option>
                    <option value="1w">1w</option>
                    <option value="1m">1m</option>
                    <option value="3m">3m</option>
                    <option value="1y">1y</option>
                  </select>
                </label>
              )}

              {extractedAssetViews.length === 0 && (
                <label>
                  confidence (0-100)
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
                  summary
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
                source_url
                <input
                  type="url"
                  value={form.source_url}
                  onChange={(event) => setForm((prev) => ({ ...prev, source_url: event.target.value }))}
                  style={{ display: "block", width: "100%" }}
                  required
                />
              </label>

              <label>
                as_of
                <input
                  type="date"
                  value={form.as_of}
                  onChange={(event) => setForm((prev) => ({ ...prev, as_of: event.target.value }))}
                  style={{ display: "block", width: "100%" }}
                  required
                />
              </label>

              <button type="submit" disabled={submitting || extraction.status !== "pending"}>
                {submitting ? "Submitting..." : extractedAssetViews.length > 0 ? "Approve Batch" : "Approve"}
              </button>
            </form>
          </section>

          <section style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "10px" }}>
            <h2 style={{ marginTop: 0 }}>Reject</h2>
            <label>
              reason (optional)
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
              Reject
            </button>
          </section>

          {actionError && <p style={{ color: "crimson" }}>{actionError}</p>}
          {extraction.status !== "pending" && (
            <p style={{ color: "#666" }}>
              当前状态是 {extraction.status}，不可再次审核。reviewed_by={extraction.reviewed_by ?? "N/A"}
            </p>
          )}
        </div>
      )}
    </main>
  );
}
