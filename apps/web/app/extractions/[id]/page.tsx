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
  last_error: string | null;
  reviewed_at: string | null;
  reviewed_by: string | null;
  review_note: string | null;
  applied_kol_view_id: number | null;
  created_at: string;
  raw_post: RawPost;
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

function pickFirstAssetSymbol(extracted: Record<string, unknown>): string | null {
  const assets = extracted.assets;
  if (!Array.isArray(assets) || assets.length === 0) {
    return null;
  }
  const first = assets[0] as Record<string, unknown>;
  if (typeof first.symbol !== "string" || !first.symbol.trim()) {
    return null;
  }
  return first.symbol.trim().toUpperCase();
}

function todayIsoDate(): string {
  return new Date().toISOString().slice(0, 10);
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

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      setError(null);
      setErrorStatus(null);
      setActionError(null);

      if (Number.isNaN(extractionId)) {
        setError("Invalid extraction id");
        setLoading(false);
        return;
      }

      try {
        const [extractionRes, assetsRes, kolsRes] = await Promise.all([
          fetch(`/api/extractions/${extractionId}`, { cache: "no-store" }),
          fetch("/api/assets", { cache: "no-store" }),
          fetch("/api/kols", { cache: "no-store" }),
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

        const extractionData = extractionBody as Extraction;
        const assetList = assetsBody as Asset[];
        const kolList = kolsBody as Kol[];

        setExtraction(extractionData);
        setAssets(assetList);
        setKols(kolList);

        const defaults = pickDefaults(extractionData.extracted_json, extractionData.raw_post.url);
        const extractedSymbol = pickFirstAssetSymbol(extractionData.extracted_json);
        const matchedAsset = extractedSymbol
          ? assetList.find((asset) => asset.symbol.toUpperCase() === extractedSymbol)
          : null;

        if (extractedSymbol && !matchedAsset) {
          setMatchingAssetHint(`提取到资产 ${extractedSymbol}，未匹配资产，请手动选择/先创建资产。`);
        } else {
          setMatchingAssetHint(null);
        }

        setForm({
          kol_id: kolList[0] ? String(kolList[0].id) : "",
          asset_id: matchedAsset ? String(matchedAsset.id) : assetList[0] ? String(assetList[0].id) : "",
          stance: "neutral",
          horizon: "1w",
          confidence: "50",
          summary: "",
          source_url: extractionData.raw_post.url,
          as_of: todayIsoDate(),
          ...defaults,
        });
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
    const assetMissing = !pickFirstAssetSymbol(extracted);

    if (stanceMissing) hints.push("stance: 模型未判断/信息不足");
    if (horizonMissing) hints.push("horizon: 模型未判断/信息不足");
    if (assetMissing) hints.push("asset: 模型未判断/信息不足");
    return hints;
  }, [extraction]);

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
            {extraction.last_error && <div style={{ color: "crimson" }}>last_error: {extraction.last_error}</div>}
            {missingInferenceHints.length > 0 && (
              <div style={{ color: "#8a5800", marginTop: "6px" }}>
                {missingInferenceHints.join("；")}。请在下方人工补齐后再审核。
              </div>
            )}
            <pre style={{ whiteSpace: "pre-wrap", margin: 0 }}>{JSON.stringify(extraction.extracted_json, null, 2)}</pre>
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
            <form onSubmit={approve} style={{ display: "grid", gap: "8px", maxWidth: "720px" }}>
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
                {submitting ? "Submitting..." : "Approve"}
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
