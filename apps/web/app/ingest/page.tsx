"use client";

import Link from "next/link";
import type { FormEvent } from "react";
import { useEffect, useState } from "react";

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
};

type FormState = {
  platform: "x" | "reddit" | "other";
  author_handle: string;
  url: string;
  content_text: string;
  posted_at: string;
};

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
  const [statusError, setStatusError] = useState<string | null>(null);

  useEffect(() => {
    const loadStatus = async () => {
      try {
        const res = await fetch("/api/extractor-status", { cache: "no-store" });
        const body = (await res.json()) as ExtractorStatus | { detail?: string };
        if (!res.ok) {
          throw new Error("detail" in body ? (body.detail ?? "Load extractor status failed") : "Load failed");
        }
        setExtractorStatus(body as ExtractorStatus);
      } catch (err) {
        setStatusError(err instanceof Error ? err.message : "Load extractor status failed");
      }
    };
    void loadStatus();
  }, []);

  const onSubmit = async (event: FormEvent<HTMLFormElement>) => {
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
      if (form.posted_at) {
        payload.posted_at = new Date(form.posted_at).toISOString();
      }

      const res = await fetch("/api/ingest/manual", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const body = (await res.json()) as ManualIngestResponse | { detail?: string };
      if (!res.ok) {
        throw new Error("detail" in body ? (body.detail ?? "Submit failed") : "Submit failed");
      }
      setResult(body as ManualIngestResponse);
      setForm((prev) => ({ ...prev, url: "", content_text: "" }));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <main style={{ padding: "24px", fontFamily: "monospace", display: "grid", gap: "12px" }}>
      <h1>Manual Ingest</h1>
      <p>
        标准化导入入口（仅手动输入，不接入真实 X/Reddit API）。<Link href="/dashboard">返回 Dashboard</Link>
      </p>
      <section style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "10px", maxWidth: "760px" }}>
        <h2 style={{ marginTop: 0 }}>Extractor Status</h2>
        {statusError && <p style={{ color: "crimson" }}>{statusError}</p>}
        {!statusError && !extractorStatus && <p>Loading extractor status...</p>}
        {extractorStatus && (
          <p style={{ margin: 0 }}>
            EXTRACTOR_MODE: <strong>{extractorStatus.mode}</strong> | default_model: {extractorStatus.default_model} |
            has_api_key: {extractorStatus.has_api_key ? "yes" : "no"}
          </p>
        )}
      </section>

      <form onSubmit={onSubmit} style={{ display: "grid", gap: "8px", maxWidth: "760px" }}>
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
          <input
            value={form.author_handle}
            onChange={(event) => setForm((prev) => ({ ...prev, author_handle: event.target.value }))}
            style={{ display: "block", width: "100%" }}
            required
          />
        </label>

        <label>
          url
          <input
            type="url"
            value={form.url}
            onChange={(event) => setForm((prev) => ({ ...prev, url: event.target.value }))}
            style={{ display: "block", width: "100%" }}
            required
          />
        </label>

        <label>
          content_text
          <textarea
            value={form.content_text}
            onChange={(event) => setForm((prev) => ({ ...prev, content_text: event.target.value }))}
            rows={5}
            style={{ display: "block", width: "100%" }}
            required
          />
        </label>

        <label>
          posted_at (optional)
          <input
            type="datetime-local"
            value={form.posted_at}
            onChange={(event) => setForm((prev) => ({ ...prev, posted_at: event.target.value }))}
            style={{ display: "block", width: "100%" }}
          />
        </label>

        <button type="submit" disabled={submitting}>
          {submitting ? "Submitting..." : "Create Raw Post + Pending Extraction"}
        </button>
      </form>

      {error && <p style={{ color: "crimson" }}>{error}</p>}
      {result && (
        <div style={{ border: "1px solid #96d29a", borderRadius: "8px", padding: "10px", background: "#f4fff5" }}>
          <p style={{ margin: 0 }}>
            提交成功。raw_post_id={result.extraction.raw_post_id}，extraction_id={result.extraction_id}，
            extractor_name={result.extraction.extractor_name}
          </p>
          {result.extraction.last_error && (
            <p style={{ color: "crimson", marginBottom: 0 }}>last_error: {result.extraction.last_error}</p>
          )}
          <p style={{ marginBottom: 0 }}>
            <Link href={`/extractions/${result.extraction_id}`}>去审核该 extraction</Link>
          </p>
        </div>
      )}
    </main>
  );
}
