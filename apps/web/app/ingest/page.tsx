"use client";

import Link from "next/link";
import type { FormEvent } from "react";
import { useState } from "react";

type ManualIngestResponse = {
  raw_post: { id: number };
  extraction: { id: number };
  extraction_id: number;
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
        <p style={{ color: "green" }}>
          ingest 成功。raw_post=#{result.raw_post.id}，extraction=#{result.extraction_id}，
          <Link href={`/extractions/${result.extraction_id}`}>去审核该 extraction</Link>
        </p>
      )}
    </main>
  );
}
