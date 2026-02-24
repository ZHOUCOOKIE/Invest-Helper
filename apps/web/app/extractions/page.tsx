"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";

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
  reviewed_at: string | null;
  reviewed_by: string | null;
  review_note: string | null;
  applied_kol_view_id: number | null;
  created_at: string;
  raw_post: RawPost;
};

type ClearPendingResponse = {
  deleted_extractions_count: number;
  deleted_raw_posts_count: number;
  scoped_author_handle: string | null;
};

const PAGE_SIZE = 20;

function summarizeExtraction(extracted: Record<string, unknown>): string {
  if (typeof extracted.summary === "string" && extracted.summary.trim()) {
    return extracted.summary;
  }

  const candidates = extracted.candidates;
  if (Array.isArray(candidates) && candidates.length > 0) {
    const first = candidates[0] as Record<string, unknown>;
    if (typeof first?.summary === "string" && first.summary.trim()) {
      return first.summary;
    }
  }

  return "(no summary)";
}

export default function ExtractionsPage() {
  const searchParams = useSearchParams();
  const initialStatus = (searchParams.get("status") as Extraction["status"] | null) ?? "pending";
  const [status, setStatus] = useState<Extraction["status"]>(initialStatus);
  const [offset, setOffset] = useState(0);
  const [items, setItems] = useState<Extraction[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(searchParams.get("msg"));
  const [clearAlsoDeleteRawPosts, setClearAlsoDeleteRawPosts] = useState(false);
  const requestSeqRef = useRef(0);

  const canPrev = useMemo(() => offset > 0, [offset]);
  const canNext = useMemo(() => items.length === PAGE_SIZE, [items.length]);

  const load = async () => {
    const requestSeq = requestSeqRef.current + 1;
    requestSeqRef.current = requestSeq;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`/api/extractions?status=${status}&limit=${PAGE_SIZE}&offset=${offset}`, {
        cache: "no-store",
      });
      const body = (await res.json()) as Extraction[] | { detail?: string };
      if (!res.ok) {
        throw new Error("detail" in body ? (body.detail ?? "Load failed") : "Load failed");
      }
      if (requestSeq !== requestSeqRef.current) return;
      const dedupedMap = new Map<number, Extraction>();
      for (const item of body as Extraction[]) {
        dedupedMap.set(item.id, item);
      }
      const deduped = Array.from(dedupedMap.values()).sort((a, b) => {
        if (a.created_at === b.created_at) return b.id - a.id;
        return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
      });
      setItems(deduped);
    } catch (err) {
      if (requestSeq !== requestSeqRef.current) return;
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      if (requestSeq === requestSeqRef.current) {
        setLoading(false);
      }
    }
  };

  useEffect(() => {
    void load();
  }, [status, offset]);

  const rejectInline = async (extractionId: number) => {
    const reason = window.prompt("Reject reason (optional):", "") ?? "";
    setError(null);
    setMessage(null);
    try {
      const res = await fetch(`/api/extractions/${extractionId}/reject`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reason: reason.trim() || null }),
      });
      const body = (await res.json()) as { detail?: string };
      if (!res.ok) {
        throw new Error(body.detail ?? `Reject failed: ${res.status}`);
      }
      setMessage(`Extraction #${extractionId} rejected.`);
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Reject failed");
    }
  };

  const clearPending = async () => {
    const confirmText = window.prompt(
      "Dangerous operation: this will hard delete pending/failed extractions. Type YES to continue.",
      "",
    );
    if (confirmText !== "YES") return;
    setError(null);
    setMessage(null);
    try {
      const params = new URLSearchParams({
        confirm: "YES",
        also_delete_raw_posts: clearAlsoDeleteRawPosts ? "true" : "false",
      });
      const res = await fetch(`/api/admin/extractions/pending?${params.toString()}`, { method: "DELETE" });
      const body = (await res.json()) as ClearPendingResponse | { detail?: string };
      if (!res.ok) {
        throw new Error("detail" in body ? (body.detail ?? "Clear pending failed") : "Clear pending failed");
      }
      const done = body as ClearPendingResponse;
      setMessage(
        `Deleted extractions=${done.deleted_extractions_count}, raw_posts=${done.deleted_raw_posts_count}.`,
      );
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Clear pending failed");
    }
  };

  return (
    <main style={{ padding: "24px", fontFamily: "monospace" }}>
      <h1>Extractions Review</h1>
      <p>
        默认展示 pending，可切换状态；Approve 进入详情页，Reject 可直接操作。
      </p>

      <div style={{ display: "flex", gap: "8px", alignItems: "center", marginBottom: "12px" }}>
        <label>
          status
          <select
            value={status}
            onChange={(event) => {
              setStatus(event.target.value as Extraction["status"]);
              setOffset(0);
            }}
            style={{ marginLeft: "8px" }}
          >
            <option value="pending">pending</option>
            <option value="approved">approved</option>
            <option value="rejected">rejected</option>
          </select>
        </label>
        <button type="button" onClick={() => void load()} disabled={loading}>
          {loading ? "Loading..." : "Refresh"}
        </button>
        {status === "pending" && (
          <>
            <label>
              <input
                type="checkbox"
                checked={clearAlsoDeleteRawPosts}
                onChange={(event) => setClearAlsoDeleteRawPosts(event.target.checked)}
              />{" "}
              also delete raw posts
            </label>
            <button type="button" onClick={() => void clearPending()} disabled={loading}>
              Clear Pending (Delete)
            </button>
          </>
        )}
      </div>

      {message && <p style={{ color: "green" }}>{message}</p>}
      {error && <p style={{ color: "crimson" }}>{error}</p>}

      {!loading && items.length === 0 && <p>No extractions.</p>}

      <div style={{ display: "grid", gap: "10px" }}>
        {items.map((item) => (
          <article key={item.id} style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "10px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", gap: "8px", marginBottom: "6px" }}>
              <strong>#{item.id} [{item.status}]</strong>
              <small>created_at: {item.created_at}</small>
            </div>
            <div>
              {item.raw_post.platform} / @{item.raw_post.author_handle}
            </div>
            <div>
              <a href={item.raw_post.url} target="_blank" rel="noreferrer">
                {item.raw_post.url}
              </a>
            </div>
            <div>posted_at: {item.raw_post.posted_at}</div>
            <div style={{ marginTop: "6px" }}>summary: {summarizeExtraction(item.extracted_json)}</div>
            <div style={{ display: "flex", gap: "8px", marginTop: "8px" }}>
              <Link href={`/extractions/${item.id}`}>Approve</Link>
              <button type="button" onClick={() => void rejectInline(item.id)} disabled={item.status !== "pending"}>
                Reject
              </button>
            </div>
          </article>
        ))}
      </div>

      <div style={{ display: "flex", gap: "8px", marginTop: "12px" }}>
        <button
          type="button"
          onClick={() => setOffset((prev) => Math.max(0, prev - PAGE_SIZE))}
          disabled={!canPrev || loading}
        >
          Prev
        </button>
        <button type="button" onClick={() => setOffset((prev) => prev + PAGE_SIZE)} disabled={!canNext || loading}>
          Next
        </button>
        <small>offset={offset}</small>
      </div>
    </main>
  );
}
