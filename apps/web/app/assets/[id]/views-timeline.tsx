import Link from "next/link";
import { useMemo, useState } from "react";
import { buildTimelineBuckets, buildTimelineLayout } from "./timeline-layout";

type TimelineItem = {
  id: number;
  kol_id: number;
  kol_display_name?: string | null;
  kol_handle?: string | null;
  stance: "bull" | "bear" | "neutral" | string;
  horizon: "intraday" | "1w" | "1m" | "3m" | "1y" | string;
  confidence: number;
  summary: string;
  source_url: string;
  as_of: string;
  created_at: string;
  posted_at?: string | null;
  extraction_id?: number | null;
};

type TimelineResponse = {
  asset_id: number;
  days: number;
  since_date: string;
  generated_at: string;
  items: TimelineItem[];
};

type Props = {
  data: TimelineResponse | null;
  loading: boolean;
  error: string | null;
};

type TimelineRow = {
  horizon: string;
  bucketUnit: string;
  windowStartLabel: string;
  windowEndLabel: string;
};

type TimelinePoint = {
  key: string;
  horizon: string;
  rowIndex: number;
  xPercent: number;
  pointRadius: number;
  bucketLabel: string;
  bucketUnit: string;
  count: number;
  avgConfidence: number;
  counts: { bull: number; bear: number; neutral: number };
  topKols: string[];
  dominantStance: "bull" | "bear" | "neutral" | string;
  items: TimelineItem[];
};

const stanceStyle = {
  bull: { color: "#12803c", fill: "#dff7e8" },
  bear: { color: "#bd1f2d", fill: "#fde7ea" },
  neutral: { color: "#677285", fill: "#eef0f3" },
} as const;

function toDisplayName(item: TimelineItem): string {
  if (item.kol_display_name && item.kol_display_name.trim()) return item.kol_display_name.trim();
  if (item.kol_handle && item.kol_handle.trim()) return `@${item.kol_handle.trim()}`;
  return `KOL-${item.kol_id}`;
}

function formatBucketLabel(bucketLabel: string, bucketUnit: string): string {
  if (bucketUnit === "hour") return bucketLabel.replace("T", " ").replace(":00Z", ":00 UTC");
  return `${bucketLabel} UTC`;
}

function formatRowRange(row: { bucketUnit: string; windowStartLabel: string; windowEndLabel: string }): string {
  if (row.bucketUnit === "hour") {
    return `${row.windowStartLabel.replace("T", " ")} ~ ${row.windowEndLabel.replace("T", " ")}`;
  }
  return `${row.windowStartLabel} ~ ${row.windowEndLabel}`;
}

export function AssetViewsTimeline({ data, loading, error }: Props) {
  const [hoveredKey, setHoveredKey] = useState<string | null>(null);
  const [selectedKey, setSelectedKey] = useState<string | null>(null);

  const prepared = useMemo(() => {
    const baseItems = data?.items ?? [];
    const now = new Date();
    const { rows, buckets } = buildTimelineBuckets(baseItems, now) as { rows: TimelineRow[]; buckets: TimelinePoint[] };
    const layout = buildTimelineLayout(buckets, now) as { points: TimelinePoint[] };
    const pointsByRow = new Map<number, TimelinePoint[]>();
    for (const point of layout.points) {
      const current = pointsByRow.get(point.rowIndex) ?? [];
      current.push(point);
      pointsByRow.set(point.rowIndex, current);
    }
    const pointByKey = new Map<string, TimelinePoint>(layout.points.map((point) => [point.key, point]));
    return { rows, pointsByRow, pointByKey };
  }, [data]);

  const hoveredBucket = hoveredKey ? prepared.pointByKey.get(hoveredKey) ?? null : null;
  const selectedBucket = selectedKey ? prepared.pointByKey.get(selectedKey) ?? null : null;

  return (
    <section style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "12px" }}>
      <h2 style={{ marginTop: 0, marginBottom: "8px" }}>观点时间线（24h / 1w / 1m / 3m / 1y）</h2>
      {loading && <p>Loading timeline...</p>}
      {error && <p style={{ color: "crimson" }}>{error}</p>}
      {!loading && !error && (
        <div style={{ display: "grid", gap: "10px" }}>
          {prepared.rows.map((row, rowIndex) => {
            const rowPoints = prepared.pointsByRow.get(rowIndex) ?? [];
            return (
              <div key={row.horizon} style={{ display: "grid", gap: "4px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: "11px", color: "#5a6473" }}>
                  <strong style={{ fontSize: "12px", color: "#2c3440" }}>{row.horizon}</strong>
                  <span>{formatRowRange(row)}</span>
                </div>
                <div style={{ position: "relative", height: "72px", borderBottom: "1px dashed #e5e7eb" }}>
                  <div
                    style={{
                      position: "absolute",
                      left: 0,
                      right: 0,
                      top: "36px",
                      borderTop: "2px solid #e5e7eb",
                    }}
                  />
                  {rowPoints.map((point) => {
                    const style = stanceStyle[point.dominantStance as keyof typeof stanceStyle] ?? stanceStyle.neutral;
                    return (
                      <button
                        key={point.key}
                        type="button"
                        onMouseEnter={() => setHoveredKey(point.key)}
                        onMouseLeave={() => setHoveredKey((prev) => (prev === point.key ? null : prev))}
                        onFocus={() => setHoveredKey(point.key)}
                        onBlur={() => setHoveredKey((prev) => (prev === point.key ? null : prev))}
                        onClick={() => setSelectedKey(point.key)}
                        style={{
                          position: "absolute",
                          left: `${point.xPercent}%`,
                          top: "36px",
                          transform: "translate(-50%, -50%)",
                          width: `${point.pointRadius * 2}px`,
                          height: `${point.pointRadius * 2}px`,
                          borderRadius: "999px",
                          border: `1px solid ${style.color}`,
                          backgroundColor: style.fill,
                          cursor: "pointer",
                          padding: 0,
                        }}
                        title={`${row.horizon} ${formatBucketLabel(point.bucketLabel, point.bucketUnit)} count=${point.count}`}
                        aria-label={`${row.horizon} point ${point.bucketLabel}`}
                      />
                    );
                  })}
                  {hoveredBucket && hoveredBucket.horizon === row.horizon && (
                    <div
                      style={{
                        position: "absolute",
                        left: `${hoveredBucket.xPercent}%`,
                        top: "4px",
                        transform: "translateX(-50%)",
                        backgroundColor: "#111827",
                        color: "#f3f4f6",
                        borderRadius: "6px",
                        padding: "6px 8px",
                        fontSize: "11px",
                        minWidth: "220px",
                        maxWidth: "280px",
                        pointerEvents: "none",
                        zIndex: 20,
                      }}
                    >
                      <div>{formatBucketLabel(hoveredBucket.bucketLabel, hoveredBucket.bucketUnit)}</div>
                      <div>
                        bull {hoveredBucket.counts.bull} / bear {hoveredBucket.counts.bear} / neutral {hoveredBucket.counts.neutral}
                      </div>
                      <div>total {hoveredBucket.count}, avg confidence {hoveredBucket.avgConfidence}</div>
                      <div>top KOL: {hoveredBucket.topKols.join(", ") || "N/A"}</div>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
          {!data?.items?.length && <p style={{ margin: 0 }}>暂无观点。</p>}
        </div>
      )}
      {selectedBucket && (
        <div
          role="dialog"
          aria-modal="true"
          onClick={() => setSelectedKey(null)}
          style={{
            position: "fixed",
            inset: 0,
            backgroundColor: "rgba(0,0,0,0.35)",
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            zIndex: 60,
            padding: "12px",
          }}
        >
          <div
            onClick={(event) => event.stopPropagation()}
            style={{
              width: "min(840px, 100%)",
              maxHeight: "85vh",
              overflowY: "auto",
              borderRadius: "10px",
              backgroundColor: "#fff",
              border: "1px solid #ddd",
              padding: "12px",
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: "8px", marginBottom: "8px" }}>
              <h3 style={{ margin: 0, fontSize: "16px" }}>
                {selectedBucket.horizon} {formatBucketLabel(selectedBucket.bucketLabel, selectedBucket.bucketUnit)} | {selectedBucket.count} 条
              </h3>
              <button type="button" onClick={() => setSelectedKey(null)}>
                关闭
              </button>
            </div>
            <div style={{ display: "grid", gap: "8px" }}>
              {selectedBucket.items.map((item) => {
                const style = stanceStyle[(item.stance as keyof typeof stanceStyle) ?? "neutral"] ?? stanceStyle.neutral;
                return (
                  <article key={item.id} style={{ border: "1px solid #e5e7eb", borderRadius: "8px", padding: "8px", fontSize: "12px" }}>
                    <div style={{ display: "flex", gap: "8px", flexWrap: "wrap", alignItems: "center" }}>
                      <strong>{toDisplayName(item)}</strong>
                      <span style={{ color: style.color, fontWeight: 700 }}>{item.stance}</span>
                      <span>horizon: {item.horizon}</span>
                      <span>confidence: {item.confidence}</span>
                    </div>
                    <div style={{ marginTop: "4px", color: "#374151" }}>{item.summary?.trim() ? item.summary : "未提供理由"}</div>
                    <div style={{ marginTop: "4px", display: "flex", gap: "10px", flexWrap: "wrap" }}>
                      <span>
                        source:{" "}
                        {item.source_url ? (
                          <a href={item.source_url} target="_blank" rel="noreferrer">
                            {item.source_url}
                          </a>
                        ) : (
                          "N/A"
                        )}
                      </span>
                      {item.extraction_id ? <Link href={`/extractions/${item.extraction_id}`}>查看 extraction #{item.extraction_id}</Link> : <span>extraction: N/A</span>}
                    </div>
                  </article>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
