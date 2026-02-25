import { useMemo, useState } from "react";
import { buildTimelineLayout } from "./timeline-layout";

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

type TimelineGroup = {
  key: string;
  stance: "bull" | "bear" | "neutral";
  horizon: string;
  day: string;
  rowIndex: number;
  xPercent: number;
  stackOffsetPx: number;
  items: TimelineItem[];
};

const HORIZON_ORDER: Array<"intraday" | "1w" | "1m" | "3m" | "1y"> = ["intraday", "1w", "1m", "3m", "1y"];
const HORIZON_RANK = new Map(HORIZON_ORDER.map((value, idx) => [value, idx]));

const stanceStyles = {
  bull: { backgroundColor: "#e8f8ee", borderColor: "#0a7f2e", color: "#0a7f2e", y: -18 },
  bear: { backgroundColor: "#fdecec", borderColor: "#b42318", color: "#b42318", y: 18 },
  neutral: { backgroundColor: "#f4f4f5", borderColor: "#6b7280", color: "#6b7280", y: 0 },
} as const;

function toDisplayName(item: TimelineItem): string {
  if (item.kol_display_name && item.kol_display_name.trim()) return item.kol_display_name.trim();
  if (item.kol_handle && item.kol_handle.trim()) return `@${item.kol_handle.trim()}`;
  return `KOL-${item.kol_id}`;
}

function toAsOfDay(item: TimelineItem): string {
  if (item.as_of) return item.as_of.slice(0, 10);
  if (item.created_at) return item.created_at.slice(0, 10);
  return "";
}

function primaryTimeMs(item: TimelineItem): number {
  if (item.as_of) return new Date(`${item.as_of.slice(0, 10)}T00:00:00Z`).getTime();
  return new Date(item.created_at).getTime();
}

function compareTimelineItems(a: TimelineItem, b: TimelineItem): number {
  const timeDelta = primaryTimeMs(b) - primaryTimeMs(a);
  if (timeDelta !== 0) return timeDelta;
  const confidenceDelta = b.confidence - a.confidence;
  if (confidenceDelta !== 0) return confidenceDelta;
  const horizonDelta =
    (HORIZON_RANK.get(a.horizon as (typeof HORIZON_ORDER)[number]) ?? 999) -
    (HORIZON_RANK.get(b.horizon as (typeof HORIZON_ORDER)[number]) ?? 999);
  if (horizonDelta !== 0) return horizonDelta;
  return b.id - a.id;
}

export function AssetViewsTimeline({ data, loading, error }: Props) {
  const [selectedView, setSelectedView] = useState<TimelineItem | null>(null);
  const [expandedKeys, setExpandedKeys] = useState<Record<string, boolean>>({});

  const prepared = useMemo(() => {
    if (!data) {
      return {
        horizons: [] as string[],
        groups: [] as TimelineGroup[],
        rows: [] as Array<{ key: string; startLabel: string; endLabel: string }>,
        visualStartLabel: "",
        splitRows: false,
        effectiveWindowDays: 0,
      };
    }
    const groupsMap = new Map<string, TimelineGroup>();

    for (const item of data.items) {
      const stance = item.stance === "bull" || item.stance === "bear" ? item.stance : "neutral";
      const horizon = item.horizon;
      const day = toAsOfDay(item);
      const key = `${horizon}|${stance}|${day}`;
      const current = groupsMap.get(key);
      if (current) {
        current.items.push(item);
        continue;
      }
      groupsMap.set(key, { key, stance, horizon, day, xPercent: 0, rowIndex: 0, stackOffsetPx: 0, items: [item] });
    }

    for (const group of groupsMap.values()) {
      group.items.sort(compareTimelineItems);
    }

    const layout = buildTimelineLayout(Array.from(groupsMap.values()), data.since_date);
    const groups = (layout.groups as TimelineGroup[]).sort((a, b) => a.rowIndex - b.rowIndex || a.xPercent - b.xPercent || a.key.localeCompare(b.key));
    const horizonSet = new Set(groups.map((item) => item.horizon));
    const horizons = [
      ...HORIZON_ORDER.filter((horizon) => horizonSet.has(horizon)),
      ...Array.from(horizonSet).filter((horizon) => !HORIZON_ORDER.includes(horizon as (typeof HORIZON_ORDER)[number])),
    ];
    return {
      horizons,
      groups,
      rows: layout.rows,
      visualStartLabel: layout.visualStartLabel,
      splitRows: layout.splitRows,
      effectiveWindowDays: layout.effectiveWindowDays,
    };
  }, [data]);

  return (
    <section style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "12px" }}>
      <h2 style={{ marginTop: 0 }}>近30天观点时间线</h2>
      {loading && <p>Loading timeline...</p>}
      {error && <p style={{ color: "crimson" }}>{error}</p>}
      {!loading && !error && (!data || prepared.groups.length === 0) && <p>近30天暂无观点。</p>}
      {!loading && !error && data && prepared.groups.length > 0 && (
        <div style={{ overflowX: "auto", paddingBottom: "8px" }}>
          <p style={{ marginTop: 0, marginBottom: "8px", fontSize: "12px", color: "#666" }}>
            近30天组件，可视区间 {prepared.visualStartLabel} ~ today（有效窗口 {prepared.effectiveWindowDays} 天）
          </p>
          {prepared.splitRows && prepared.rows.length === 2 && (
            <p style={{ marginTop: 0, marginBottom: "8px", fontSize: "12px", color: "#666" }}>
              高密度模式：row-1 {prepared.rows[0].startLabel} ~ {prepared.rows[0].endLabel}，row-2{" "}
              {prepared.rows[1].startLabel} ~ {prepared.rows[1].endLabel}
            </p>
          )}
          <div style={{ minWidth: "960px", display: "grid", gap: "14px" }}>
            {prepared.horizons.map((horizon) => {
              const horizonGroups = prepared.groups.filter((item) => item.horizon === horizon);
              const rows = prepared.splitRows ? [0, 1] : [0];
              return (
                <div key={horizon} style={{ display: "grid", gap: "10px", borderBottom: "1px dashed #eee", paddingBottom: "8px" }}>
                  <div style={{ fontSize: "12px", color: "#666" }}>{horizon}</div>
                  {rows.map((rowIndex) => {
                    const rowMeta = prepared.rows[rowIndex];
                    const rowGroups = horizonGroups.filter((item) => item.rowIndex === rowIndex);
                    return (
                      <div key={`${horizon}-${rowIndex}`} style={{ position: "relative", height: "84px", overflow: "visible" }}>
                        <div
                          style={{
                            position: "absolute",
                            left: 0,
                            right: 0,
                            top: "38px",
                            borderTop: "2px solid #e5e7eb",
                          }}
                        />
                        {rowGroups.map((group) => {
                          const style = stanceStyles[group.stance];
                          const expanded = Boolean(expandedKeys[group.key]);
                          const visibleItems = expanded ? group.items : group.items.slice(0, 2);
                          return (
                            <div
                              key={group.key}
                              style={{
                                position: "absolute",
                                left: `calc(${group.xPercent}% - 6px)`,
                                top: `calc(38px + ${style.y + group.stackOffsetPx}px)`,
                                transform: "translate(-50%, -50%)",
                                border: `1px solid ${style.borderColor}`,
                                backgroundColor: style.backgroundColor,
                                borderRadius: "10px",
                                padding: "4px 6px",
                                maxWidth: "220px",
                                whiteSpace: "nowrap",
                                overflow: "hidden",
                                textOverflow: "ellipsis",
                              }}
                              title={`${group.day} ${group.stance}`}
                            >
                              <div style={{ display: "flex", gap: "4px", alignItems: "center", flexWrap: "wrap" }}>
                                {visibleItems.map((view) => (
                                  <button
                                    key={view.id}
                                    type="button"
                                    onClick={() => setSelectedView(view)}
                                    style={{
                                      border: "none",
                                      background: "transparent",
                                      color: style.color,
                                      textDecoration: "underline",
                                      cursor: "pointer",
                                      fontSize: "11px",
                                      padding: 0,
                                    }}
                                  >
                                    {toDisplayName(view)}
                                  </button>
                                ))}
                                {group.items.length > 2 && !expanded && (
                                  <button
                                    type="button"
                                    onClick={() => setExpandedKeys((prev) => ({ ...prev, [group.key]: true }))}
                                    style={{
                                      border: "none",
                                      background: "transparent",
                                      color: style.color,
                                      cursor: "pointer",
                                      fontSize: "11px",
                                      padding: 0,
                                    }}
                                  >
                                    +{group.items.length - 2}
                                  </button>
                                )}
                              </div>
                            </div>
                          );
                        })}
                        {rowMeta && (
                          <div style={{ position: "absolute", left: 0, right: 0, top: "54px", display: "flex", justifyContent: "space-between", fontSize: "11px", color: "#666" }}>
                            <span>{rowMeta.startLabel}</span>
                            <span>{rowMeta.endLabel === rowMeta.startLabel ? "today" : rowMeta.endLabel}</span>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              );
            })}
          </div>
        </div>
      )}
      {selectedView && (
        <div
          role="dialog"
          aria-modal="true"
          onClick={() => setSelectedView(null)}
          style={{
            position: "fixed",
            inset: 0,
            backgroundColor: "rgba(0,0,0,0.35)",
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            zIndex: 50,
            padding: "12px",
          }}
        >
          <div
            onClick={(event) => event.stopPropagation()}
            style={{
              width: "min(560px, 100%)",
              borderRadius: "10px",
              backgroundColor: "#fff",
              border: "1px solid #ddd",
              padding: "12px",
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", gap: "8px", alignItems: "center" }}>
              <h3 style={{ margin: 0 }}>{toDisplayName(selectedView)}</h3>
              <button type="button" onClick={() => setSelectedView(null)}>
                关闭
              </button>
            </div>
            <p style={{ marginTop: "10px", marginBottom: "8px" }}>{selectedView.summary || "未提供理由"}</p>
            <div style={{ fontSize: "12px", color: "#444", display: "grid", gap: "4px" }}>
              <span>confidence: {selectedView.confidence}</span>
              <span>horizon: {selectedView.horizon}</span>
              <span>as_of: {selectedView.as_of || "-"}</span>
              <span>
                source_url:{" "}
                {selectedView.source_url ? (
                  <a href={selectedView.source_url} target="_blank" rel="noreferrer">
                    {selectedView.source_url}
                  </a>
                ) : (
                  "N/A"
                )}
              </span>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
