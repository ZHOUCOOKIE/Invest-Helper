const HOUR_MS = 3_600_000;
const DAY_MS = 86_400_000;
const EDGE_CLAMP_MIN = 2;
const EDGE_CLAMP_MAX = 98;

export const DISPLAY_HORIZONS = ["intraday", "1w", "1m", "3m", "1y"];

const HORIZON_META = {
  intraday: { windowMs: 24 * HOUR_MS, bucket: "hour" },
  "1w": { windowMs: 7 * DAY_MS, bucket: "day" },
  "1m": { windowMs: 30 * DAY_MS, bucket: "day" },
  "3m": { windowMs: 90 * DAY_MS, bucket: "day" },
  "1y": { windowMs: 365 * DAY_MS, bucket: "day" },
};

const HORIZON_INDEX = new Map(DISPLAY_HORIZONS.map((value, idx) => [value, idx]));

function toDate(value) {
  if (value instanceof Date) return value;
  if (typeof value === "string" || typeof value === "number") {
    const parsed = new Date(value);
    if (!Number.isNaN(parsed.getTime())) return parsed;
  }
  return new Date();
}

function toUtcHourStartMs(value) {
  const date = toDate(value);
  return Date.UTC(
    date.getUTCFullYear(),
    date.getUTCMonth(),
    date.getUTCDate(),
    date.getUTCHours(),
    0,
    0,
    0,
  );
}

function toUtcDayStartMs(value) {
  const date = toDate(value);
  return Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate(), 0, 0, 0, 0);
}

function toIsoHour(ms) {
  return new Date(ms).toISOString().slice(0, 13) + ":00Z";
}

function toIsoDay(ms) {
  return new Date(ms).toISOString().slice(0, 10);
}

function toDisplayName(item) {
  if (typeof item.kol_display_name === "string" && item.kol_display_name.trim()) return item.kol_display_name.trim();
  if (typeof item.kol_handle === "string" && item.kol_handle.trim()) return `@${item.kol_handle.trim()}`;
  return `KOL-${item.kol_id ?? "?"}`;
}

function normalizeStance(value) {
  if (value === "bull" || value === "bear") return value;
  return "neutral";
}

function parseAsOfMs(item) {
  if (typeof item.as_of === "string" && item.as_of) {
    const parsed = new Date(`${item.as_of.slice(0, 10)}T00:00:00Z`);
    if (!Number.isNaN(parsed.getTime())) return parsed.getTime();
  }
  return null;
}

function parseIsoMs(value) {
  if (typeof value !== "string" || !value) return null;
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return null;
  return parsed.getTime();
}

function primaryTimeMs(item, fallbackNowMs) {
  const postedAt = parseIsoMs(item.posted_at);
  if (postedAt !== null) return postedAt;
  const asOf = parseAsOfMs(item);
  if (asOf !== null) return asOf;
  const createdAt = parseIsoMs(item.created_at);
  if (createdAt !== null) return createdAt;
  return fallbackNowMs;
}

function compareBucketItems(a, b, fallbackNowMs) {
  const timeDelta = primaryTimeMs(b, fallbackNowMs) - primaryTimeMs(a, fallbackNowMs);
  if (timeDelta !== 0) return timeDelta;
  const confidenceDelta = (b.confidence ?? 0) - (a.confidence ?? 0);
  if (confidenceDelta !== 0) return confidenceDelta;
  return (b.id ?? 0) - (a.id ?? 0);
}

function pointRadiusForCount(horizon, count) {
  const safeCount = Math.max(1, count);
  if (horizon === "3m") {
    return Math.min(15, 4 + Math.round(Math.sqrt(safeCount - 1) * 3));
  }
  return Math.min(12, 4 + Math.round(Math.sqrt(safeCount - 1) * 2));
}

function dominantStance(counts) {
  const ordered = [
    { stance: "bull", count: counts.bull },
    { stance: "bear", count: counts.bear },
    { stance: "neutral", count: counts.neutral },
  ].sort((a, b) => b.count - a.count);
  if (!ordered[0] || ordered[0].count === 0) return "neutral";
  if (ordered[1] && ordered[0].count === ordered[1].count) return "neutral";
  return ordered[0].stance;
}

function topKols(items) {
  const score = new Map();
  for (const item of items) {
    const name = toDisplayName(item);
    const current = score.get(name) ?? { count: 0, confidence: 0 };
    score.set(name, { count: current.count + 1, confidence: current.confidence + (item.confidence ?? 0) });
  }
  return Array.from(score.entries())
    .sort((a, b) => b[1].count - a[1].count || b[1].confidence - a[1].confidence || a[0].localeCompare(b[0]))
    .slice(0, 3)
    .map(([name]) => name);
}

function timelineRows(nowInput) {
  const now = toDate(nowInput);
  const nowMs = now.getTime();
  const todayStartMs = toUtcDayStartMs(now);
  return DISPLAY_HORIZONS.map((horizon) => {
    const config = HORIZON_META[horizon];
    if (horizon === "intraday") {
      const windowEndMs = nowMs;
      const windowStartMs = windowEndMs - config.windowMs;
      return {
        horizon,
        bucketUnit: "hour",
        windowMs: config.windowMs,
        windowStartMs,
        windowEndMs,
        windowStartLabel: toIsoHour(windowStartMs),
        windowEndLabel: toIsoHour(windowEndMs),
      };
    }
    const days = Math.round(config.windowMs / DAY_MS);
    const windowStartMs = todayStartMs - (days - 1) * DAY_MS;
    const windowEndMs = windowStartMs + config.windowMs;
    return {
      horizon,
      bucketUnit: "day",
      windowMs: config.windowMs,
      windowStartMs,
      windowEndMs,
      windowStartLabel: toIsoDay(windowStartMs),
      windowEndLabel: toIsoDay(windowEndMs - DAY_MS),
    };
  });
}

export function buildTimelineBuckets(items, nowInput = new Date()) {
  const rows = timelineRows(nowInput);
  const rowByHorizon = new Map(rows.map((row) => [row.horizon, row]));
  const nowMs = toDate(nowInput).getTime();
  const bucketMap = new Map();

  for (const item of items ?? []) {
    const horizon = typeof item.horizon === "string" ? item.horizon : "";
    if (!rowByHorizon.has(horizon)) continue;
    const row = rowByHorizon.get(horizon);
    const atMs = primaryTimeMs(item, nowMs);
    if (atMs < row.windowStartMs || atMs >= row.windowEndMs) continue;
    const bucketStartMs = row.bucketUnit === "hour" ? toUtcHourStartMs(atMs) : toUtcDayStartMs(atMs);
    const key = `${horizon}|${bucketStartMs}`;
    const stance = normalizeStance(item.stance);
    const current = bucketMap.get(key);
    if (!current) {
      bucketMap.set(key, {
        key,
        horizon,
        bucketStartMs,
        bucketCenterMs: row.bucketUnit === "hour" ? bucketStartMs + HOUR_MS / 2 : bucketStartMs + DAY_MS / 2,
        bucketLabel: row.bucketUnit === "hour" ? toIsoHour(bucketStartMs) : toIsoDay(bucketStartMs),
        bucketUnit: row.bucketUnit,
        counts: { bull: stance === "bull" ? 1 : 0, bear: stance === "bear" ? 1 : 0, neutral: stance === "neutral" ? 1 : 0 },
        totalConfidence: Number(item.confidence ?? 0),
        items: [item],
      });
      continue;
    }
    current.items.push(item);
    current.totalConfidence += Number(item.confidence ?? 0);
    current.counts[stance] += 1;
  }

  const buckets = Array.from(bucketMap.values()).map((bucket) => {
    bucket.items.sort((a, b) => compareBucketItems(a, b, nowMs));
    const count = bucket.items.length;
    return {
      ...bucket,
      count,
      avgConfidence: count > 0 ? Number((bucket.totalConfidence / count).toFixed(1)) : 0,
      dominantStance: dominantStance(bucket.counts),
      topKols: topKols(bucket.items),
      pointRadius: pointRadiusForCount(bucket.horizon, count),
    };
  });

  buckets.sort(
    (a, b) =>
      (HORIZON_INDEX.get(a.horizon) ?? 999) - (HORIZON_INDEX.get(b.horizon) ?? 999) ||
      a.bucketStartMs - b.bucketStartMs ||
      a.key.localeCompare(b.key),
  );

  return { rows, buckets };
}

export function buildTimelineLayout(buckets, nowInput = new Date()) {
  const rows = timelineRows(nowInput).map((row, idx) => ({ ...row, rowIndex: idx }));
  const rowByHorizon = new Map(rows.map((row) => [row.horizon, row]));
  const points = (buckets ?? [])
    .map((bucket) => {
      const row = rowByHorizon.get(bucket.horizon);
      if (!row) return null;
      const spanMs = Math.max(1, row.windowEndMs - row.windowStartMs);
      const rawPercent = ((bucket.bucketCenterMs - row.windowStartMs) / spanMs) * 100;
      const radius = Math.max(3, Number(bucket.pointRadius ?? 4));
      const edgeInset = Math.min(6, Math.max(EDGE_CLAMP_MIN, Math.round(radius * 0.35)));
      return {
        ...bucket,
        rowIndex: row.rowIndex,
        xPercentRaw: rawPercent,
        xPercent: Math.max(edgeInset, Math.min(EDGE_CLAMP_MAX, rawPercent)),
      };
    })
    .filter(Boolean)
    .sort((a, b) => a.rowIndex - b.rowIndex || a.xPercent - b.xPercent || a.key.localeCompare(b.key));
  return { rows, points };
}
