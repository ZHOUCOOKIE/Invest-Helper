import test from "node:test";
import assert from "node:assert/strict";

import { buildTimelineBuckets, buildTimelineLayout } from "../app/assets/[id]/timeline-layout.js";

const NOW_ISO = "2026-02-26T12:00:00Z";

function view(overrides = {}) {
  return {
    id: overrides.id ?? 1,
    kol_id: overrides.kol_id ?? 10,
    kol_display_name: overrides.kol_display_name ?? "Alice",
    kol_handle: overrides.kol_handle ?? "alice",
    stance: overrides.stance ?? "bull",
    horizon: overrides.horizon ?? "1w",
    confidence: overrides.confidence ?? 70,
    summary: overrides.summary ?? "s",
    source_url: overrides.source_url ?? `https://x.com/post/${overrides.id ?? 1}`,
    as_of: overrides.as_of ?? "2026-02-26",
    created_at: overrides.created_at ?? "2026-02-26T10:00:00Z",
    posted_at: overrides.posted_at ?? null,
    extraction_id: overrides.extraction_id ?? null,
  };
}

test("renders fixed 5 rows and includes 1y", () => {
  const { rows } = buildTimelineBuckets(
    [view({ horizon: "1y" }), view({ id: 2, horizon: "1w" }), view({ id: 3, horizon: "3m" })],
    NOW_ISO,
  );
  assert.deepEqual(rows.map((row) => row.horizon), ["intraday", "1w", "1m", "3m", "1y"]);
  assert.equal(rows.some((row) => row.horizon === "1y"), true);
});

test("each horizon keeps full fixed window length", () => {
  const { rows } = buildTimelineBuckets([], NOW_ISO);
  const byHorizon = new Map(rows.map((row) => [row.horizon, row]));
  const toleranceMs = 1_000;
  assert.ok(Math.abs(byHorizon.get("intraday").windowMs - 24 * 3_600_000) <= toleranceMs);
  assert.ok(Math.abs(byHorizon.get("1w").windowMs - 7 * 86_400_000) <= toleranceMs);
  assert.ok(Math.abs(byHorizon.get("1m").windowMs - 30 * 86_400_000) <= toleranceMs);
  assert.ok(Math.abs(byHorizon.get("3m").windowMs - 90 * 86_400_000) <= toleranceMs);
  assert.ok(Math.abs(byHorizon.get("1y").windowMs - 365 * 86_400_000) <= toleranceMs);
});

test("intraday buckets by hour while 1w/1m/3m/1y bucket by day", () => {
  const items = [
    view({ id: 1, horizon: "intraday", posted_at: "2026-02-26T10:10:00Z", as_of: "2026-02-26" }),
    view({ id: 2, horizon: "intraday", posted_at: "2026-02-26T10:55:00Z", as_of: "2026-02-26", stance: "bear" }),
    view({ id: 3, horizon: "1w", as_of: "2026-02-25", posted_at: "2026-02-25T02:00:00Z" }),
    view({ id: 4, horizon: "1w", as_of: "2026-02-25", posted_at: "2026-02-25T08:00:00Z", stance: "neutral" }),
    view({ id: 5, horizon: "1m", as_of: "2026-02-20", posted_at: "2026-02-20T09:00:00Z" }),
    view({ id: 6, horizon: "3m", as_of: "2026-02-01", posted_at: "2026-02-01T09:00:00Z" }),
    view({ id: 7, horizon: "3m", as_of: "2026-02-01", posted_at: "2026-02-01T10:00:00Z", stance: "bear" }),
    view({ id: 8, horizon: "1y", as_of: "2025-03-01", posted_at: "2025-03-01T10:00:00Z" }),
    view({ id: 9, horizon: "1y", as_of: "2025-03-01", posted_at: "2025-03-01T11:00:00Z", stance: "neutral" }),
    view({ id: 10, horizon: "1y", as_of: "2024-02-26", posted_at: "2024-02-26T11:00:00Z" }),
  ];
  const { buckets } = buildTimelineBuckets(items, NOW_ISO);
  const intraday = buckets.filter((bucket) => bucket.horizon === "intraday");
  const week = buckets.filter((bucket) => bucket.horizon === "1w");
  const month = buckets.filter((bucket) => bucket.horizon === "1m");
  const quarter = buckets.filter((bucket) => bucket.horizon === "3m");
  const year = buckets.filter((bucket) => bucket.horizon === "1y");
  assert.equal(intraday.length, 1);
  assert.equal(intraday[0].count, 2);
  assert.equal(week.length, 1);
  assert.equal(week[0].count, 2);
  assert.equal(month.length, 1);
  assert.equal(month[0].count, 1);
  assert.equal(quarter.length, 1);
  assert.equal(quarter[0].count, 2);
  assert.equal(year.length, 1);
  assert.equal(year[0].count, 2);
});

test("3m point radius increases with bucket count", () => {
  const items = [
    view({ id: 1, horizon: "3m", as_of: "2026-02-01", posted_at: "2026-02-01T08:00:00Z" }),
    view({ id: 2, horizon: "3m", as_of: "2026-01-15", posted_at: "2026-01-15T08:00:00Z" }),
    view({ id: 3, horizon: "3m", as_of: "2026-01-15", posted_at: "2026-01-15T09:00:00Z" }),
    view({ id: 4, horizon: "3m", as_of: "2026-01-15", posted_at: "2026-01-15T10:00:00Z" }),
  ];
  const { buckets } = buildTimelineBuckets(items, NOW_ISO);
  const single = buckets.find((bucket) => bucket.horizon === "3m" && bucket.count === 1);
  const dense = buckets.find((bucket) => bucket.horizon === "3m" && bucket.count === 3);
  assert.ok(single);
  assert.ok(dense);
  assert.ok(dense.pointRadius > single.pointRadius);
});

test("edge clamp keeps points inward", () => {
  const items = [
    view({ id: 1, horizon: "1w", as_of: "2026-02-20", posted_at: "2026-02-20T01:00:00Z" }),
    view({ id: 2, horizon: "1w", as_of: "2026-02-26", posted_at: "2026-02-26T10:00:00Z" }),
  ];
  const { buckets } = buildTimelineBuckets(items, NOW_ISO);
  const layout = buildTimelineLayout(buckets, NOW_ISO);
  const weekPoints = layout.points.filter((point) => point.horizon === "1w");
  assert.equal(weekPoints.length, 2);
  assert.ok(Math.min(...weekPoints.map((item) => item.xPercent)) >= 2);
  assert.ok(Math.max(...weekPoints.map((item) => item.xPercent)) <= 98);
});
