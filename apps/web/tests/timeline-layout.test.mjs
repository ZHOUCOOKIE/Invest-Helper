import test from "node:test";
import assert from "node:assert/strict";

import { buildTimelineLayout } from "../app/assets/[id]/timeline-layout.js";

test("shrinks visible range when data only exists in recent days", () => {
  const today = new Date().toISOString().slice(0, 10);
  const items = [
    { key: "1", day: today, horizon: "1w", stance: "bull", items: [] },
    { key: "2", day: today, horizon: "1w", stance: "bear", items: [] },
  ];
  const layout = buildTimelineLayout(items, "2000-01-01");
  assert.equal(layout.groups.length, 2);
  assert.notEqual(layout.visualStartLabel, "2000-01-01");
  assert.ok(layout.effectiveWindowDays >= 14);
});

test("clamps edge markers inward to avoid cutoff", () => {
  const today = new Date();
  const day30 = new Date(today.getTime() - 30 * 86_400_000).toISOString().slice(0, 10);
  const day0 = today.toISOString().slice(0, 10);
  const items = [
    { key: "a", day: day30, horizon: "1w", stance: "bull", items: [] },
    { key: "b", day: day0, horizon: "1w", stance: "bear", items: [] },
  ];
  const layout = buildTimelineLayout(items, day30);
  const xs = layout.groups.map((item) => item.xPercent);
  assert.ok(Math.min(...xs) >= 2);
  assert.ok(Math.max(...xs) <= 98);
});

test("splits into two rows only when density is high", () => {
  const today = new Date().toISOString().slice(0, 10);
  const sameDayDense = Array.from({ length: 6 }).map((_, idx) => ({
    key: `k-${idx}`,
    day: today,
    horizon: "1w",
    stance: idx % 2 ? "bull" : "bear",
    items: [],
  }));
  const layout = buildTimelineLayout(sameDayDense, "2026-01-01");
  assert.equal(layout.splitRows, true);
  assert.equal(layout.rows.length, 2);
});
