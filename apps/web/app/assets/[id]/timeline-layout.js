const DAY_MS = 86_400_000;
const MIN_WINDOW_DAYS = 14;
const AXIS_PADDING_DAYS = 1;
const EDGE_CLAMP_MIN = 2;
const EDGE_CLAMP_MAX = 98;

function toDayStartUtc(value) {
  const date = value instanceof Date ? value : new Date(value);
  return new Date(Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate()));
}

function addDays(day, days) {
  return new Date(day.getTime() + days * DAY_MS);
}

function diffDays(a, b) {
  return Math.floor((a.getTime() - b.getTime()) / DAY_MS);
}

function toIsoDay(value) {
  return toDayStartUtc(value).toISOString().slice(0, 10);
}

export function buildTimelineLayout(groups, sinceDateIso) {
  const today = toDayStartUtc(new Date());
  const fallbackStart = toDayStartUtc(`${sinceDateIso}T00:00:00Z`);
  const datedGroups = groups
    .map((group) => {
      const day = typeof group.day === "string" && group.day ? group.day : toIsoDay(today);
      return { ...group, dayDate: toDayStartUtc(`${day}T00:00:00Z`) };
    })
    .sort((a, b) => a.dayDate.getTime() - b.dayDate.getTime() || a.key.localeCompare(b.key));
  const earliest = datedGroups[0]?.dayDate ?? today;
  const latest = datedGroups[datedGroups.length - 1]?.dayDate ?? today;
  const dataStart = addDays(earliest, -AXIS_PADDING_DAYS);
  const dataEnd = addDays(latest > today ? latest : today, AXIS_PADDING_DAYS);
  const minWindowStart = addDays(dataEnd, -(MIN_WINDOW_DAYS - 1));
  const visualStart = (dataStart < minWindowStart ? dataStart : minWindowStart) > fallbackStart
    ? (dataStart < minWindowStart ? dataStart : minWindowStart)
    : fallbackStart;
  const visualEnd = dataEnd;
  const effectiveSpanDays = Math.max(1, diffDays(visualEnd, visualStart));
  const effectiveWindowDays = effectiveSpanDays + 1;
  const markerDensity = datedGroups.length / Math.max(1, effectiveWindowDays);
  const countsByDay = new Map();
  for (const group of datedGroups) {
    const key = toIsoDay(group.dayDate);
    countsByDay.set(key, (countsByDay.get(key) ?? 0) + 1);
  }
  const maxPerDay = Math.max(0, ...Array.from(countsByDay.values()));
  const splitRows = markerDensity >= 1.4 || maxPerDay >= 4 || datedGroups.length >= 28;

  const rows = [];
  if (!splitRows) {
    rows.push({ key: "single", start: visualStart, end: visualEnd });
  } else {
    const firstSpan = Math.floor(effectiveSpanDays / 2);
    const firstEnd = addDays(visualStart, firstSpan);
    const secondStart = addDays(firstEnd, 1);
    rows.push({ key: "early", start: visualStart, end: firstEnd });
    rows.push({ key: "recent", start: secondStart <= visualEnd ? secondStart : visualEnd, end: visualEnd });
  }

  const layoutGroups = datedGroups.map((group) => {
    const rowIndex =
      rows.length === 1
        ? 0
        : group.dayDate.getTime() <= rows[0].end.getTime()
          ? 0
          : 1;
    const row = rows[rowIndex];
    const span = Math.max(1, diffDays(row.end, row.start));
    const bounded = Math.max(0, Math.min(span, diffDays(group.dayDate, row.start)));
    const rawPercent = (bounded / span) * 100;
    return {
      ...group,
      rowIndex,
      xPercent: Math.max(EDGE_CLAMP_MIN, Math.min(EDGE_CLAMP_MAX, rawPercent)),
    };
  });

  const stackByKey = new Map();
  const stackedGroups = layoutGroups.map((group) => {
    const stackKey = `${group.horizon}|${group.rowIndex}|${group.day}`;
    const next = stackByKey.get(stackKey) ?? 0;
    stackByKey.set(stackKey, next + 1);
    return {
      ...group,
      stackOffsetPx: next * 8,
    };
  });

  return {
    splitRows,
    rows: rows.map((row) => ({ ...row, startLabel: toIsoDay(row.start), endLabel: toIsoDay(row.end) })),
    visualStartLabel: toIsoDay(visualStart),
    effectiveWindowDays,
    groups: stackedGroups,
  };
}
