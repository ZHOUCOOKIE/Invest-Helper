export function getAiUploadCount(stats) {
  if (!stats || typeof stats !== "object") return 0;

  const aiCallUsed = typeof stats.ai_call_used === "number" && Number.isFinite(stats.ai_call_used) ? stats.ai_call_used : null;
  if (aiCallUsed !== null) return Math.max(0, aiCallUsed);

  const attempted =
    typeof stats.openai_call_attempted_count === "number" && Number.isFinite(stats.openai_call_attempted_count)
      ? stats.openai_call_attempted_count
      : null;
  if (attempted !== null) return Math.max(0, attempted);

  const success = typeof stats.success_count === "number" && Number.isFinite(stats.success_count) ? stats.success_count : 0;
  const failed = typeof stats.failed_count === "number" && Number.isFinite(stats.failed_count) ? stats.failed_count : 0;
  return Math.max(0, success + failed);
}

function normalizeNumericIds(ids) {
  if (!Array.isArray(ids)) return [];
  return ids.filter((id) => typeof id === "number" && Number.isFinite(id) && id > 0);
}

export function getExtractTargetIds(stats) {
  if (!stats || typeof stats !== "object") return [];

  const insertedIds = normalizeNumericIds(stats.inserted_raw_post_ids);
  const pendingFailedDedupIds = normalizeNumericIds(stats.pending_failed_dedup_ids);
  if (Array.isArray(stats.inserted_raw_post_ids) || Array.isArray(stats.pending_failed_dedup_ids)) {
    return Array.from(new Set([...insertedIds, ...pendingFailedDedupIds])).sort((a, b) => a - b);
  }

  const importedByHandle = stats.imported_by_handle && typeof stats.imported_by_handle === "object" ? stats.imported_by_handle : {};
  return Array.from(
    new Set(
      Object.values(importedByHandle).flatMap((item) =>
        item && typeof item === "object" ? normalizeNumericIds(item.raw_post_ids) : [],
      ),
    ),
  ).sort((a, b) => a - b);
}
