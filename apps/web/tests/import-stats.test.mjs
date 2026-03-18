import test from "node:test";
import assert from "node:assert/strict";

import { getAiUploadCount, getExtractTargetIds } from "../app/ingest/import-stats.js";

test("prefers ai_call_used when present so retries are reflected", () => {
  assert.equal(
    getAiUploadCount({
      ai_call_used: 5,
      openai_call_attempted_count: 4,
      success_count: 3,
      failed_count: 1,
    }),
    5,
  );
});

test("falls back to openai_call_attempted_count when ai_call_used is unavailable", () => {
  assert.equal(
    getAiUploadCount({
      success_count: 2,
      failed_count: 1,
      openai_call_attempted_count: 6,
    }),
    6,
  );
});

test("falls back to success plus failed for legacy stats payloads", () => {
  assert.equal(
    getAiUploadCount({
      success_count: 2,
      failed_count: 1,
    }),
    3,
  );
});

test("extract target ids include new inserts and failed dedup retries only", () => {
  assert.deepEqual(
    getExtractTargetIds({
      inserted_raw_post_ids: [11, 12],
      dedup_existing_raw_post_ids: [41, 42, 43],
      pending_failed_dedup_ids: [42, 90],
      imported_by_handle: {
        alice: { raw_post_ids: [11, 41, 42] },
        bob: { raw_post_ids: [12, 43, 90] },
      },
    }),
    [11, 12, 42, 90],
  );
});

test("extract target ids fall back to imported-by-handle ids for legacy payloads", () => {
  assert.deepEqual(
    getExtractTargetIds({
      imported_by_handle: {
        alice: { raw_post_ids: [9, 4, 9] },
        bob: { raw_post_ids: [7] },
      },
    }),
    [4, 7, 9],
  );
});
