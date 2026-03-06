import test from "node:test";
import assert from "node:assert/strict";

import { buildMissingInferenceHints, pickDefaults } from "../app/extractions/[id]/review-inference.js";

test("asset_views present: no missing stance/horizon hint and defaults come from first view", () => {
  const extracted = {
    source_url: "https://x.com/post/1",
    asset_views: [
      {
        symbol: "BTC",
        stance: "bull",
        horizon: "1w",
        confidence: 82,
        summary: "Momentum improving",
      },
    ],
  };

  const hints = buildMissingInferenceHints(extracted);
  assert.deepEqual(hints, []);

  const defaults = pickDefaults(extracted, "https://raw.example/post");
  assert.equal(defaults.stance, "bull");
  assert.equal(defaults.horizon, "1w");
  assert.equal(defaults.confidence, "82");
});

test("asset_views empty: still shows missing stance/horizon hints", () => {
  const extracted = {
    asset_views: [],
    assets: [{ symbol: "BTC" }],
  };

  const hints = buildMissingInferenceHints(extracted);
  assert.deepEqual(hints, ["stance: 模型未判断/信息不足", "horizon: 模型未判断/信息不足"]);
});
