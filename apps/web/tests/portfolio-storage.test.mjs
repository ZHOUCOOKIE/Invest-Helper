import test from "node:test";
import assert from "node:assert/strict";

import { normalizeHoldings, readHoldings, removeHoldingAndPersist } from "../app/portfolio/storage.js";

test("normalizeHoldings fills missing arrays and invalid ai_evaluation", () => {
  const normalized = normalizeHoldings([
    {
      local_id: "1",
      symbol: "BTC",
      support_citations: null,
      risk_citations: undefined,
      ai_evaluation: "invalid",
    },
  ]);

  assert.deepEqual(normalized, [
    {
      local_id: "1",
      symbol: "BTC",
      support_citations: [],
      risk_citations: [],
      ai_evaluation: null,
    },
  ]);
});

test("readHoldings returns normalized rows from storage", () => {
  const storage = {
    getItem(key) {
      assert.equal(key, "portfolio-key");
      return JSON.stringify([{ local_id: "1", symbol: "DOGE", support_citations: null }]);
    },
  };

  const holdings = readHoldings(storage, "portfolio-key");
  assert.equal(holdings.length, 1);
  assert.deepEqual(holdings[0].support_citations, []);
  assert.deepEqual(holdings[0].risk_citations, []);
});

test("removeHoldingAndPersist writes filtered holdings back to storage", () => {
  const writes = [];
  const storage = {
    setItem(key, value) {
      writes.push({ key, value: JSON.parse(value) });
    },
  };
  const holdings = [
    { local_id: "1", symbol: "DOGE" },
    { local_id: "2", symbol: "BTC" },
  ];

  const nextHoldings = removeHoldingAndPersist(storage, "portfolio-key", holdings, "1");

  assert.deepEqual(nextHoldings, [{ local_id: "2", symbol: "BTC" }]);
  assert.deepEqual(writes, [
    {
      key: "portfolio-key",
      value: [{ local_id: "2", symbol: "BTC" }],
    },
  ]);
});
