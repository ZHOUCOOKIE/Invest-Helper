export function normalizeHoldings(value) {
  if (!Array.isArray(value)) return [];
  return value.map((item) => ({
    ...item,
    support_citations: Array.isArray(item?.support_citations) ? item.support_citations : [],
    risk_citations: Array.isArray(item?.risk_citations) ? item.risk_citations : [],
    ai_evaluation: item?.ai_evaluation && typeof item.ai_evaluation === "object" ? item.ai_evaluation : null,
  }));
}

export function readHoldings(storage, key) {
  try {
    const raw = storage.getItem(key);
    if (!raw) return [];
    return normalizeHoldings(JSON.parse(raw));
  } catch {
    return [];
  }
}

export function writeHoldings(storage, key, holdings) {
  storage.setItem(key, JSON.stringify(holdings));
}

export function removeHoldingAndPersist(storage, key, holdings, localId) {
  const nextHoldings = holdings.filter((item) => item.local_id !== localId);
  writeHoldings(storage, key, nextHoldings);
  return nextHoldings;
}
