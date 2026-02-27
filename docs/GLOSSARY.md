# Glossary

TL;DR
- Terms here are canonical across docs.
- If another document uses conflicting words, this glossary wins.

- `Daily Digest`: Versioned daily output artifact generated from approved/latest views.
- `Replay`: Retrieve historical digest output by `date/profile/version` or `digest_id`.
- `Traceability`: Ability to map digest/view output back to `source_url`, `raw_posts`, and `post_extractions`.
- `Profile`: User-scoped rule set (`profile_kol_weights`, `profile_markets`) used by dashboard/digest filtering.
- `Version`: Incrementing digest number under same `profile_id + digest_date`.
- `Clarity Ranking`: Dashboard ranking based on direction consistency/divergence and contributor evidence.
- `Asset View`: Structured per-asset stance extracted/approved into `kol_views`.
- `Extraction`: Parsed model output persisted in `post_extractions` before/after review.
- `Review`: Approve/reject/re-extract lifecycle for `post_extractions`.
- `Authoritative Command`: `make verify`.
- `Not Implemented`: Explicit marker for capabilities absent in current code.
