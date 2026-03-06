# Glossary

Reference

TL;DR
- 本文件定义跨文档统一术语。
- 若术语冲突，以本文件为准。

- `Daily Digest`: 按 `profile_id + digest_date` 生成并覆盖存储的日度输出。
- `Replay`: 按 `date/profile` 或 `digest_id` 读取历史 digest。
- `Traceability`: 从 digest/view 回链到 `source_url`, `raw_posts`, `post_extractions` 的能力。
- `Profile`: 用户配置集合（如 `profile_kol_weights`, `profile_markets`）。
- `KOL View`: 审核后落库的结构化观点记录（`kol_views`）。
- `Extraction`: 模型抽取结果与元数据记录（`post_extractions`）。
- `Review`: 对 extraction 的 approve/reject/re-extract 生命周期。
- `Clarity Ranking`: dashboard 中基于方向一致性/分歧及证据贡献的排序。
- `Authoritative Command`: `make verify`。
- `Not Implemented`: 代码未实现能力的显式标记。
