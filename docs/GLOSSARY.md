# Glossary

Reference

- `Traceability`: 输出结果可以回链到 `source_url`、`raw_posts`、`post_extractions`。
- `Replay`: 已保存结果可按主键或业务键重新读取，而不是现场重算。
- `Extraction`: 单条帖子经过 prompt、模型和 normalize 后形成的 `post_extractions` 记录。
- `KOL View`: 审核后物化到 `kol_views` 的结构化观点。
- `Review`: 对 extraction 执行 approve、approve-batch、reject、re-extract 的生命周期。
- `Daily Digest`: 按 `(profile_id, digest_date)` 覆盖生成并回放的日报。
- `Weekly Digest`: 按 `(profile_id, report_kind, anchor_date)` 覆盖生成并回放的周报。
- `Anchor Date`: weekly digest 当前回放键中的锚点日期；语义依赖 `kind`。
- `Recent Week`: reference date 往前 6 天加当天。
- `This Week`: 从最近一个周日到 reference date。
- `Last Week`: reference date 所在周之前的完整周日到周六。
- `Profile`: 用户配置集合，包含 KOL 权重和 market 过滤规则；当前公开 API 未暴露 profile 管理路由。
- `Portfolio Advice`: `POST /portfolio/advice` 的请求级组合建议结果，当前不持久化回放。
- `Failed Semantics`: 数据库状态可能仍是 `pending`，但业务上已被重试/进度逻辑视为失败的 extraction。
- `Authoritative`: 在文档集合中作为最终事实来源维护的文件。
- `Not Implemented`: 代码中未实现，文档必须显式标注未实现的能力。
