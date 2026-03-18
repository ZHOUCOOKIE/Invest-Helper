# Prompt 与流程说明（ZH）

## 1. SSOT
- Prompt 单一事实来源：`apps/api/services/prompts/extraction_prompt.py`
- 运行时组装入口：`apps/api/services/extraction.py`
- 当前通过 `render_prompt_bundle(...)` 渲染单条 user message
- 当前 prompt 正文是中文，实际注入字段为：
  - `author_handle`
  - `url`
  - `posted_at`
  - `content_text`
- `platform` 仍在函数签名中，但当前模板正文未输出 `platform:` 行

## 2. 当前输出协议

顶层核心字段固定为：
- `as_of`
- `source_url`
- `islibrary`
- `hasview`
- `asset_views`
- `library_entry`

规范顺序：
- 顶层：`as_of, source_url, islibrary, hasview, asset_views, library_entry`
- `asset_views[*]`：`symbol, market, stance, horizon, confidence, summary`

字段规则：
- `market` 必须严格取值：`CRYPTO|STOCK|ETF|FOREX|OTHER`
- `stance` 必须严格取值：`bull|bear|neutral`
- `horizon` 必须严格取值：`intraday|1w|1m|3m|1y`
- `asset_views[*].summary` 必须是中文，且 prompt 要求带简短理由链
- Prompt 要求模型输出 `confidence >= 80`
- 服务端 normalize 最终保留 `confidence >= 70`
- auto-review 自动通过阈值仍是 `>= 80`
- `hasview` 最终不是信任模型原值，而是按归一化后的 `asset_views` 自动回填

严格语义：
- “直接提及”要求帖子正文本身出现该资产 ticker、name 或 symbol 字符串
- 只有当帖子形成真实的未来导向或行动导向投资主张时，才允许 `hasview=1`
- 教学示例、梗图、讽刺、纯复盘、非投资主张内容应保持 `hasview=0`

## 3. Library 分支现状

- `islibrary=0` 时，`library_entry` 必须为 `null`
- `islibrary=1` 时，`library_entry` 当前必须是 `{tag, summary}`
- `tag` 枚举为：
  - `macro`
  - `industry`
  - `thesis`
  - `strategy`
  - `risk`
  - `events`
- 当前实现仍保留严格测试型约束：`library_entry.summary` 必须精确等于 `测试`
- `islibrary=1` 且 `library_entry` 无效时，不会报错终止，而是自动降级为 `islibrary=0`

## 4. Normalize 策略

- 保留 parse/repair/unwrap 能力，包括 BOM、code fence、外层包裹对象和最外层 JSON 提取
- 不做 `market/stance/horizon` 的同义词映射归一化
- `asset_views` 会经过：
  - 结构筛选
  - symbol 质量校验
  - 置信度筛选
  - 中文 summary 校验
- `meta` 会被压缩，只保留影响自动审核、失败解释和回放状态的关键键
- 没有有效键时，`meta` 会被直接省略

## 5. Auto-review

- `hasview=0` 自动拒绝
- 自动通过要求 `hasview=1` 且满足阈值流程
- 自动审核结果写入 `meta.auto_policy_applied`、`meta.auto_review_reason`、`meta.auto_review_threshold`
- 用户触发的 `re-extract` 在运行时校验通过后，也走同一套标准 auto-review 流程

## 6. 解析失败语义

- 模型返回内容但解析失败时，仍创建 extraction 记录
- 数据库 `status` 可能仍为 `pending`
- 错误信息进入 `last_error` 与 parse 相关 meta
- 进度统计和重试筛选会把这类记录按 failed semantics 处理

## 7. 可追溯性

- 审计链路依赖 `prompt_version`、`prompt_hash`、`prompt_text`、`raw_model_output`、`parsed_model_output`
- `parsed_model_output` 当前以 DB `JSON` 保序存储，不是 `JSONB`
