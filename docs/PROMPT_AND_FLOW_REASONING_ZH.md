# Prompt 与流程说明（ZH）

## 1. SSOT
- Prompt 单一事实来源：`apps/api/services/prompts/extraction_prompt.py`
- 运行时调用链：
  - 单条模板渲染：`render_prompt_bundle(...)`
  - OpenAI `messages` 仅发送 1 条（`role=user`）
  - 组装入口：`apps/api/services/extraction.py`
- 当前 prompt 正文是中文说明，实际渲染到正文中的输入字段为：`author_handle/url/posted_at/content_text`
- `platform` 参数仍保留在 render 函数签名里，但当前模板正文不再输出 `platform:` 行
- `lang` 结论：当前仅为提示语言字段，不参与业务逻辑/审计/回放，已从模板与 render 参数中移除。

## 2. 新输出协议
模型输出 JSON 顶层需要包含 6 个核心字段：
- `as_of`
- `source_url`
- `islibrary`
- `hasview`
- `asset_views`
- `library_entry`
键顺序约束（读写规范）：
- 顶层：`as_of, source_url, islibrary, hasview, asset_views, library_entry`
- `asset_views[*]`：`symbol, market, stance, horizon, confidence, summary`

说明：
- `islibrary` 只能是 `0|1`（int）
- `market` 必须严格取值：`CRYPTO|STOCK|ETF|FOREX|OTHER`（legacy auto 枚举已删除）
- `stance` 必须严格取值：`bull|bear|neutral`
- `horizon` 必须严格取值：`intraday|1w|1m|3m|1y`
- `hasview` 必须是 `0|1`，且最终会按 `asset_views` 是否为空重算
- `asset_views` 项固定：`{symbol,market,stance,horizon,confidence,summary}`
- Prompt 对模型约束为 `confidence>=80`，服务端 normalize 最终保留阈值为 `confidence>=70`，auto-review 阈值为 `confidence>=80`
- `summary` 中文约束只检查：
  - `asset_views[*].summary`
  - `library_entry.summary`
- `asset_views[*].summary` 还要求带一个简短理由链，而不是只给结论
- “直接提及”采用严格规则：帖子正文必须出现该资产的 ticker/name/symbol 字符串
- 只有帖子对具体资产形成真实的未来导向/行动导向投资主张时，才允许 `hasview=1`
- `library_entry`:
  - `islibrary=0` 时必须是 `null`
  - `islibrary=1` 时必须是 `{tag,summary}`
  - `tag` 枚举：`macro/industry/thesis/strategy/risk/events`
  - `summary` 必须精确等于 `测试`

## 3. Normalize 策略
- 保留 parse/repair/unwrap 既有能力（BOM/code fence/outer object/最外层 JSON）
- 不再使用 alias map 做 `stance/horizon/market` 关键词/同义词归一化
- `hasview` 由最终 `asset_views` 自动回填（空数组则为 `0`）
- `islibrary=1` 且 `library_entry` 无效时：降级为 `islibrary=0`
- `meta` 现在会在读写归一化时被压缩，只保留影响自动审核/异常解释/回放状态的关键键；如果没有有效键，会直接省略 `meta`

## 4. Auto-review
- asset 分支：沿用资产视图阈值流程
- `hasview=0` 会自动拒绝
- 自动通过必须满足 `hasview=1` 且走置信度阈值路径（`>=80`）
- 记录 `meta.auto_policy_applied`
- 自动拒绝原因键为 `meta.auto_review_reason`（历史 `auto_reject_*` 已清理）
- 用户手动 `re-extract` 在运行时校验通过后，也走同一套标准 auto-review 逻辑

## 4.1 解析失败语义
- 模型返回内容但解析失败时，仍会落一条 extraction（DB `status=pending`）
- 错误信息进入 `last_error` 与精简后的 parse meta
- 运行时分类将其判为 failed 语义（用于进度统计与重试）

## 5. 可追溯性
- 通过 `prompt_hash` + `prompt_version` + `raw_model_output` + `parsed_model_output` 回放与审计。
- `parsed_model_output` 以 DB `JSON` 保序存储（非 `JSONB`）。
