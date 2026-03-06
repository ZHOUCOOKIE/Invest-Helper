# Prompt 与流程说明（ZH）

## 1. SSOT
- Prompt 单一事实来源：`apps/api/services/prompts/extraction_prompt.py`
- 运行时调用链：
  - 单条模板渲染：`render_prompt_bundle(...)`
  - OpenAI `messages` 仅发送 1 条（`role=user`）
  - 组装入口：`apps/api/services/extraction.py`
- `lang` 结论：当前仅为提示语言字段，不参与业务逻辑/审计/回放，已从模板与 render 参数中移除。

## 2. 新输出协议
模型输出 JSON 顶层需要包含 6 个核心字段：
- `as_of`
- `source_url`
- `islibrary`
- `hasview`
- `asset_views`
- `library_entry`

说明：
- `islibrary` 只能是 `0|1`（int）
- `market` 必须严格取值：`CRYPTO|STOCK|ETF|FOREX|OTHER`（legacy auto 枚举已删除）
- `stance` 必须严格取值：`bull|bear|neutral`
- `horizon` 必须严格取值：`intraday|1w|1m|3m|1y`
- `hasview` 必须是 `0|1`，且最终会按 `asset_views` 是否为空重算
- `asset_views` 项固定：`{symbol,market,stance,horizon,confidence,summary}`
- Prompt 对模型约束为 `confidence>=80`，服务端 normalize 最终保留阈值为 `confidence>=70`
- `summary` 中文约束只检查：
  - `asset_views[*].summary`
  - `library_entry.summary`
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

## 4. Auto-review
- asset 分支：沿用资产视图阈值流程
- `hasview=0` 会自动拒绝
- 自动通过必须满足 `hasview=1` 且走置信度阈值路径（`>=70`）
- 记录 `meta.auto_policy_applied`

## 5. 可追溯性
- 通过 `prompt_hash` + `prompt_version` + `raw_model_output` + `parsed_model_output` 回放与审计。
