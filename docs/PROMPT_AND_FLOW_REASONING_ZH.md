# PROMPT & FLOW Snapshot（方案A已落地）

Reference (ZH)

TL;DR
- 本文记录当前已实现的 prompt 组装、text_json 解析、normalize、重试与 auto-review 规则。
- 所有条目均对应代码事实（`apps/api/main.py`、`apps/api/services/extraction.py`、`apps/api/services/prompts/extract_v1.py`）。

## 1) Prompt 组装

- `provider_detected`：`detect_provider_from_base_url(base_url)`，`openrouter.ai` => `openrouter`。
- `output_mode_used`：`resolve_extraction_output_mode(base_url)`，OpenRouter 强制 `text_json`。
- chat completions payload 固定包含：
  - `model`
  - `temperature=0`
  - `max_tokens=settings.openai_max_output_tokens`
  - `messages=[system,user]`
- `text_json` 模式不携带 `response_format/json_schema`。

## 2) messages 语义

- `messages[0]`（system）：仅放短硬规则（JSON-only、top-level 仅 6 字段、content_kind 分支、`asset_views.confidence>=70`、NoneAny 兜底、禁止 `asset_views[*].drivers/reasoning`）。
- `messages[1]`（user）：`extract_v1` checklist 模板（固定输出结构 + 可交易标的范围 + 自检门槛 + assets/asset_views 一致性 + library 分支）。

## 3) extract_v1 输出字段（已实现）

模型被要求输出的顶层字段：
- `content_kind`
- `as_of`
- `source_url`
- `assets`
- `asset_views`
- `library_entry`

说明：
- system hard rule 将 top-level 固定为：`as_of/source_url/content_kind/assets/asset_views/library_entry`。
- 顶层不允许出现：`stance/horizon/confidence/summary/reasoning/event_tags`。
- `reasoning` 不再要求输出；模型若输出会被忽略。
- `event_tags` 不再要求输出；模型若输出会被忽略。
- `asset_views` 仅保留 `symbol/stance/horizon/confidence/summary`。

## 4) text_json 解析与 meta

`parse_text_json_object` 路径：
- 读 `raw_model_output` 文本
- 去 BOM
- 去 code fence
- 截取最外层 JSON object
- 支持 unwrap `{"extracted_json": {...}}`

可观测 meta：
- `parse_strategy_used`
- `repaired`
- `raw_len`
- `raw_saved_len`
- `raw_truncated`
- `parse_error`
- `parse_error_reason`

## 5) normalize 规则（已实现）

### 5.1 horizon
- 仅接受枚举 `intraday|1w|1m|3m|1y`。
- 非枚举值统一回退 `1w`，并写：
  - `meta.horizon_coerced=true`
  - `meta.horizon_original`
  - `meta.horizon_final=1w`
  - `meta.horizon_coerce_reason=invalid_horizon_enum`

### 5.2 assets / NoneAny / content_kind / library_entry
- `assets` 最终统一为对象数组 `[{symbol,name,market}]`。
- 若模型给字符串数组（如 `["NVDA","CRCL"]`）：
  - 转为对象数组
  - `meta.assets_normalized_from_strings=true`
  - `meta.assets_default_market="AUTO"`
- 空/非法 symbol 会丢弃并记录 drop 计数。
- normalize 后 `assets` 为空则补 `NoneAny` 哨兵对象，并写 `meta.assets_filled_noneany=true`。
- 混合检测（优先于 drop）：
  - 若原始 `assets` 同时包含 `NoneAny` 与其他 symbol
  - `meta.noneany_mixed_with_symbols=true`
  - 上层写 `last_error=noneany_mixed_with_symbols`，归类 failed
- `library_entry` 先于 `content_kind` 分支 normalize：
  - 非法 shape / `confidence<80` / tags 非法（非枚举或长度不在 `1..2`）=> 强制置 `null`
  - 并写 `meta.library_entry_dropped=true` + `meta.library_entry_drop_reason`
- `content_kind` 可观测：
  - `meta.content_kind_raw`（模型原始声明）
  - `meta.content_kind_original`（normalize 前枚举值）
  - 缺失或非法时 `meta.content_kind_defaulted=true`，默认回退 `asset`
- `content_kind="library"` 强制：
  - 必须有有效 `library_entry`（`confidence>=80` + tags 合法），否则降级为 `content_kind="asset"`（`meta.library_downgraded=true`）
  - 降级 reason 仅允许：`low_library_confidence/invalid_library_tags/invalid_library_shape`
  - `assets=[NoneAny 哨兵对象]`
  - `asset_views=[]`
  - 若模型输出了非空 `asset_views`，服务端会清空并写：
    - `meta.library_asset_views_cleared`
    - `meta.library_asset_views_original_count`
    - `meta.library_asset_views_final_count`
- 顶层 `library_tags` 已移除，唯一真相是 `library_entry.tags`。
  - 旧数据若仍含 `library_tags`，读取时忽略，不再写入新 extraction。
  - 新抽取若模型误输出顶层 `library_tags`，normalize 会剥离并写 `meta.library_tags_stripped=true`。
- `content_kind="asset"` 收敛规则：
  - 仅保留 `asset_views.confidence>=70`
  - `assets` 与最终 `asset_views` symbols 同步（去重，顺序跟随 `asset_views`）
  - 若过滤后 `asset_views=[]`，则 `assets=[NoneAny]`

### 5.3 symbol 质量校验

对 `assets[*].symbol` 与 `asset_views[*].symbol` 同步执行：
- trim + 折叠空白
- 长度 `1..30`
- 允许：
  - 英文数字分隔符模式 `^[A-Za-z0-9][A-Za-z0-9._\\-/]{0,29}$`
  - 或 CJK 简称（如 `茅台`）
- 禁止：
  - 控制字符、换行、Tab、零宽字符
  - 结构符号注入（如 `{ } [ ]`）

meta：
- `dropped_invalid_symbol_count`
- `dropped_invalid_symbols_sample`（最多 3）
- `asset_views_dropped_empty_symbol_count`
- `assets_dropped_empty_symbol_count`

### 5.4 symbol 范围与格式（prompt 强约束）
- 仅允许“可落地可交易”标的：指数/ETF/商品/外汇/加密/股票名，且必须被原文显式支持。
- 禁止将宏观变量/主题词当作 symbol（如：流动性、风险偏好、通胀、AI硬件）。
- A股/港股：symbol 使用中文股票名或简称。
- 美股/加密/ETF/指数：symbol 可用 ticker 或英文名（例：`NVDA/Nvidia/BTC/SPX/IGV`）。
- 仅当无法合理映射到上述范围时，才允许 `NoneAny`。

## 6) 失败判定与重试（已实现）

failed 条件（任一成立）：
- `last_error` 非空
- `meta.parse_error=true`
- `meta.parse_error_reason in {"truncated_output","invalid_json"}`

额外重试预算：
- 总额外预算固定 1（`meta.extra_retry_budget_total=1`）
- 截断重试（text_json parse 疑似截断）与 invalid_json 纠错重试二选一
- 或 summary 中文纠正重试
- 三者互斥，不叠加

语言重试失败：
- `last_error` 包含 `summary_language_violation_after_retry`

截断重试失败：
- `last_error` 包含 `parse_error_truncated_output_after_retry`

invalid_json 重试失败：
- `last_error` 包含 `parse_error_invalid_json_after_retry`

invalid_json 专项提示（短）：
- 只输出 1 个合法 JSON object
- 无 markdown
- 字符串必须正确转义，禁止未转义裸引号

## 7) Auto-review（已实现）

执行时机：
- normalize 成功
- 新 `extracted_json` 已写入 extraction 并 `flush`
- 再执行 auto-review

规则：
- `trigger in {"auto","bulk"}` 执行阈值审核
  - `content_kind="asset"`：顶层 `confidence >= 70` => auto approved；`<70` => auto rejected
  - `content_kind="library"`：`library_entry.confidence >= 70` => auto approved；`<70` => auto rejected
- `trigger="user"` 不执行阈值审核
- NoneAny 强规则：
  - `content_kind="asset"` 且 `assets==[NoneAny]` => 强制 auto rejected
  - `content_kind="library"`（仅有效 `library_entry`）不触发 NoneAny 强制拒绝，按阈值（auto/bulk）或 pending（user）处理
- meta 策略标记：
  - `threshold_asset`
  - `threshold_library`
  - `noneany_asset_forced_reject`
  - `no_auto_review_user_trigger`

## 8) 列表收敛

- `GET /extractions` 默认 latest-only（每个 `raw_post` 仅最新一条）
- `show_history=true` 才返回历史版本

## 9) Not Implemented

- Library 独立前端页面（当前仅后端字段与审核语义已落地）。
