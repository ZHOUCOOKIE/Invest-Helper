# PROMPT & FLOW Snapshot: Reasoning 必须中文

> 目标：只做代码核对快照，说明“reasoning 必须中文”在本仓库中的端到端约束与保障。以下内容全部来自仓库现有代码（未改业务逻辑）。

## A) 端到端链路图（真实调用顺序）

### A1. 单条入口：`POST /raw-posts/{raw_post_id}/extract`

1. API 入口：`apps/api/main.py` -> `extract_raw_post(...)`
2. `extract_raw_post` 调用：`create_pending_extraction(db, raw_post, force_reextract=force)`
3. `create_pending_extraction` 内：
   - `_build_extraction_input(raw_post)`（构造截断后的 extraction_input）
   - `select_extractor(settings)`（选择 `OpenAIExtractor` 或 `DummyExtractor`）
   - `_load_assets_for_prompt` / `_load_aliases_for_prompt` / `_load_known_asset_symbols`
   - `build_extract_prompt(...)`（`apps/api/services/prompts/__init__.py`）
   - `extractor.set_prompt_bundle(prompt_bundle)`
4. Prompt 渲染：`build_extract_prompt -> render_extract_v1_prompt`（`apps/api/services/prompts/extract_v1.py`）
5. 发送模型请求：`OpenAIExtractor.extract -> OpenAIExtractor._call_openai`（`apps/api/services/extraction.py`）
   - `_build_user_prompt(raw_post)` 取 `prompt_bundle.text`
   - 构造 `messages=[{"role":"system"...},{"role":"user"...}]`
   - `httpx.Client().post(f"{self.base_url}/chat/completions", json=request_payload)`
6. 返回解析与 normalize：
   - text_json 模式：`parse_text_json_object(content)`
   - `_coerce_call_result(...)`
   - `normalize_extracted_json(...)`
   - `ExtractionPayload.model_validate(...)`
   - `payload.model_dump(mode="json")`
7. 在 `create_pending_extraction` 中再次统一 normalize：`normalize_extracted_json(..., include_meta=True, ...)`
8. reasoning 中文检测：`_detect_reasoning_language(extracted_json.get("reasoning"))`
9. 若 non_zh 且 extractor 为 OpenAI：
   - `extractor.set_reasoning_language_retry_hint(True)`
   - 再次 `extractor.extract(...)`（最多一次）
   - 再次 `_detect_reasoning_language(...)`
   - 若仍 non_zh：`last_error += "reasoning_language_violation_after_retry"`
10. 元数据组装：`_build_extraction_meta(...)` 写入 `extracted_json["meta"]`
11. 落库对象：`PostExtraction(... extracted_json=..., last_error=...)`
12. `db.add(extraction)` -> `await db.flush()`（`create_pending_extraction` 内）
13. 回到入口 `extract_raw_post`：`await db.commit()`

### A2. 批量入口：`POST /raw-posts/extract-batch`

1. API 入口：`extract_raw_posts_batch(...)` -> `_extract_raw_posts_batch_core(payload, db)`
2. `_extract_raw_posts_batch_core` 内每条 `_process_one`：调用 `create_pending_extraction(...)`
3. `_process_one` 成功后：`await local_db.commit()`
4. `create_pending_extraction` 内部流程与 A1 第 3-12 步相同（包含中文检测与一次纠正重试）
5. 若异常且非可重试/重试耗尽：后续 `_create_failed_extraction(...)` 写失败记录，`meta.parse_error=True`，再 `flush/commit`

### A3. 异步任务入口：`POST /extract-jobs`

1. API 入口：`create_extract_job(...)` 创建任务
2. 后台任务：`_run_extract_job(job_id)`
3. `_run_extract_job` 分片调用 `_extract_raw_posts_batch_core(...)`
4. 每条最终仍走 `create_pending_extraction(...)`（即复用 A1 全链路，包括 reasoning 中文检测/一次重试/落库）

### A4. 导入触发入口：`POST /ingest/x/import?trigger_extraction=true`

1. API 入口：`import_x_posts(..., trigger_extraction=True, ...)`
2. 对目标 raw_post 循环调用：`create_pending_extraction(db, raw_post)`
3. 循环结束统一：`await db.commit()`
4. 每条 extraction 内部仍复用 A1 全链路

### A5. 手工导入入口：`POST /ingest/manual`

1. API 入口：`ingest_manual(...)`
2. `await db.flush()` 后调用 `create_pending_extraction(db, raw_post)`
3. `await db.commit()`
4. extraction 内部复用 A1 全链路

---

## B) 提示词快照（原文）

> 下列文本均为仓库现有字符串原文。

### B1. `extract_v1` 模板（用户侧 prompt 文本）

文件：`apps/api/services/prompts/extract_v1.py`，常量 `EXTRACT_V1_TEMPLATE`。

```text
"[InvestPulse Extraction Prompt v1]\n"
"Task: Extract structured investment-view signals from a single post.\n\n"
"Context:\n"
"- platform: {platform}\n"
"- author_handle: {author_handle}\n"
"- url: {url}\n"
"- posted_at: {posted_at}\n"
"- lang: {lang}\n\n"
"Reference Assets (may be empty):\n"
"{assets_block}\n\n"
"Alias -> Symbol Map (may be empty):\n"
"{aliases_block}\n\n"
"Post Content:\n"
"{content_text}\n\n"
"Output contract:\n"
"- Return ONLY one JSON object. No markdown, no code fences, no explanation text.\n"
"- Must include asset_views as primary output for persistence/review.\n"
"- Every asset_views[] item must include symbol/stance/horizon/confidence/reasoning/summary.\n"
"- extracted_json.reasoning must be Chinese. Even if post content is English, explain in Chinese.\n"
"- reasoning may keep necessary proper nouns/tickers/URLs, but sentence body must be Chinese.\n"
"- asset_views[].symbol must prioritize Alias -> Symbol map and Reference Assets symbols.\n"
"- Do not invent new symbols. If uncertain, leave symbol empty and choose closest known symbol when possible.\n"
"- You may output multiple asset views.\n"
"- Keep global fields for compatibility: reasoning, assets, event_tags, stance, horizon, confidence, summary, source_url, as_of.\n"
"- stance in bull/bear/neutral; horizon in intraday/1w/1m/3m/1y; confidence in integer 0-100.\n"
"- Confidence means impact relevance (not truth/probability): 0-30 barely related, 31-60 weak impact, 61-80 direct medium impact, 81-100 strong impact.\n"
"- as_of must be date-only in YYYY-MM-DD (no time).\n"
```

说明：`extract_v1` 在本仓库里作为 `user` prompt 被发送（见 `OpenAIExtractor._build_user_prompt`）。

### B2. 服务端 system prompt 相关原文（由 `_call_openai` 组装）

文件：`apps/api/services/extraction.py`，函数 `OpenAIExtractor._call_openai`。

`strict_json_hint`：

```text
"Return exactly one JSON object. "
"Do not include markdown fences, explanations, or any extra text."
```

`reasoning_language_rule`：

```text
"extracted_json.reasoning must be written in Chinese. "
"Even when the input post is English, explain in Chinese. "
"You may keep proper nouns/tickers/URLs, but sentence body must be Chinese."
```

`json_mode_hard_rules`（text_json 模式附加规则）：

```text
"JSON mode hard rules (中英都适用): "
"only return one JSON object. "
"stance must be one of bull/bear/neutral; if uncertain use neutral. "
"horizon must be one of intraday/1w/1m/3m/1y; if uncertain use 1w. "
"confidence should be integer 0-100. "
"include reasoning (1-3 short sentences). "
"include event_tags as string array. "
"assets should be array of objects with symbol/name/market, symbol is required. "
"assets[*].market must be one of CRYPTO/STOCK/ETF/FOREX/OTHER/AUTO. "
"asset_views must be array of per-asset views, each contains "
"symbol/stance/horizon/confidence/reasoning/summary and optional drivers. "
"top-level reasoning field must be Chinese prose (proper nouns/tickers/URLs allowed). "
"as_of must be date-only string in YYYY-MM-DD (no time)."
```

`system_prompt` 组装模板：

```text
"You extract structured investment-view signals from a single post. "
"Only output fields required by schema. "
"If stance/horizon/confidence cannot be inferred, return null. "
"Do not guess symbols; keep assets empty when unknown. "
f"{reasoning_language_rule} "
f"{retry_reasoning_rule if self.reasoning_language_retry_hint else ''} "
f"{strict_json_hint} "
f"{json_mode_hard_rules if response_mode == EXTRACTION_OUTPUT_TEXT_JSON else ''}"
```

### B3. reasoning 中文约束原文所在段落

1. `extract_v1` user 模板 `Output contract` 段：

```text
"- extracted_json.reasoning must be Chinese. Even if post content is English, explain in Chinese.\n"
"- reasoning may keep necessary proper nouns/tickers/URLs, but sentence body must be Chinese.\n"
```

2. `_call_openai` 的 `reasoning_language_rule` 段：

```text
"extracted_json.reasoning must be written in Chinese. "
"Even when the input post is English, explain in Chinese. "
"You may keep proper nouns/tickers/URLs, but sentence body must be Chinese."
```

3. text_json 附加规则 `json_mode_hard_rules` 里再次强调：

```text
"top-level reasoning field must be Chinese prose (proper nouns/tickers/URLs allowed). "
```

### B4. “纠正重试”专用提示语原文

文件：`apps/api/services/extraction.py`，`retry_reasoning_rule`：

```text
"Correction retry: previous output had non-Chinese reasoning. "
"Now strictly ensure extracted_json.reasoning is Chinese narrative."
```

触发方式：`create_pending_extraction` 在首次检测 non_zh 后调用 `extractor.set_reasoning_language_retry_hint(True)`，随后第二次 `extractor.extract(...)` 时 `system_prompt` 会拼上该段；`finally` 中再置回 `False`。

---

## C) 最终发送给 OpenRouter 的 payload（代码推导结构）

来源函数：`apps/api/services/extraction.py` -> `OpenAIExtractor._call_openai`

```json
{
  "model": "<settings.openai_model>",
  "temperature": 0,
  "max_tokens": "<settings.openai_max_output_tokens>",
  "messages": [
    {
      "role": "system",
      "content": "<system_prompt>"
    },
    {
      "role": "user",
      "content": "<user_prompt>"
    }
  ]
}
```

`messages` 的来源标注：

1. `messages[0].content`（system）：`_call_openai` 内 `system_prompt`，由 `reasoning_language_rule` + `strict_json_hint` +（条件）`retry_reasoning_rule` +（text_json 条件）`json_mode_hard_rules` 拼接。
2. `messages[1].content`（user）：`_build_user_prompt(raw_post)`；优先使用 `extractor.set_prompt_bundle(...)` 注入的 `PromptBundle.text`，该文本来自 `build_extract_prompt -> render_extract_v1_prompt`。

`response_format/json_schema`：

1. 仅当 `response_mode == "structured"` 时设置：
   - `request_payload["response_format"] = {"type": "json_schema", "json_schema": {...}}`
2. `text_json` 模式下不设置 `response_format`（即不传 json_schema）

请求 URL：

1. `f"{self.base_url}/chat/completions"`
2. `self.base_url` 来自 `OpenAIExtractor.__init__(base_url=...)`
3. `base_url` 由 `select_extractor(settings)` 传入 `settings.openai_base_url`

模型/provider/output_mode 来源：

1. 模型名：`settings.openai_model` -> `OpenAIExtractor.model_name`
2. `base_url`：`settings.openai_base_url` -> `OpenAIExtractor.base_url`
3. `provider_detected`：`detect_provider_from_base_url(self.base_url)`（`openrouter.ai` => `openrouter`）
4. `output_mode`：`resolve_extraction_output_mode(self.base_url)`（OpenRouter => `text_json`）
5. 在落库 `meta` 中记录：`provider_detected`、`output_mode_used`

---

## D) 服务端保障策略（按代码判断条件）

### D1. reasoning 语言检测触发条件与 non_zh 判定

函数：`apps/api/main.py` -> `_detect_reasoning_language(reasoning)`。

判定规则：

1. 非字符串或空字符串：返回 `"zh"`
2. `cjk_count = len(_CJK_RE.findall(text))`，`_CJK_RE = re.compile(r"[\u3400-\u9fff]")`
3. 若 `cjk_count >= 2`：返回 `"zh"`
4. `en_word_count = len(_EN_WORD_RE.findall(text))`，`_EN_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")`
5. 若 `cjk_count == 0 and en_word_count >= 6`：返回 `"non_zh"`
6. 若 `cjk_count <= 1 and en_word_count >= 12`：返回 `"non_zh"`
7. 否则返回 `"zh"`

### D2. “最多一次纠正重试”实现位置

函数：`apps/api/main.py` -> `create_pending_extraction(...)`。

关键条件：

1. 仅当 `reasoning_language_violation` 为真（即首次检测 non_zh）
2. 且 `force_fail_reason is None`
3. 且 `isinstance(extractor, OpenAIExtractor)`

执行逻辑：

1. `reasoning_language_retry_used = True`
2. 再次消耗一次调用预算 `try_consume_openai_call_budget(...)`
3. `extractor.set_reasoning_language_retry_hint(True)` 后仅重试一次 `extractor.extract(...)`
4. `finally` 中 `extractor.set_reasoning_language_retry_hint(False)`

无循环结构；因此该纠正重试最多一次。

### D3. 两次都 non_zh 时如何落库

仍在 `create_pending_extraction(...)`：

1. 重试后再次 `reasoning_language = _detect_reasoning_language(...)`
2. 若仍 `"non_zh"`：
   - `last_error` 追加 `"reasoning_language_violation_after_retry"`
3. 然后 `extracted_json["meta"] = _build_extraction_meta(...)`，其中写入：
   - `reasoning_language`
   - `reasoning_language_violation`
   - `reasoning_language_retry_used`
   - 若有错误还会标记 `parse_error`（条件：`"json" in last_error.lower() and "openai" in last_error.lower()`）
4. `PostExtraction(..., extracted_json=..., last_error=last_error)` 后 `db.flush()`；commit 由上层入口执行

### D4. 为什么 failed 不会被 has-result guard 当作“已有可用结果”

相关函数：`apps/api/main.py`

1. `classify_extraction_state(...)`：
   - 只要 `has_last_error` 为真，或 `meta.parse_error`，或 `meta.dummy_fallback`，或 `extractor_name=="dummy"`，即判定 `FAILED`
2. `_has_extracted_result(...)` 仅把 `success/approved/rejected` 当作有结果
3. `_is_result_available_extraction(...)` 直接返回 `_has_extracted_result(...)`
4. 因此：带 `last_error` 的 non_zh 失败记录不会命中“已有可用结果”的 guard

### D5. 继承该行为的入口

凡调用 `create_pending_extraction(...)` 的入口都继承该策略：

1. `POST /raw-posts/{raw_post_id}/extract` -> `extract_raw_post`
2. `POST /ingest/manual` -> `ingest_manual`
3. `POST /ingest/x/import`（`trigger_extraction=true`）-> `import_x_posts`
4. `POST /raw-posts/extract-batch` -> `_extract_raw_posts_batch_core` -> `_process_one`
5. `POST /extract-jobs` -> `_run_extract_job` -> `_extract_raw_posts_batch_core`
6. `POST /ingest/x/retry-failed` -> `retry_failed_x_extractions`

---

## E) 相关文件路径 + HEAD

相关文件：

1. `apps/api/main.py`
2. `apps/api/services/extraction.py`
3. `apps/api/services/prompts/extract_v1.py`
4. `apps/api/services/prompts/__init__.py`
5. `apps/api/settings.py`
6. `apps/api/models.py`

当前 HEAD：

```text
755080e2df3bfc8cf47c4e05746a02fc451fcb17
```

---

## F) 自检命令（只给命令，不执行）

### F1. 文本/函数定位命令

```bash
rg -n "create_pending_extraction|_detect_reasoning_language|_build_extraction_meta|_has_extracted_result|classify_extraction_state" apps/api/main.py
rg -n "@app\.post\(\"/(raw-posts/extract-batch|extract-jobs|ingest/x/import|ingest/manual|raw-posts/.*/extract)" apps/api/main.py
rg -n "EXTRACT_V1_TEMPLATE|render_extract_v1_prompt|build_extract_prompt" apps/api/services/prompts apps/api/main.py
rg -n "class OpenAIExtractor|_call_openai|_build_user_prompt|set_reasoning_language_retry_hint|messages|response_format|json_schema|text_json|reasoning_language_rule|retry_reasoning_rule|json_mode_hard_rules" apps/api/services/extraction.py
rg -n "reasoning_language_violation_after_retry|reasoning_language_retry_used|reasoning_language_violation|reasoning_language" apps/api/main.py apps/api/tests
rg -n "parse_text_json_object|normalize_extracted_json|ExtractionPayload\.model_validate" apps/api/services/extraction.py apps/api/main.py
rg -n "PostExtraction\(|extracted_json=|last_error=|await db\.flush\(|await db\.commit\(" apps/api/main.py
```

### F2. 建议测试命令

```bash
uv run pytest -q tests/test_extractor_openai_and_fallback.py -k "reasoning_non_zh_retries_once"
uv run pytest -q tests/test_extractor_openai_and_fallback.py -k "reasoning_non_zh_twice_marks_failed"
uv run pytest -q tests/test_extract_jobs.py
uv run pytest -q tests/test_dashboard_and_ingest.py -k "extract_batch"
uv run pytest -q
```
