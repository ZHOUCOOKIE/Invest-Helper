from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib

EXTRACT_V1_TEMPLATE = """[InvestPulse 提取提示词]

任务：从一条帖子中提取结构化投资信号。只返回且必须精确返回一个 JSON 对象，不得输出其他任何内容。

输入：

author_handle: {author_handle}

url: {url}

posted_at: {posted_at}

帖子内容：
{content_text}

输出硬性规则（必须遵守）：

输出必须且只能是一个 JSON 对象。
不要使用 markdown。
不要使用代码围栏。
不要输出额外文本。

顶层字段必须且只能是：
as_of, source_url, islibrary, hasview, asset_views, library_entry

固定 JSON 结构：
{
"as_of": "YYYY-MM-DD",
"source_url": "string",
"islibrary": 0,
"hasview": 0,
"asset_views": [
{
"symbol": "string",
"market": "string",
"stance": "bull|bear|neutral",
"horizon": "intraday|1w|1m|3m|1y",
"confidence": 80,
"summary": "string"
}
],
"library_entry": {
  "tag": "macro|industry|thesis|strategy|risk|events",
"summary": "测试"
}
}

字段约束：

as_of：
必须为 YYYY-MM-DD。

source_url：
必须与输入的 url 完全一致。

islibrary：
必须为整数 0 或 1。

hasview：
必须为整数 0 或 1。

market：
必须是 CRYPTO、STOCK、ETF、FOREX、OTHER 之一。

stance：
必须是 bull、bear、neutral 之一。
优先采用作者明确表达的立场。
如果帖子描述的是某个事件/趋势，而你将其映射到一个可能受影响的可交易标的，只有当帖子强烈暗示方向性时才可推断 stance。
如果没有明确的单向预测，则将 stance 设为 neutral。
如果该观点是条件性的，例如“如果 X 那么 Y”、区间规则、阈值规则，除非作者明确表示该条件预计会在所选 horizon 内发生或正在发生，否则将 stance 设为 neutral。

horizon：
必须是 intraday、1w、1m、3m、1y 之一。
优先采用作者明确给出的时间点、持有窗口或预期持续时间。
如果没有明确说明，则选择内容所隐含的最可靠 horizon，即最近的合理窗口。

summary：
总结帖子内容的观点
必须为中文，只校验 summary 的语言。
其中必须包含一个简短的论证理由，例如“事件X → 机制Y → 影响该资产Z”。

confidence：
必须为整数 80..100。
含义：confidence 表示在给定所描述事件/趋势的前提下，你对帖子内容是否会对该资产产生有意义影响，或是否与该资产高度相关，包括合理且直接受影响的目标，的确定程度。

85..100：明显是在谈论一个特定资产 / 直接事件 / 具体主张。

80..85：范围更宽，但仍然很可能对该资产产生有意义的影响。
如果你认为某个资产的 confidence < 80：完全不要输出该资产。

asset_views 输出条件与判定规则：

你可以为明确提及的资产输出 asset_views。
对于未提及的资产，只有当该事件/趋势意味着对该可交易目标存在高度可能且直接的影响，且 confidence >= 80，并且你能够用简短、具体的理由说明该映射时，才可以输出。

直接提及规则（严格）：
只有当帖子文本本身包含该资产的 ticker、name 或 symbol 字符串时，你才可以认定该资产是“明确提及”的。
不要使用作者历史、典型价格水平或你自己的先验知识来判断该资产是什么。

只有当帖子对该特定资产提出了真实的方向性投资主张（明确提出或强烈暗示）时，才设置 hasview=1 并输出 asset_views 条目。

方向性投资主张必须是面向未来的或行动导向的，例如预期、预测、买入计划、卖出计划、持有计划、目标、价位。
仅仅分享过去收益或单纯炒作，不构成主张。

不要将以下内容视为方向性投资主张：
假设性示例；
教学式清单；
说明性内容；
主要是讽刺、道德评判、反问或梗图的帖子。
在所有这些情况下，都将 hasview 设为 0，并输出 asset_views: []。

symbol 规则：

中国内地/香港股票：
symbol 必须为完整中文股票名称。

美国/海外股票：
symbol 必须为可交易 ticker。

ETF/指数：
允许使用常见 ticker 或 symbol。

CRYPTO：
使用标准 symbol 或 pair。

FOREX/大宗商品：
优先使用标准代码或可交易 symbol。

示例 symbol：
"贵州茅台", "NVDA", "SPX", "IGV", "BTC", "XAUUSD", "WTI"

Library 分支规则：

如果 islibrary=0：
library_entry 必须为 null。

如果 islibrary=1：

islibrary=1 必须表示该帖子具有很高价值，值得保存以供反复阅读，例如深度洞见、强分析、严密推理链。
请使用你自己的严格判断。

library_entry 必须包含：
tag（只能有 1 个）；
summary（中文）。

tag 必须从以下值中选择：
macro、industry、thesis、strategy、risk、events

重要：
library_entry.summary 必须精确等于 "测试"，且不能有其他任何内容。

现在只输出最终的 JSON 对象。
"""


@dataclass
class PromptBundle:
    version: str
    text: str
    hash: str

def render_prompt_bundle(
    *,
    platform: str,
    author_handle: str,
    url: str,
    posted_at: datetime,
    content_text: str,
) -> PromptBundle:
    prompt_text = (
        EXTRACT_V1_TEMPLATE.replace("{platform}", platform)
        .replace("{author_handle}", author_handle)
        .replace("{url}", url)
        .replace("{posted_at}", posted_at.isoformat())
        .replace("{content_text}", content_text)
    )
    prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
    return PromptBundle(version="extract_v1", text=prompt_text, hash=prompt_hash)
