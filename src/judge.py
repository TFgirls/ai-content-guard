"""判定层：规则 + LLM 双层。

这里有两个容易写错、但直接影响结论对错的地方：

1. **冲突合并**：规则层是确定性命中（"根治"就是"根治"），LLM 偶尔会"讲道理"
   判成 low。若直接采用 LLM 结果，等于让不确定的一方推翻确定的一方。
   因此取两者中更严格的一级，并标记 conflict 交人工确认。
2. **失败可见**：原来 except 全静默，只留一个布尔值，出问题无从排查。
   现在记录 status / error 并写日志。
"""

import json
import logging
import re
from typing import List

from openai import OpenAI
from pydantic import BaseModel, Field, field_validator

from .config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from .jsonutil import loads
from .rules import RISK_ORDER, rule_risk, scan_rules

logger = logging.getLogger(__name__)

# 规则模式下的「把握度」：确定性词表命中给得高，未命中给得低——
# 未命中不代表安全，只代表规则层没话可说。
_RULE_CONFIDENCE = {"high": 0.90, "medium": 0.80, "low": 0.40}


class JudgeResult(BaseModel):
    risk_level: str = "low"  # low | medium | high | unknown
    categories: List[str] = Field(default_factory=list)
    reasons: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    llm_used: bool = False
    # judged=正常判定 / no_text=OCR 无文字 / llm_failed=LLM 调用失败退回规则
    status: str = "judged"
    # 规则与 LLM 结论不一致时为 True，需人工确认
    conflict: bool = False
    error: str = ""

    @field_validator("categories", "reasons", mode="before")
    @classmethod
    def _coerce_list(cls, v):
        """模型可能返回单个字符串而非数组，宽容处理。"""
        if v is None:
            return []
        if isinstance(v, str):
            return [x.strip() for x in re.split(r"[；;\n]", v) if x.strip()]
        if isinstance(v, (list, tuple)):
            return [str(x).strip() for x in v if str(x).strip()]
        return [str(v)]

    @field_validator("confidence", mode="before")
    @classmethod
    def _coerce_conf(cls, v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0.0
        return min(max(f, 0.0), 1.0)


_SYSTEM = (
    "你是电商与广告内容合规审核员。只依据给定文本判断是否存在违规风险，"
    "关注：极限词、医疗功效宣称、虚假承诺、联系方式导流。"
)

_PROMPT = """待审核文本：
{text}

规则引擎命中项：
{hits}

请输出 JSON，且只输出 JSON：
{{"risk_level": "low|medium|high", "categories": ["类别"], "reasons": ["原因"], "confidence": 0.0}}
判断标准：出现医疗功效/虚假承诺/联系方式导流 -> high；仅极限词 -> medium；无明显问题 -> low。
注意：文本由 OCR 识别，可能存在错字与断行，请按语义理解，不要因为断行就认为没有违规。"""


_client = None


def _get_client() -> OpenAI:
    """复用同一个 client，避免逐条重建连接。"""
    global _client
    if _client is None:
        _client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL, timeout=30)
    return _client


def _rule_only(hits, rule_level: str) -> JudgeResult:
    return JudgeResult(
        risk_level=rule_level,
        categories=sorted({h["category"] for h in hits}),
        reasons=[f"{h['category']}：{h.get('evidence') or h['match']}" for h in hits[:6]],
        confidence=_RULE_CONFIDENCE.get(rule_level, 0.40),
        llm_used=False,
        status="judged",
    )


def merge_levels(rule_level: str, llm_level: str):
    """取更严格的一级，并返回是否冲突。"""
    if RISK_ORDER.get(llm_level, -1) < 0:
        return rule_level, False
    if RISK_ORDER[llm_level] >= RISK_ORDER.get(rule_level, 0):
        return llm_level, llm_level != rule_level
    return rule_level, True


def judge(text: str, use_llm: bool = True) -> JudgeResult:
    """规则 + 大模型双层判定；关闭 LLM 或无 Key 时退回纯规则。"""
    if not text or not text.strip():
        return JudgeResult(
            risk_level="unknown",
            confidence=0.0,
            status="no_text",
            reasons=["OCR 未识别到文字，无法判定合规性"],
        )

    hits = scan_rules(text)
    rule_level = rule_risk(hits)

    if not use_llm or not DEEPSEEK_API_KEY:
        return _rule_only(hits, rule_level)

    try:
        resp = _get_client().chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _PROMPT.format(text=text, hits=json.dumps(hits, ensure_ascii=False))},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        data = loads(resp.choices[0].message.content)
        result = JudgeResult(**data)

        merged, conflict = merge_levels(rule_level, result.risk_level)
        result.risk_level = merged
        result.conflict = conflict
        result.categories = sorted(set(result.categories) | {h["category"] for h in hits})
        if conflict:
            # 两层结论打架，本身就是「不确定」的信号：置信度压到阈值以下强制转人工，
            # 否则会出现「判了高危、置信度 0.95」这种自相矛盾的展示。
            result.confidence = round(
                min(result.confidence, _RULE_CONFIDENCE.get(rule_level, 0.40), 0.70), 2
            )
            result.reasons.append(
                f"规则层判 {rule_level}（命中：{'、'.join(h['match'] for h in hits[:3])}），"
                f"模型判 {data.get('risk_level')}，已按更严格一方处理，置信度下调待人工确认"
            )
        result.llm_used = True
        result.status = "judged"
        return result

    except Exception as e:  # 网络/配额/鉴权/JSON 解析失败都不该阻断主流程，但必须留痕
        logger.warning("LLM 判定失败，退回规则层：%s: %s", type(e).__name__, e)
        fallback = _rule_only(hits, rule_level)
        fallback.status = "llm_failed"
        fallback.error = f"{type(e).__name__}: {e}"
        return fallback
