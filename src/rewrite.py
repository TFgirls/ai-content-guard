"""改写层：把违规文案改成合规版本。

两点改进（对应原来的两个坑）：

1. 原实现只把「违规类别」丢给模型，模型不知道到底哪几个词犯规，
   只能靠猜——这也是「改写后仍有残留」的主要来源。现在把命中的具体词一并传入。
2. 原实现失败时返回原文，且调用方无法区分「没改」和「改得干净」。
   现在返回 RewriteOutcome，带 ok/error，让上层能正确展示。
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import List

from openai import OpenAI
from pydantic import BaseModel, Field, field_validator

from .config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from .jsonutil import loads
from .logutil import warn_once

logger = logging.getLogger(__name__)


class RewriteResult(BaseModel):
    rewritten: str = ""
    changes: List[str] = Field(default_factory=list)

    @field_validator("rewritten", mode="before")
    @classmethod
    def _coerce_text(cls, v):
        if isinstance(v, (list, tuple)):
            return "\n".join(str(x) for x in v)
        return "" if v is None else str(v)

    @field_validator("changes", mode="before")
    @classmethod
    def _coerce_changes(cls, v):
        """模型经常把 changes 写成一段话而不是数组。

        以前这里严格按 List[str] 校验，模型一「偷懒」整条改写就作废
        （实测 10 条违规里 3 条因此没改写），所以要宽容转换。
        """
        if v is None:
            return []
        if isinstance(v, str):
            return [x.strip() for x in re.split(r"[；;\n]", v) if x.strip()]
        if isinstance(v, (list, tuple)):
            out = []
            for x in v:
                if isinstance(x, dict):
                    out.append("；".join(str(k) for k in x.values()))
                elif str(x).strip():
                    out.append(str(x).strip())
            return out
        return [str(v)]


@dataclass
class RewriteOutcome:
    text: str
    changes: List[str] = field(default_factory=list)
    ok: bool = False
    error: str = ""


_PROMPT = """下面是一段电商/广告文案，审核判定存在违规风险。

违规类别：{cats}
必须处理的具体表述：{words}

原文：
{text}

请改写为合规版本：保留营销意图和口语感，删除或替换上面列出的违规表述；
不得虚构资质、疗效或承诺；不要添加原文没有的信息。
注意原文由 OCR 识别，可能含换行或错字，改写时按通顺语句还原。
只输出 JSON：
{{"rewritten": "改写后的完整文案", "changes": ["改了什么，一句话"]}}"""

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL, timeout=40)
    return _client


def rewrite(text: str, categories: List[str], hits: List[dict] | None = None) -> RewriteOutcome:
    """把违规文案改写为合规版本。"""
    if not DEEPSEEK_API_KEY:
        return RewriteOutcome(text, ["未配置 DEEPSEEK_API_KEY，已跳过改写"], ok=False, error="missing_api_key")

    words = [h.get("evidence") or h["match"] for h in (hits or [])]
    last_err: Exception | None = None

    # 模型偶发返回畸形 JSON（实测约 1/10 概率），重试一次通常就能恢复。
    # 只对解析失败重试；网络错误重试意义不大，直接降级。
    for attempt in (1, 2):
        try:
            return _call_once(text, categories, words)
        except json.JSONDecodeError as e:
            last_err = e
            logger.warning("改写返回非 JSON（第 %d 次尝试）：%s", attempt, e)
            continue
        except Exception as e:  # 网络/配额/鉴权失败
            warn_once(logger, f"rewrite:{type(e).__name__}", "改写调用失败：%s: %s", type(e).__name__, e)
            return RewriteOutcome(
                text,
                [f"改写调用失败（{type(e).__name__}），返回原文"],
                ok=False,
                error=f"{type(e).__name__}: {e}",
            )

    return RewriteOutcome(
        text,
        ["改写返回内容无法解析，返回原文"],
        ok=False,
        error=f"JSONDecodeError: {last_err}",
    )


def _call_once(text: str, categories: List[str], words: List[str]) -> RewriteOutcome:
    """单次调用并解析。解析失败抛 JSONDecodeError，交给上层决定是否重试。"""
    resp = _get_client().chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {
                "role": "user",
                "content": _PROMPT.format(
                    text=text,
                    cats="、".join(categories) or "违规表述",
                    words="、".join(words) or "（未给出具体词，请自行判断）",
                ),
            }
        ],
        temperature=0.3,
        response_format={"type": "json_object"},
    )
    data = loads(resp.choices[0].message.content)
    result = RewriteResult(**data)
    out = (result.rewritten or "").strip()

    if not out:
        return RewriteOutcome(text, ["改写结果为空，保留原文"], ok=False, error="empty_result")

    # 长度约束：改写把 100 字变成 300 字或砍成 10 字，都视为异常
    ratio = len(out) / max(len(text), 1)
    if not (0.3 <= ratio <= 3.0):
        logger.warning("改写长度异常：原文 %d 字 -> 改写 %d 字", len(text), len(out))
        return RewriteOutcome(
            out, result.changes + [f"改写长度异常（{ratio:.1f} 倍），请人工确认"], ok=True
        )

    return RewriteOutcome(out, result.changes, ok=True)
