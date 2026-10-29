"""Agent 层：把「判定 → 改写 → 复检」从单次调用升级为带反馈的循环。

## 为什么自己控制循环，不用 LangChain 的 AgentExecutor

这个流程是**确定性**的：扫词表 → 有命中就改写 → 复检 → 仍有残留就带着
「上一轮残留了哪些词」再改一次。工具选择没有歧义，交给模型自主决策只会
带来不可控（不知道它会调什么）和不可测（没法断言每轮做了什么）。

所以这里只借用 LangChain 的 Tool 抽象来表达「有哪些工具可用」，
控制流和终止条件自己写。面试时可以正面回答「为什么不全交给 Agent」。

## 循环结构

    round 1: scan(原文)   → 命中 → rewrite → 复检
    round 2: scan(改写稿) → 命中 → rewrite（带上轮残留作反馈）→ 复检
    round N: 零残留 → 通过；或达到上限 → 转人工

每轮都落 RoundLog，形成可观测的 trace——这既是界面上「第几次才改干净」的
数据来源，也是面试里「Agent 观测度量」的证据。
"""

import json
import time
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.tools import tool

from .config import LLM_TIMEOUT, MAX_REWRITE_ROUNDS, REQUEST_BUDGET
from .rewrite import rewrite
from .rules import scan_rules


# ---------------- LangChain 工具 ----------------
# 用 tool 装饰器声明，循环里通过 .invoke() 调用，是真在链路里的工具，
# 不是摆设。返回值统一 JSON 字符串，便于外部（API / 界面）直接消费。


@tool
def scan_violations(text: str) -> str:
    """扫描一段文案中的违规表述，返回 JSON 数组；空数组表示未命中。"""
    return json.dumps(scan_rules(text), ensure_ascii=False)


@tool
def rewrite_violations(text: str, words: str) -> str:
    """把违规文案改写为合规版本。

    words 为必须处理的具体表述，用顿号分隔。
    返回 JSON：{"ok": bool, "text": str, "changes": [...], "error": str}
    """
    hits = [
        {"category": "违规表述", "match": w.strip(), "evidence": w.strip()}
        for w in words.replace("，", "、").split("、")
        if w.strip()
    ]
    out = rewrite(text, ["违规表述"], hits)
    return json.dumps(
        {"ok": out.ok, "text": out.text, "changes": out.changes, "error": out.error},
        ensure_ascii=False,
    )


def _scan(text: str) -> List[dict]:
    return json.loads(scan_violations.invoke({"text": text}))


# ---------------- 数据结构 ----------------


@dataclass
class RoundLog:
    """一轮的完整记录。"""

    round: int
    hits: List[dict] = field(default_factory=list)      # 本轮输入时的命中
    rewritten: str = ""                                  # 本轮改写结果
    changes: List[str] = field(default_factory=list)     # 修改说明
    residual: List[dict] = field(default_factory=list)   # 本轮改写后仍残留
    ok: bool = False                                     # 本轮改写是否成功执行
    error: str = ""


@dataclass
class AgentTrace:
    """整条循环的可观测记录。"""

    rounds: List[RoundLog] = field(default_factory=list)
    final_text: str = ""
    resolved: bool = False     # 最终零残留
    escalated: bool = False    # 需要人工介入
    reason: str = ""           # 终止原因

    @property
    def attempts(self) -> int:
        """实际发生改写的轮次数。"""
        return sum(1 for r in self.rounds if r.rewritten)

    @property
    def first_hit_count(self) -> int:
        return len(self.rounds[0].hits) if self.rounds else 0

    def summary(self) -> str:
        """一句话结论，供界面和 API 直接展示。"""
        if self.resolved:
            if self.attempts == 0:
                return "原文未命中违规词，无需改写"
            return f"第 {self.attempts} 轮改写后零残留"
        if self.escalated:
            return f"转人工：{self.reason}"
        return "未完成"


# ---------------- 主流程 ----------------


def review_text(
    text: str,
    categories: Optional[List[str]] = None,
    max_rounds: Optional[int] = None,
    initial_violation: bool = False,
    budget: Optional[float] = None,
) -> AgentTrace:
    """对一段文案执行「判定 → 改写 → 复检」循环。

    initial_violation=True 表示判定层已判违规、但规则层可能一条都没命中
    （语义变体，如「三天见效，不吃药也能好」）。这种情况必须强制改写一轮，
    否则循环只看规则扫描会直接把违规内容判成「无需改写」放过去。
    """
    limit = max_rounds or MAX_REWRITE_ROUNDS
    deadline = time.monotonic() + (budget if budget is not None else REQUEST_BUDGET)
    trace = AgentTrace()

    if not text or not text.strip():
        trace.final_text = text or ""
        trace.escalated = True
        trace.reason = "文本为空，无法判定"
        return trace

    current = text
    hits = _scan(current)

    for i in range(1, limit + 1):
        # 第 1 轮不看预算（否则什么都没做就退出）；之后每轮先看还剩多少时间，
        # 剩余不足半次调用就别开新一轮了，开了大概率也会超时
        if i > 1:
            remain = deadline - time.monotonic()
            if remain < LLM_TIMEOUT * 0.5:
                trace.final_text = current
                trace.escalated = True
                trace.reason = f"时间预算不足（剩 {max(remain, 0):.0f}s），第 {i} 轮未执行"
                return trace

        need_rewrite = bool(hits) or (i == 1 and initial_violation)
        log = RoundLog(round=i, hits=hits)

        if not need_rewrite:
            # 原文就没命中时才补一条记录；改写后零残留的情况，
            # 上一轮的 residual 已经是 0，再追加空轮次会误导展示
            if not trace.rounds:
                log.ok = True
                trace.rounds.append(log)
            trace.final_text = current
            trace.resolved = True
            trace.reason = "零残留"
            return trace

        cats = categories or sorted({h["category"] for h in hits}) or ["违规表述"]
        out = rewrite(current, cats, hits)
        log.rewritten, log.changes, log.ok, log.error = out.text, out.changes, out.ok, out.error

        if not out.ok:
            # 没真改写（没 Key / 网络失败 / 返回异常），重试没有意义
            log.residual = hits
            trace.rounds.append(log)
            trace.final_text = current
            trace.escalated = True
            trace.reason = f"改写未执行（{out.error or '未知'}）"
            return trace

        current = out.text
        hits = _scan(current)
        log.residual = hits
        trace.rounds.append(log)

    # 跑满上限仍有残留
    trace.final_text = current
    trace.escalated = True
    trace.reason = f"已达最大改写轮次（{limit}），仍残留 {len(hits)} 处"
    return trace
