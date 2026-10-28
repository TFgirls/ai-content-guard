"""主链路：OCR -> 规则 + LLM 判定 -> Agent 循环改写 -> 改写后复检。"""

from dataclasses import dataclass, field
from typing import List, Optional

from .agent import AgentTrace, review_text
from .config import OCR_LOW_SCORE, REVIEW_CONFIDENCE
from .judge import JudgeResult, judge
from .ocr import extract_text, mean_score
from .rewrite import rewrite
from .rules import scan_rules


@dataclass
class Analysis:
    ocr_text: str = ""
    lines: List[dict] = field(default_factory=list)
    rule_hits: List[dict] = field(default_factory=list)
    judgment: Optional[JudgeResult] = None
    rewritten: str = ""
    changes: List[str] = field(default_factory=list)
    residual_hits: List[dict] = field(default_factory=list)
    # 改写是否真的执行成功；False 时不能把「原文残留」当作「改得不干净」
    rewrite_ok: bool = False
    rewrite_error: str = ""
    # judged / no_text
    status: str = "judged"
    ocr_score: float = 0.0
    # Agent 循环的逐轮记录（use_agent=True 时才有）
    trace: Optional[AgentTrace] = None

    @property
    def undetermined(self) -> bool:
        """无法判定：OCR 没识别到字，或判定层明确返回 unknown。

        这种结果绝不能当作「合规」——「没审出问题」和「没有问题」是两件事。
        """
        if self.status == "no_text":
            return True
        return bool(self.judgment) and self.judgment.risk_level == "unknown"

    @property
    def is_violation(self) -> bool:
        return bool(self.judgment) and self.judgment.risk_level in {"medium", "high"}

    @property
    def review_reasons(self) -> List[str]:
        """需要人工复核的原因列表（空列表 = 可自动放行）。

        原实现只在 is_violation 时才检查置信度，导致「规则漏检」的样本
        （低风险、低置信度）反而被静默放行——恰好是最该看的一类。
        """
        if self.undetermined:
            return ["OCR 未识别到文字，无法判定合规性"]

        j = self.judgment
        if j is None:
            return ["判定未执行"]

        reasons: List[str] = []
        if self.is_violation and j.confidence < REVIEW_CONFIDENCE:
            reasons.append(f"违规判定置信度 {j.confidence:.2f} 低于阈值 {REVIEW_CONFIDENCE}")
        if j.conflict:
            reasons.append("规则层与模型结论不一致，建议人工确认")
        if not j.llm_used:
            reasons.append("本次仅规则层判定，未做语义复核")
        if self.lines and self.ocr_score < OCR_LOW_SCORE:
            reasons.append(f"OCR 平均置信度 {self.ocr_score:.2f} 偏低，文字可能识别有误")
        if self.trace and self.trace.escalated and self.is_violation:
            reasons.append(f"改写未收敛：{self.trace.reason}")
        return reasons

    @property
    def needs_review(self) -> bool:
        return bool(self.review_reasons)


def detect(image_bytes: bytes, use_llm: bool = True) -> Analysis:
    """只做判定，不改写。评测脚本走这条路径，省 LLM 调用。"""
    text, lines = extract_text(image_bytes)
    a = Analysis(
        ocr_text=text,
        lines=lines,
        rule_hits=scan_rules(text),
        ocr_score=mean_score(lines),
    )
    if not text.strip():
        a.status = "no_text"
        a.judgment = JudgeResult(
            risk_level="unknown",
            confidence=0.0,
            status="no_text",
            reasons=["OCR 未识别到文字，无法判定合规性"],
        )
        return a

    a.judgment = judge(text, use_llm=use_llm)
    return a


def analyze(image_bytes: bytes, use_agent: bool = True) -> Analysis:
    """完整链路：OCR -> 判定 -> 改写 -> 复检。

    use_agent=True 走多轮循环（带反馈、可观测）；
    False 退回单轮改写，保留下来用于对比与回退。
    """
    a = detect(image_bytes)
    if a.undetermined:
        return a

    if not a.is_violation:
        a.rewritten, a.changes = a.ocr_text, []
        return a

    if use_agent:
        trace = review_text(
            a.ocr_text,
            a.judgment.categories,
            initial_violation=a.is_violation,
        )
        a.trace = trace
        a.rewritten = trace.final_text
        first = trace.rounds[0] if trace.rounds else None
        a.changes = first.changes if first else []
        a.rewrite_ok = bool(first and first.ok and first.rewritten)
        a.rewrite_error = (first.error if first else "") or ""
        a.residual_hits = trace.rounds[-1].residual if trace.rounds else []
        return a

    out = rewrite(a.ocr_text, a.judgment.categories, a.rule_hits)
    a.rewritten, a.changes = out.text, out.changes
    a.rewrite_ok, a.rewrite_error = out.ok, out.error
    # 只有真的改写过才复检；否则「原文的命中」会被误读成「改写不彻底」
    if out.ok:
        a.residual_hits = scan_rules(a.rewritten)
    return a
