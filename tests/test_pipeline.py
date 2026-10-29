"""主链路测试：无法判定的语义、降级路径、复核原因。

不打网络、不加载 OCR 模型，靠 fake_ocr 和 mock 的 judge 控制输入。
"""

from src.judge import JudgeResult
from src.pipeline import analyze, detect
from src.rewrite import RewriteOutcome


class Test无法判定:
    def test_OCR无文字不判为合规(self, fake_ocr):
        """没有内容 ≠ 内容安全。早期版本这里给 low + 置信度 1.0，是错的。"""
        fake_ocr("")
        a = analyze(b"fake")

        assert a.undetermined is True
        assert a.judgment.risk_level == "unknown"
        assert a.judgment.confidence == 0.0
        assert a.needs_review is True
        assert a.is_violation is False

    def test_无法判定时不做改写(self, fake_ocr):
        fake_ocr("")
        a = analyze(b"fake")
        assert a.rewritten == ""
        assert a.trace is None


class Test降级路径:
    def test_无Key时改写未执行且不报残留(self, fake_ocr, no_api_key):
        """没改写就不该有 residual，否则会被误读成「改写不彻底」。"""
        fake_ocr("三天根治颈椎病，无效退款")
        a = analyze(b"fake")

        assert a.is_violation is True
        assert a.rewrite_ok is False
        assert a.residual_hits == []
        assert "missing_api_key" in a.rewrite_error

    def test_无Key时提示未做语义复核(self, fake_ocr, no_api_key):
        fake_ocr("秋季新款，舒适透气")
        a = analyze(b"fake")

        assert a.is_violation is False
        assert any("仅规则层" in r for r in a.review_reasons)


class Test复核原因:
    def test_规则漏检也要进复核队列(self, fake_ocr, monkeypatch):
        """模型判违规、规则零命中——最该人工看的一类，不能被静默放行。"""
        fake_ocr("三天见效，不吃药也能好")
        monkeypatch.setattr(
            "src.pipeline.judge",
            lambda text, use_llm=True: JudgeResult(
                risk_level="high", categories=["医疗功效"], confidence=0.9, llm_used=True
            ),
        )
        monkeypatch.setattr(
            "src.agent.rewrite",
            lambda *a, **k: RewriteOutcome("三天感受变化", ["改"], ok=True),
        )

        a = analyze(b"fake")
        assert a.is_violation is True
        assert a.rule_hits == []          # 规则确实没命中
        assert a.trace is not None
        assert a.trace.attempts >= 1      # 但被强制改写了

    def test_低置信度违规转人工(self, fake_ocr, monkeypatch):
        fake_ocr("全网最低价")
        monkeypatch.setattr(
            "src.pipeline.judge",
            lambda text, use_llm=True: JudgeResult(
                risk_level="medium", categories=["极限词"], confidence=0.5, llm_used=True
            ),
        )
        a = analyze(b"fake")

        assert a.needs_review is True
        assert any("置信度" in r for r in a.review_reasons)

    def test_OCR质量差会提示(self, fake_ocr, no_api_key):
        fake_ocr("秋季新款", score=0.31)
        a = analyze(b"fake")

        assert a.ocr_score < 0.6
        assert any("OCR" in r for r in a.review_reasons)


class Test合规样本:
    def test_合规时不触发改写(self, fake_ocr, no_api_key):
        fake_ocr("秋季新款，舒适透气，支持七天无理由退换")
        a = analyze(b"fake")

        assert a.is_violation is False
        assert a.rewritten == a.ocr_text
        assert a.trace is None
        assert a.residual_hits == []

    def test_detect不触发改写(self, fake_ocr, no_api_key):
        fake_ocr("三天根治颈椎病")
        a = detect(b"fake")

        assert a.judgment.risk_level == "high"
        assert a.rewritten == ""      # detect 只判定
        assert a.trace is None
