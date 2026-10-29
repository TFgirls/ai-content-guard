"""判定层测试：规则与模型的冲突合并、失败降级、字段容错。

不打网络：把 _get_client 换成假客户端，直接喂模型会返回的内容。
"""

import json

import pytest

from src.judge import JudgeResult, judge, merge_levels


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class _Resp:
    def __init__(self, content):
        self.choices = [_Choice(content)]


def make_client(payload=None, error=None):
    """构造一个假客户端：要么返回 payload，要么抛 error。"""

    class _Completions:
        def create(self, **_kw):
            if error is not None:
                raise error
            return _Resp(json.dumps(payload, ensure_ascii=False))

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    return _Client()


@pytest.fixture(autouse=True)
def _enable_llm(monkeypatch):
    monkeypatch.setattr("src.judge.DEEPSEEK_API_KEY", "sk-test")


class Test冲突合并:
    def test_规则high与模型low取规则(self, monkeypatch):
        monkeypatch.setattr(
            "src.judge._get_client",
            lambda: make_client({"risk_level": "low", "categories": [], "reasons": ["看着没问题"], "confidence": 0.95}),
        )
        r = judge("三天根治颈椎病，无效退款")

        assert r.risk_level == "high"        # 不能被模型推翻
        assert r.conflict is True
        assert r.confidence < 0.75           # 冲突时压到阈值以下，强制转人工

    def test_规则low与模型high取模型(self, monkeypatch):
        monkeypatch.setattr(
            "src.judge._get_client",
            lambda: make_client({"risk_level": "high", "categories": ["医疗功效"], "reasons": ["暗示疗效"], "confidence": 0.9}),
        )
        r = judge("三天见效，不吃药也能好")

        assert r.risk_level == "high"
        assert r.conflict is True

    def test_两层一致时无冲突(self, monkeypatch):
        monkeypatch.setattr(
            "src.judge._get_client",
            lambda: make_client({"risk_level": "high", "categories": ["医疗功效"], "reasons": ["疗效宣称"], "confidence": 0.9}),
        )
        r = judge("三天根治颈椎病")

        assert r.conflict is False
        assert r.risk_level == "high"
        assert r.confidence == 0.9

    def test_冲突时置信度不超过0_7(self):
        """直接测合并函数，避免依赖 LLM 输出。"""
        merged, conflict = merge_levels("high", "low")
        assert merged == "high"
        assert conflict is True


class Test失败降级:
    def test_调用异常时退回规则并留痕(self, monkeypatch):
        monkeypatch.setattr("src.judge._get_client", lambda: make_client(error=RuntimeError("网络中断")))
        r = judge("三天根治颈椎病")

        assert r.status == "llm_failed"
        assert r.llm_used is False
        assert "RuntimeError" in r.error
        assert r.risk_level == "high"        # 规则层兜住

    def test_没有Key时走纯规则(self, monkeypatch):
        monkeypatch.setattr("src.judge.DEEPSEEK_API_KEY", "")
        r = judge("三天根治颈椎病")

        assert r.llm_used is False
        assert r.status == "judged"
        assert r.risk_level == "high"

    def test_空文本判为无法判定(self, monkeypatch):
        r = judge("   ")
        assert r.risk_level == "unknown"
        assert r.status == "no_text"
        assert r.confidence == 0.0


class Test字段容错:
    def test_模型把categories写成字符串(self, monkeypatch):
        monkeypatch.setattr(
            "src.judge._get_client",
            lambda: make_client({"risk_level": "high", "categories": "医疗功效", "reasons": "含疗效宣称", "confidence": 0.8}),
        )
        r = judge("三天根治颈椎病")

        assert r.categories == ["医疗功效"]
        assert r.reasons == ["含疗效宣称"]

    def test_置信度越界被钳制(self, monkeypatch):
        monkeypatch.setattr(
            "src.judge._get_client",
            lambda: make_client({"risk_level": "low", "categories": [], "reasons": [], "confidence": 1.8}),
        )
        assert judge("秋季新款").confidence <= 1.0

    def test_置信度非数值归零(self, monkeypatch):
        monkeypatch.setattr(
            "src.judge._get_client",
            lambda: make_client({"risk_level": "low", "categories": [], "reasons": [], "confidence": "高"}),
        )
        assert judge("秋季新款").confidence == 0.0

    def test_类别取规则与模型的并集(self, monkeypatch):
        monkeypatch.setattr(
            "src.judge._get_client",
            lambda: make_client({"risk_level": "high", "categories": ["违规导流"], "reasons": [], "confidence": 0.9}),
        )
        r = judge("三天根治颈椎病，加微信abc123")

        assert "医疗功效" in r.categories     # 规则命中
        assert "违规导流" in r.categories     # 模型补充


def test_JudgeResult默认值互不共享():
    """pydantic 的 default_factory 应保证实例之间不共享同一个 list。"""
    a = JudgeResult()
    b = JudgeResult()
    a.categories.append("x")
    assert b.categories == []
