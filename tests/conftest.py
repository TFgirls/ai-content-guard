"""测试公共夹具。

原则：单元测试不打网络、不依赖真实 API Key、不加载 OCR 模型。
超时慢的和不确定的东西用 mock 顶掉，测试要能在任何机器上秒级跑完。
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rewrite import RewriteOutcome  # noqa: E402


@pytest.fixture
def scripted_rewrite(monkeypatch):
    """按脚本返回改写结果，用于构造「第几轮才改干净」的场景。

    用法：
        rewrite, state = scripted_rewrite(["第一轮结果", "第二轮结果"])
    """

    def _make(script):
        state = {"calls": 0}

        def fake(text, categories, hits=None):
            i = min(state["calls"], len(script) - 1)
            state["calls"] += 1
            item = script[i]
            if isinstance(item, RewriteOutcome):
                return item
            return RewriteOutcome(item, [f"第{i + 1}次改写"], ok=True)

        monkeypatch.setattr("src.agent.rewrite", fake)
        return fake, state

    return _make


@pytest.fixture
def fake_ocr(monkeypatch):
    """替换 OCR，返回指定文本，避免加载模型。"""

    def _make(text, score=0.99):
        lines = [{"text": t, "score": score} for t in text.split("\n")] if text else []
        monkeypatch.setattr("src.pipeline.extract_text", lambda _b: (text, lines))
        return text

    return _make


@pytest.fixture
def no_api_key(monkeypatch):
    """模拟没有配置 Key 的降级路径。"""
    monkeypatch.setattr("src.judge.DEEPSEEK_API_KEY", "")
    monkeypatch.setattr("src.rewrite.DEEPSEEK_API_KEY", "")
