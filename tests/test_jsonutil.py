"""JSON 容错解析测试。

模型会以各种方式把 JSON 弄脏：包围栏、前后加说明、字符串里用中文引号。
这些都是业务无关的噪声，不该让整条链路判失败。
"""

import json

import pytest

from src.jsonutil import loads


class Test可解析的情况:
    def test_标准json(self):
        assert loads('{"a": 1}') == {"a": 1}

    def test_markdown围栏(self):
        raw = '```json\n{"rewritten": "文案", "changes": ["改了什么"]}\n```'
        assert loads(raw)["rewritten"] == "文案"

    def test_前后有多余文字(self):
        raw = '好的，结果如下：\n{"rewritten": "文案"}\n希望对你有帮助'
        assert loads(raw)["rewritten"] == "文案"

    def test_字符串内含中文引号(self):
        raw = '{"rewritten": "他说“你好”", "changes": []}'
        assert loads(raw)["rewritten"] == "他说“你好”"

    def test_嵌套对象(self):
        raw = '{"a": {"b": [1, 2, {"c": "}"}]}}'
        assert loads(raw) == {"a": {"b": [1, 2, {"c": "}"}]}}

    def test_字符串里带花括号不误判(self):
        raw = '{"text": "这个 { 不配对", "ok": true}'
        assert loads(raw)["text"] == "这个 { 不配对"


class Test不可解析的情况:
    def test_完全没有json(self):
        with pytest.raises(json.JSONDecodeError):
            loads("这不是 JSON")

    def test_空字符串(self):
        with pytest.raises(json.JSONDecodeError):
            loads("")

    def test_只有左括号未闭合(self):
        with pytest.raises(json.JSONDecodeError):
            loads('{"a": 1')
