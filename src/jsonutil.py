"""LLM 输出的 JSON 解析容错。

模型偶发会：
- 把 JSON 包在 ```json 围栏里
- 前后多写一句「好的，结果如下：」
- 字符串里出现未转义的引号，导致严格解析失败

这些都和业务无关，却会让整条链路判失败（实测 10 条里出现过 1 条）。
所以统一走这里：先剥围栏、再试标准解析、最后退回「取第一个平衡花括号块」。
"""

import json
import re

_FENCE_START = re.compile(r"^\s*```[a-zA-Z]*\s*")
_FENCE_END = re.compile(r"\s*```\s*$")


def _strip_fence(text: str) -> str:
    s = _FENCE_START.sub("", text.strip())
    return _FENCE_END.sub("", s).strip()


def _balanced_block(text: str):
    """从第一个 { 开始，找到与之匹配的 }，返回中间那段。"""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def loads(content: str) -> dict:
    """尽可能把模型输出解析成 dict；实在不行才抛 JSONDecodeError。"""
    if not content:
        raise json.JSONDecodeError("empty content", "", 0)

    s = _strip_fence(content)

    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    block = _balanced_block(s)
    if block:
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            # 最后一个手段：把常见的中文引号换成转义后的形式再试一次
            fixed = block.replace("“", '\\"').replace("”", '\\"')
            return json.loads(fixed)

    raise json.JSONDecodeError("no json object found", s[:120], 0)
