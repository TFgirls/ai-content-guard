"""规则层：文本归一化 + 违禁词表 + 正则。

设计要点（都是踩过的坑）：

1. OCR 输出常带跨行断词（"无效退\\n款"）、字间空格（"无效 退款"）、
   全角符号（"１００％"）。直接拿原文做子串匹配会静默漏检，
   所以先归一化（NFKC 全角转半角 + 去零宽 + 去所有空白）再匹配。
2. 归一化会改变字符位置，因此额外维护「归一化索引 -> 原文字符下标」映射，
   命中项回填原文片段，界面展示不跑偏。
3. 手机号正则加数字边界，避免从订单号/流水号里切出假号码。
4. 词表按文件 mtime 缓存，改了 lexicon.json 立刻生效（原 lru_cache 会锁死）。
"""

import json
import re
import unicodedata
from typing import Dict, List, Tuple

from .config import LEXICON_PATH

# 高风险类别：命中即判 high；极限词属于 medium
HIGH_CATEGORIES = {"医疗功效", "虚假承诺", "违规导流"}
RISK_ORDER = {"low": 0, "medium": 1, "high": 2}

# 空白 + 零宽字符 + 软连字符
_DROP_CHARS = re.compile(r"[\s\u200b-\u200f\u2060\ufeff\u00ad]")

# 正则：统一带 re.I，否则 Vx / 加v 这类大小写变体会漏
_PATTERNS = [
    ("违规导流", re.compile(r"(?:微信|vx|v信)\s*[：:]?\s*[A-Za-z0-9_-]{4,}", re.I), "联系方式导流"),
    ("违规导流", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), "手机号"),
]

_LEX_CACHE: Dict[str, object] = {"mtime": None, "data": {}}


def normalize(text: str) -> Tuple[str, List[int]]:
    """归一化文本，并返回「归一化字符下标 -> 原文下标」映射。

    全角转半角（NFKC）在这里是必须的：OCR 会把 % 识别成全角 ％、
    数字识别成全角 １２３，不转换的话词表里的 "100%" 永远匹配不上。
    """
    if not text:
        return "", []
    chars: List[str] = []
    idx: List[int] = []
    for i, ch in enumerate(text):
        if _DROP_CHARS.match(ch):
            continue
        for c in unicodedata.normalize("NFKC", ch):
            chars.append(c)
            idx.append(i)
    return "".join(chars), idx


def _span(original: str, idx: List[int], start: int, end: int) -> str:
    """把归一化文本上的区间，映射回原文片段。"""
    if not idx or start < 0 or start >= len(idx):
        return ""
    hi = min(end, len(idx)) - 1
    if hi < start:
        return ""
    return original[idx[start] : idx[hi] + 1]


def _norm_word(word: str) -> str:
    return unicodedata.normalize("NFKC", word).strip()


def load_lexicon(force: bool = False) -> Dict[str, List[str]]:
    """读词表，按 mtime 失效缓存。"""
    mtime = LEXICON_PATH.stat().st_mtime if LEXICON_PATH.exists() else None
    if force or _LEX_CACHE["mtime"] != mtime:
        with open(LEXICON_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        _LEX_CACHE["data"] = {
            cat: [w for w in (_norm_word(x) for x in words) if w]
            for cat, words in raw.items()
        }
        _LEX_CACHE["mtime"] = mtime
    return _LEX_CACHE["data"]  # type: ignore[return-value]


def scan_rules(text: str, normalize_text: bool = True) -> List[dict]:
    """规则层扫描，返回命中项。

    normalize_text=False 仅用于评测里做「归一化前后」对比。
    """
    if not text:
        return []

    if normalize_text:
        target, idx = normalize(text)
    else:
        target, idx = text, list(range(len(text)))
    if not target:
        return []

    hits: List[dict] = []
    seen = set()
    # 词表里的英文/拼音缩写（加V、v信）在 OCR 结果里大小写不定，
    # 匹配时统一 casefold；中文不受影响，位置也不偏移。
    target_low = target.casefold()
    keep_pos = len(target_low) == len(target)

    for category, words in load_lexicon().items():
        for w in words:
            pos = target.find(w)
            if pos < 0 and keep_pos:
                wl = w.casefold()
                if wl != w and len(wl) == len(w):
                    pos = target_low.find(wl)
            if pos >= 0 and (category, w) not in seen:
                seen.add((category, w))
                hits.append(
                    {
                        "category": category,
                        "match": w,
                        "evidence": _span(text, idx, pos, pos + len(w)),
                        "reason": f"命中{category}词表",
                    }
                )

    for category, pattern, label in _PATTERNS:
        for m in pattern.finditer(target):
            raw = _span(text, idx, m.start(), m.end())
            key = (category, raw or m.group(0))
            if key in seen:
                continue
            seen.add(key)
            hits.append(
                {
                    "category": category,
                    "match": raw or m.group(0),
                    "evidence": raw or m.group(0),
                    "reason": label,
                }
            )

    return hits


def rule_risk(hits: List[dict]) -> str:
    """仅凭规则给出风险等级：high > medium > low。"""
    cats = {h["category"] for h in hits}
    if cats & HIGH_CATEGORIES:
        return "high"
    if cats:
        return "medium"
    return "low"
