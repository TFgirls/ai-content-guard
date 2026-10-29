"""日志工具。

批量评测和并发请求下，同一类错误（比如 Key 失效）会在每次调用时都打一遍
warning，几十条一模一样的堆栈会把真正有用的日志淹掉。这里做一次性告警：
同类错误只在第一次 warning，之后降到 debug。
"""

import logging
from typing import Set

_seen: Set[str] = set()


def warn_once(logger: logging.Logger, key: str, msg: str, *args) -> None:
    """同一 key 只 warning 一次。

    注意：这不是「吞掉错误」——第一次仍然完整可见，而且调用方拿到的
    status / error 字段依旧如实记录。只是不再刷屏。
    """
    if key in _seen:
        logger.debug(msg, *args)
        return
    _seen.add(key)
    logger.warning(msg, *args)


def reset() -> None:
    """测试用：清空已告警记录。"""
    _seen.clear()
