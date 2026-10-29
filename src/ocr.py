"""OCR 封装：RapidOCR（onnxruntime 后端），无 GPU、无需联网。

除了文字本身，还保留逐行置信度——图片模糊时 OCR 会「猜」出错误的字，
这种情况下判定结果本应降级为人工复核，而不是当作确定性结论。
"""

import io
import threading

import numpy as np
from PIL import Image

_engine = None
# 服务化之后是线程池并发调用，首次加载模型时若不加锁，
# 多个请求会同时判定 _engine is None 并各加载一份模型
_lock = threading.Lock()


def get_engine():
    """RapidOCR 单例，避免每次调用重复加载模型。"""
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:  # 双重检查：等锁期间可能已被别的线程初始化
                from rapidocr_onnxruntime import RapidOCR

                _engine = RapidOCR()
    return _engine


def extract_text(image_bytes: bytes):
    """图片字节 -> (整段文字, 逐行结果[{text, score}])。"""
    if not image_bytes:
        return "", []
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    result, _ = get_engine()(np.array(img))
    if not result:
        return "", []
    lines = [{"text": item[1], "score": round(float(item[2]), 4)} for item in result]
    return "\n".join(x["text"] for x in lines), lines


def mean_score(lines) -> float:
    """逐行置信度的均值；无结果返回 0.0。"""
    if not lines:
        return 0.0
    return round(sum(float(x.get("score", 0.0)) for x in lines) / len(lines), 4)
