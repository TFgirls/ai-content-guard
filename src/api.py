"""HTTP 接口层：把审核链路封成可调用的 API。

启动：
    uvicorn src.api:app --host 127.0.0.1 --port 8000
交互文档（自动生成）：
    http://127.0.0.1:8000/docs

设计说明：
- 端点是同步 def，FastAPI 会把它丢进线程池。OCR 与模型调用都是阻塞的，
  写成 async def 会卡住事件循环——这是接口层最容易踩的坑。
- 返回值里带完整的 Agent trace，方便调用方看到「第几轮才改干净」，
  而不是只拿到一个最终结果。
"""

from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import DEEPSEEK_MODEL, MAX_REWRITE_ROUNDS
from .pipeline import analyze

MAX_UPLOAD = 10 * 1024 * 1024  # 10MB
ALLOWED_TYPES = {"image/png", "image/jpeg", "image/jpg"}

app = FastAPI(
    title="内容合规审核 API",
    version="1.0.0",
    description="图片 OCR → 规则 + 大模型双层判定 → Agent 循环改写 → 改写后复检",
)

# demo 阶段放开跨域，正式部署应收敛到白名单
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class HitOut(BaseModel):
    category: str = ""
    match: str = ""
    evidence: str = ""
    reason: str = ""


class RoundOut(BaseModel):
    round: int
    hit_count: int
    residual_count: int
    ok: bool
    error: str = ""
    changes: List[str] = Field(default_factory=list)


class ReviewOut(BaseModel):
    status: str                       # judged / no_text
    risk_level: str                   # low / medium / high / unknown
    confidence: float
    categories: List[str] = Field(default_factory=list)
    reasons: List[str] = Field(default_factory=list)
    llm_used: bool = False
    ocr_text: str = ""
    ocr_score: float = 0.0
    rule_hits: List[HitOut] = Field(default_factory=list)
    rewritten: str = ""
    residual_hits: List[HitOut] = Field(default_factory=list)
    needs_review: bool = False
    review_reasons: List[str] = Field(default_factory=list)
    agent_summary: str = ""
    agent_rounds: List[RoundOut] = Field(default_factory=list)


@app.get("/health")
def health():
    """健康检查：供容器探针 / 负载均衡使用。"""
    return {
        "status": "ok",
        "model": DEEPSEEK_MODEL,
        "max_rewrite_rounds": MAX_REWRITE_ROUNDS,
    }


@app.get("/")
def root():
    return {"service": "内容合规审核 API", "docs": "/docs", "health": "/health"}


@app.post("/review", response_model=ReviewOut)
def review(file: UploadFile = File(...)):
    """上传一张广告/商品图，返回完整审核结果。"""
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, f"不支持的类型：{file.content_type}，仅接受 PNG / JPEG")

    data = file.file.read()
    if not data:
        raise HTTPException(400, "文件为空")
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, f"文件超过 {MAX_UPLOAD // 1024 // 1024}MB")

    try:
        a = analyze(data)
    except Exception as e:  # OCR / 模型异常统一转 500，并保留原因
        raise HTTPException(500, f"处理失败：{type(e).__name__}: {e}")

    j = a.judgment
    rounds = []
    if a.trace:
        rounds = [
            RoundOut(
                round=r.round,
                hit_count=len(r.hits),
                residual_count=len(r.residual),
                ok=r.ok,
                error=r.error,
                changes=r.changes[:3],
            )
            for r in a.trace.rounds
        ]

    return ReviewOut(
        status=a.status,
        risk_level=j.risk_level if j else "unknown",
        confidence=j.confidence if j else 0.0,
        categories=(j.categories if j else []),
        reasons=(j.reasons if j else []),
        llm_used=bool(j and j.llm_used),
        ocr_text=a.ocr_text,
        ocr_score=a.ocr_score,
        rule_hits=[HitOut(**{k: h.get(k, "") for k in HitOut.model_fields}) for h in a.rule_hits],
        rewritten=a.rewritten,
        residual_hits=[
            HitOut(**{k: h.get(k, "") for k in HitOut.model_fields}) for h in a.residual_hits
        ],
        needs_review=a.needs_review,
        review_reasons=a.review_reasons,
        agent_summary=a.trace.summary() if a.trace else "",
        agent_rounds=rounds,
    )
