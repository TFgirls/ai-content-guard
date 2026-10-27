import hashlib
import html

import streamlit as st

from src.config import DEEPSEEK_API_KEY, DEEPSEEK_MODEL, OCR_LOW_SCORE, REVIEW_CONFIDENCE
from src.evaluate import evaluate_all
from src.pipeline import analyze

st.set_page_config(page_title="内容合规审核台", page_icon="🛡️", layout="wide")

RISK_STYLE = {
    "high": ("高危", "#FCEBEB", "#791F1F", "#F7C1C1"),
    "medium": ("中风险", "#FAEEDA", "#854F0B", "#FAC775"),
    "low": ("低风险", "#EAF3DE", "#3B6D11", "#C0DD97"),
    "unknown": ("无法判定", "#F1EFE8", "#5F5E5A", "#D3D1C7"),
}


def chip(text: str, bg: str, fg: str) -> str:
    # OCR 文本来自用户上传的图片，必须转义后再进 HTML，否则可被注入标签
    safe = html.escape(str(text), quote=True)
    return (
        f'<span style="display:inline-block;font-size:12px;padding:2px 9px;margin:2px 4px 2px 0;'
        f'border-radius:6px;background:{bg};color:{fg};border:0.5px solid {fg}33;">{safe}</span>'
    )


def esc(text: str) -> str:
    return html.escape(str(text), quote=True)


def brief(err: str, limit: int = 90) -> str:
    """错误信息去掉 request_id 之类噪声，界面上只留关键那句。"""
    s = (err or "未知原因").split("(request_id")[0].strip()
    return s if len(s) <= limit else s[:limit] + "…"


st.title("内容合规审核台")
st.caption("上传图片 → OCR 提取文字 → 规则 + 大模型判定 → 违规自动改写 → 改写后复检")

with st.sidebar:
    st.subheader("运行状态")
    st.write(
        "大模型："
        + (f"已配置（{DEEPSEEK_MODEL}）" if DEEPSEEK_API_KEY else "未配置（退回纯规则判定）")
    )
    st.caption(f"人工复核阈值：置信度 < {REVIEW_CONFIDENCE}")
    st.caption(f"OCR 质量预警阈值：平均置信度 < {OCR_LOW_SCORE}")
    st.divider()
    st.subheader("批量评测")
    st.caption("跑 data/samples + labels.csv，输出判定指标与噪声鲁棒性对比")
    if st.button("运行评测（仅规则）", width="stretch"):
        try:
            m = evaluate_all(use_llm=False)
            st.success(
                f"准确率 {m['accuracy']:.0%} · 召回率 {m['recall']:.0%} · F1 {m['f1']:.0%}"
            )
            st.caption(m["report_path"])
        except Exception as e:
            st.error(f"评测失败：{e}")

up = st.file_uploader("上传一张广告/商品图", type=["png", "jpg", "jpeg"])

if not up:
    st.info("上传图片后自动开始审核。也可以先点左侧「运行评测」看基准数据。")
    st.stop()

data = up.getvalue()
# 用内容摘要做缓存键，避免「同名同大小但内容不同」的图命中旧结果
cache_key = hashlib.md5(data).hexdigest()
if st.session_state.get("cache_key") != cache_key:
    with st.spinner("识别与判定中..."):
        try:
            st.session_state["result"] = analyze(data)
        except Exception as e:
            st.session_state["result"] = None
            st.session_state["error"] = f"{type(e).__name__}: {e}"
    st.session_state["cache_key"] = cache_key

if st.session_state.get("error") and st.session_state.get("result") is None:
    st.error(f"处理失败：{st.session_state['error']}")
    st.caption("常见原因：图片损坏、OCR 模型未就绪、依赖缺失。")
    st.stop()

a = st.session_state["result"]
j = a.judgment

left, right = st.columns([1, 1.35], gap="large")

with left:
    st.image(data, caption=up.name, width="stretch")
    if a.lines:
        st.caption(f"OCR 平均置信度 {a.ocr_score:.3f} · 识别 {len(a.lines)} 行")

with right:
    level = j.risk_level if j else "unknown"
    label, bg, fg, border = RISK_STYLE.get(level, RISK_STYLE["unknown"])
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(
            f'<span style="font-size:13px;padding:4px 12px;border-radius:999px;'
            f'background:{bg};color:{fg};border:0.5px solid {border};">{esc(label)}</span>',
            unsafe_allow_html=True,
        )
    with c2:
        st.metric("置信度", f"{j.confidence:.2f}" if j else "—")

    # 无法判定 ≠ 合规，必须明确区别于绿色通过
    if a.undetermined:
        st.warning("图片中未识别到文字，**无法判定合规性**。请不要把它当成审核通过。")

    if a.review_reasons:
        st.info("需人工复核：\n" + "\n".join(f"- {r}" for r in a.review_reasons))

    st.markdown("**提取文字（OCR）**")
    st.code(a.ocr_text or "（未识别到文字）", language=None)

    if DEEPSEEK_API_KEY and j and not j.llm_used and j.status == "llm_failed":
        st.warning(f"大模型调用失败，本次为纯规则判定。原因：{brief(j.error)}")

    if a.rule_hits:
        st.markdown("**规则层命中**")
        chips = "".join(
            chip(f"{h['category']} · {h.get('evidence') or h['match']}", "#FCEBEB", "#A32D2D")
            for h in a.rule_hits
        )
        st.markdown(chips, unsafe_allow_html=True)
    elif not a.undetermined:
        st.success("规则层未命中违禁词")

    if j and j.reasons:
        st.markdown("**判定理由**")
        st.markdown(
            "".join(f'<div style="font-size:13px;color:#444441;">· {esc(r)}</div>' for r in j.reasons),
            unsafe_allow_html=True,
        )

    if a.is_violation:
        st.markdown("**改写对比**")
        o, n = st.columns(2)
        with o:
            st.caption("原文")
            st.markdown(
                f'<div style="font-size:13px;line-height:1.7;color:#888780;text-decoration:line-through;">'
                f"{esc(a.ocr_text)}</div>",
                unsafe_allow_html=True,
            )
        with n:
            st.caption("改写版")
            st.markdown(
                f'<div style="font-size:13px;line-height:1.7;color:#085041;">{esc(a.rewritten)}</div>',
                unsafe_allow_html=True,
            )
        if a.changes:
            st.caption("修改说明：" + "；".join(a.changes))

        # 关键区分：没改写成功 ≠ 改得不干净
        if not a.rewrite_ok:
            st.warning(
                f"改写未执行（{brief(a.rewrite_error)}），上图为原文，不代表已验证合规。"
            )
        elif a.residual_hits:
            st.error("改写后仍残留违规：" + "、".join(h.get("evidence") or h["match"] for h in a.residual_hits))
        else:
            st.success("改写后复检：无残留违禁词")
