import hashlib
import html
import io
import sys

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

# 用 `python app.py` 直接启动时，Streamlit 没有脚本运行上下文：
# st.stop() 依赖异常通知 runtime 来中止，bare mode 下没人接住，
# 代码会一路跑到 up.getvalue() 报 "NoneType has no attribute getvalue"——
# 那个报错完全看不出真正原因。这里提前拦住并说清楚。
if get_script_run_ctx(suppress_warning=True) is None:
    print(
        "\n[提示] 这是一个 Streamlit 应用，不能用 python app.py 直接运行。\n"
        "请改用下面的命令启动：\n\n"
        "    streamlit run app.py\n",
        flush=True,
    )
    sys.exit(0)

from src.config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_MODEL,
    REVIEW_CONFIDENCE,
    SAMPLES_DIR,
)
from PIL import Image

from src.evaluate import evaluate_all
from src.pipeline import analyze

st.set_page_config(page_title="内容合规审核台", page_icon="🛡️", layout="wide")

ACCENT = "#12A594"

RISK_STYLE = {
    "high": ("高危", "#FCEBEB", "#A32D2D", "#F0C2C2"),
    "medium": ("中风险", "#FAEEDA", "#92400E", "#F1D08A"),
    "low": ("低风险", "#E9F3DB", "#3F6212", "#C7DDA0"),
    "unknown": ("无法判定", "#EFEEE9", "#57534E", "#D6D3CD"),
}

# 空态展示的三张样图：覆盖「规则命中 / 应该放过 / 规则漏检」三种情况
SAMPLE_PICKS = [
    ("ad_02.png", "规则命中", "医疗功效 + 虚假承诺，规则层直接抓出"),
    ("ad_07.png", "应当放过", "合规文案，检验会不会误报"),
    ("hard_01.png", "规则漏检", "换词说法，靠大模型补语义"),
]

CSS = """
<style>
.block-container { padding-top: 2.1rem; max-width: 1260px; }

/* 标题区 */
.hero-sub {
  font-size: 13.5px; opacity: .62; margin: -10px 0 16px; letter-spacing: .01em;
}
.cap-row { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:22px; }
.cap {
  font-size: 12px; padding: 4px 11px; border-radius: 999px;
  color: #0F8A7C; background: rgba(18,165,148,.10);
  border: 1px solid rgba(18,165,148,.28);
}

/* 上传区：撑高 + 虚线 + 垂直居中，让主操作有存在感 */
[data-testid="stFileUploaderDropzone"] {
  min-height: 150px; border-radius: 16px; padding: 18px;
  display: flex; flex-direction: column;
  align-items: center; justify-content: center; gap: 4px;
  border: 1.5px dashed rgba(18,165,148,.50);
  background: rgba(18,165,148,.045);
  transition: border-color .15s, background .15s;
}
[data-testid="stFileUploaderDropzone"]:hover {
  border-color: #12A594; background: rgba(18,165,148,.09);
}
[data-testid="stFileUploaderDropzone"] button { border-radius: 10px; }
[data-testid="stFileUploaderDropzone"] small { opacity: .45; font-size: 11.5px; }

/* 图片统一成卡片感，否则白底样图和页面糊在一起 */
[data-testid="stImage"] img {
  border-radius: 10px;
  border: 1px solid rgba(128,128,128,.16);
}

/* 本地演示不需要云部署入口 */
[data-testid="stAppDeployButton"] { display: none; }
[data-testid="stHeader"] { background: transparent; }

/* 分区小标题 */
.sec-label {
  font-size: 13px; font-weight: 600; opacity: .72;
  letter-spacing: .04em; margin: 28px 0 2px;
}

/* 样图卡片 */
.pick-name { font-size: 13.5px; font-weight: 600; margin: 10px 0 1px; }
.pick-desc { font-size: 12px; opacity: .62; line-height: 1.55; min-height: 38px; }

/* 侧栏键值行 */
.kv {
  display:flex; justify-content:space-between; gap:10px;
  font-size: 12.5px; padding: 6px 0;
  border-bottom: 1px solid rgba(128,128,128,.14);
}
.kv span { opacity: .58; }
.kv b { font-weight: 600; }

/* OCR 文本框：等宽字体读中文很难受，换成常规字体 + 保留换行 */
.ocr-box {
  white-space: pre-wrap; font-size: 13.5px; line-height: 1.8;
  background: var(--secondary-background-color);
  padding: 12px 15px; border-radius: 10px;
  border: 1px solid rgba(128,128,128,.14);
}

/* 改写对比 */
.diff-old {
  font-size: 13px; line-height: 1.8; color: #8A8681;
  text-decoration: line-through; text-decoration-thickness: 1px;
}
.diff-new { font-size: 13px; line-height: 1.8; color: #0B7A6B; font-weight: 500; }

.reason-row { font-size: 13px; line-height: 1.7; opacity: .8; }

/* 指标数字收一号，别和判定标签抢层级 */
[data-testid="stMetricValue"] { font-size: 1.7rem; line-height: 1.25; }
[data-testid="stMetricLabel"] { font-size: 12.5px; opacity: .6; }
</style>
"""

# 有结果时把上传区压成一条，否则它会一直占着首屏
CSS_COMPACT_UPLOAD = """
<style>
[data-testid="stFileUploaderDropzone"] {
  min-height: 56px; padding: 8px 16px;
  flex-direction: row; gap: 12px; border-radius: 12px;
}
[data-testid="stFileUploaderDropzone"] small { display: none; }
[data-testid="stFileUploader"] [data-testid="stWidgetLabel"] { display: none; }
</style>
"""


def esc(text: str) -> str:
    return html.escape(str(text), quote=True)


def chip(text: str, bg: str, fg: str) -> str:
    # OCR 文本来自用户上传的图片，必须转义后再进 HTML，否则可被注入标签
    safe = esc(text)
    return (
        f'<span style="display:inline-block;font-size:12px;padding:3px 10px;margin:3px 5px 3px 0;'
        f'border-radius:7px;background:{bg};color:{fg};border:0.5px solid {fg}33;'
        f'font-weight:500;">{safe}</span>'
    )


def brief(err: str, limit: int = 90) -> str:
    """错误信息去掉 request_id 之类噪声，界面上只留关键那句。"""
    s = (err or "未知原因").split("(request_id")[0].strip()
    return s if len(s) <= limit else s[:limit] + "…"


def kv(key: str, value: str) -> str:
    return f'<div class="kv"><span>{esc(key)}</span><b>{esc(value)}</b></div>'


def reset():
    for k in ("picked", "cache_key", "result", "error"):
        st.session_state[k] = None


st.markdown(CSS, unsafe_allow_html=True)

# ---------------- 顶部 ----------------
st.title("内容合规审核台")
st.markdown(
    '<div class="hero-sub">一张广告图进来，判定、改写、复检一起出去</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="cap-row">'
    + "".join(f'<span class="cap">{c}</span>' for c in ["规则可解释", "大模型补语义", "自动合规改写", "改写后复检"])
    + "</div>",
    unsafe_allow_html=True,
)

# ---------------- 侧栏 ----------------
with st.sidebar:
    st.markdown("**运行状态**")
    st.markdown(
        kv("大模型", DEEPSEEK_MODEL if DEEPSEEK_API_KEY else "未配置")
        + kv("判定模式", "规则 + 大模型" if DEEPSEEK_API_KEY else "仅规则")
        + kv("复核阈值", f"{REVIEW_CONFIDENCE}"),
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("**批量评测**")
    st.caption("跑 18 张样例，输出判定指标与归一化增益")
    if st.button("运行评测", width="stretch"):
        try:
            m = evaluate_all(use_llm=False)
            st.success(f"准确率 {m['accuracy']:.0%} · 召回 {m['recall']:.0%} · F1 {m['f1']:.0%}")
            st.caption(m["report_path"])
        except Exception as e:
            st.error(f"评测失败：{e}")

# ---------------- 取图：上传优先，其次点选的样图 ----------------
up = st.file_uploader("上传一张广告 / 商品图", type=["png", "jpg", "jpeg"])

if up:
    st.session_state["picked"] = None
    source_name, data = up.name, up.getvalue()
elif st.session_state.get("picked"):
    source_name, data = st.session_state["picked"]
else:
    st.markdown('<div class="sec-label">快速试用</div>', unsafe_allow_html=True)
    st.caption("没有素材？点任意一张样图，直接跑完整链路。")
    cols = st.columns(3, gap="medium")
    for col, (fname, tag, desc) in zip(cols, SAMPLE_PICKS):
        path = SAMPLES_DIR / fname
        if not path.exists():
            continue
        with col:
            st.image(str(path), width="stretch")
            st.markdown(f'<div class="pick-name">{esc(tag)}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="pick-desc">{esc(desc)}</div>', unsafe_allow_html=True)
            if st.button("用这张试用", key=f"pick_{fname}", width="stretch"):
                st.session_state["picked"] = (fname, path.read_bytes())
                st.rerun()
    st.stop()

# ---------------- 处理 ----------------
# 已经拿到图了：上传区收成一条，把首屏让给结果
st.markdown(CSS_COMPACT_UPLOAD, unsafe_allow_html=True)

cache_key = hashlib.md5(data).hexdigest()
if st.session_state.get("cache_key") != cache_key:
    with st.spinner("识别与判定中…"):
        try:
            st.session_state["result"] = analyze(data)
            st.session_state["error"] = None
        except Exception as e:
            st.session_state["result"] = None
            st.session_state["error"] = f"{type(e).__name__}: {e}"
    st.session_state["cache_key"] = cache_key

if st.session_state.get("error") and st.session_state.get("result") is None:
    st.error(f"处理失败：{st.session_state['error']}")
    st.caption("常见原因：图片损坏、OCR 模型未就绪、依赖缺失。")
    if st.button("重新开始"):
        reset()
        st.rerun()
    st.stop()

a = st.session_state["result"]
j = a.judgment

left, right = st.columns([1, 1.6], gap="large")

with left:
    st.image(data, width="stretch")
    try:
        _w, _h = Image.open(io.BytesIO(data)).size
        size_txt = f"{_w} × {_h}"
    except Exception:
        size_txt = "—"
    st.markdown(
        kv("文件", source_name)
        + kv("尺寸", size_txt)
        + kv("大小", f"{len(data) / 1024:.0f} KB")
        + (kv("OCR", f"{len(a.lines)} 行 · {a.ocr_score:.3f}") if a.lines else ""),
        unsafe_allow_html=True,
    )
    if st.button("换一张图片", width="stretch"):
        reset()
        st.rerun()

with right:
    level = j.risk_level if j else "unknown"
    label, bg, fg, border = RISK_STYLE.get(level, RISK_STYLE["unknown"])
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.markdown(
            f'<span style="display:inline-block;font-size:13px;font-weight:600;padding:5px 14px;'
            f'border-radius:999px;background:{bg};color:{fg};border:1px solid {border};">{esc(label)}</span>',
            unsafe_allow_html=True,
        )
    with c2:
        st.metric("置信度", f"{j.confidence:.2f}" if j else "—")

    # 无法判定 ≠ 合规，必须明确区别于绿色通过
    if a.undetermined:
        st.warning("图片中未识别到文字，**无法判定合规性**。请不要把它当成审核通过。")

    if a.review_reasons:
        st.caption("需人工复核")
        st.markdown(
            "".join(f'<div class="reason-row">· {esc(r)}</div>' for r in a.review_reasons),
            unsafe_allow_html=True,
        )

    st.markdown("**提取文字**")
    st.markdown(f'<div class="ocr-box">{esc(a.ocr_text or "（未识别到文字）")}</div>', unsafe_allow_html=True)

    if DEEPSEEK_API_KEY and j and not j.llm_used and j.status == "llm_failed":
        st.warning(f"大模型调用失败，本次为纯规则判定。原因：{brief(j.error)}")

    if a.rule_hits:
        st.markdown("**规则层命中**")
        st.markdown(
            "".join(
                chip(f"{h['category']} · {h.get('evidence') or h['match']}", "#FCEBEB", "#A32D2D")
                for h in a.rule_hits
            ),
            unsafe_allow_html=True,
        )
    elif not a.undetermined:
        st.success("规则层未命中违禁词")

    if j and j.reasons:
        st.markdown("**判定理由**")
        st.markdown(
            "".join(f'<div class="reason-row">· {esc(r)}</div>' for r in j.reasons),
            unsafe_allow_html=True,
        )

    if a.is_violation:
        st.markdown("**改写对比**")
        o, n = st.columns(2, gap="medium")
        with o:
            st.caption("原文")
            st.markdown(f'<div class="diff-old">{esc(a.ocr_text)}</div>', unsafe_allow_html=True)
        with n:
            st.caption("改写版")
            st.markdown(f'<div class="diff-new">{esc(a.rewritten)}</div>', unsafe_allow_html=True)
        if a.changes:
            st.markdown("**修改说明**")
            st.markdown(
                "".join(f'<div class="reason-row">· {esc(c)}</div>' for c in a.changes),
                unsafe_allow_html=True,
            )

        # Agent 循环的逐轮记录：这是「改写重试到收敛」的直接证据
        if a.trace and a.trace.attempts:
            st.markdown("**改写过程**")
            st.markdown(
                "".join(
                    f'<div class="reason-row">· 第 {r.round} 轮：命中 {len(r.hits)} 处 '
                    f'→ 改写后残留 {len(r.residual)} 处</div>'
                    for r in a.trace.rounds
                ),
                unsafe_allow_html=True,
            )

        # 关键区分：没改写成功 ≠ 改得不干净
        if not a.rewrite_ok:
            st.warning(f"改写未执行（{brief(a.rewrite_error)}），上图为原文，不代表已验证合规。")
        elif a.residual_hits:
            st.error(
                "改写后仍残留违规："
                + "、".join(h.get("evidence") or h["match"] for h in a.residual_hits)
            )
        else:
            st.success("改写后复检：无残留违禁词")
