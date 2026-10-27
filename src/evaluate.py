"""审核效果评测：判定指标 + 改写复检指标 + OCR 噪声鲁棒性对比。

用法（项目根目录）：
    python -m src.evaluate                # 规则 + DeepSeek 判定
    python -m src.evaluate --no-llm       # 仅规则，快且不花钱
    python -m src.evaluate --with-rewrite # 额外评测改写效果（多一次 LLM 调用/条）

原来的评测只覆盖判定，而项目的卖点是「改写 + 复检量化」——
没有改写指标，等于宣称的能力没有证据。这里补上。
"""

import argparse
import csv
import json
from pathlib import Path

from .config import EVAL_CSV, NOISE_CASES, REPORT_PATH, SAMPLES_DIR
from .pipeline import analyze, detect
from .rules import scan_rules


def _load_labels(path: Path = EVAL_CSV):
    if not path.exists():
        return []
    # utf-8-sig：Excel 导出的 CSV 带 BOM，用 utf-8 读会让首列名变成 \ufefffilename
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        name = (r.get("filename") or "").strip()
        label = (r.get("label") or "").strip()
        if not name or label not in {"0", "1"}:
            continue
        out.append({"filename": name, "label": int(label), "note": (r.get("note") or "").strip()})
    return out


def _predict(img_path: Path, use_llm: bool = True):
    a = detect(img_path.read_bytes(), use_llm=use_llm)
    j = a.judgment
    return {
        "pred": 1 if a.is_violation else 0,
        "undetermined": a.undetermined,
        "level": j.risk_level if j else "unknown",
        "conf": j.confidence if j else 0.0,
        "llm_used": bool(j and j.llm_used),
        "llm_status": (j.status if j else "unknown"),
        "error": (j.error if j else ""),
        "ocr_score": a.ocr_score,
    }


def _rewrite_stats(rows, use_llm: bool):
    """对真实违规样本跑完整链路，量化改写效果。

    指标定义：
    - 改写执行率：成功拿到改写文本的占比（排除 Key 失效/网络失败）
    - 零残留率：改写后规则层一条都不命中的占比
    - 平均残留词数：改写后仍命中的违禁词个数
    """
    total = executed = clean = 0
    residual_total = 0
    details = []
    for r in rows:
        if r["label"] != 1:
            continue
        path = SAMPLES_DIR / r["filename"]
        if not path.exists():
            continue
        total += 1
        a = analyze(path.read_bytes()) if use_llm else None
        if a is None or not a.is_violation:
            details.append((r["filename"], "-", "未判为违规，跳过改写", "-", "-"))
            continue
        if not a.rewrite_ok:
            details.append(
                (r["filename"], len(a.rule_hits), "改写未执行", "-", (a.rewrite_error or "")[:40])
            )
            continue
        executed += 1
        n_res = len(a.residual_hits)
        residual_total += n_res
        if n_res == 0:
            clean += 1
        details.append(
            (
                r["filename"],
                len(a.rule_hits),
                f"{n_res} 残留",
                "干净" if n_res == 0 else "有残留",
                (a.changes[0] if a.changes else "")[:40],
            )
        )
    return {
        "total": total,
        "executed": executed,
        "clean": clean,
        "residual_total": residual_total,
        "clean_rate": clean / executed if executed else 0.0,
        "exec_rate": executed / total if total else 0.0,
        "avg_residual": residual_total / executed if executed else 0.0,
        "details": details,
    }


def _hit_texts(hits):
    """命中项里可用于比对的文本（归一化、小写）。"""
    from .rules import normalize

    out = []
    for h in hits:
        for s in (h.get("match", ""), h.get("evidence", "")):
            if s:
                out.append(normalize(s)[0].lower())
    return out


def _noise_compare():
    """文本层噪声用例：同一批脏文本，对比归一化开 / 关的召回与误报。

    为什么不走 OCR：RapidOCR 自己就会把全角符号和字间空格规范化掉，
    用图片造噪声根本测不出来。这类缺陷只有直接喂脏文本才暴露。
    """
    if not NOISE_CASES.exists():
        return None
    cases = json.loads(NOISE_CASES.read_text(encoding="utf-8"))
    if not cases:
        return None

    from .rules import normalize

    viol = {"raw": 0, "norm": 0, "total": 0}
    clean = {"raw": 0, "norm": 0, "total": 0}
    details = []

    for c in cases:
        text = c["text"]
        label = int(c["label"])
        target = normalize(c.get("target") or "")[0].lower()

        raw_hits = scan_rules(text, normalize_text=False)
        norm_hits = scan_rules(text, normalize_text=True)
        if target:
            raw_ok = any(target in t for t in _hit_texts(raw_hits))
            norm_ok = any(target in t for t in _hit_texts(norm_hits))
        else:  # 未指定目标词：只要有任何命中即算（用于「不该误报」的样本）
            raw_ok = bool(raw_hits)
            norm_ok = bool(norm_hits)

        if label == 1:
            viol["total"] += 1
            viol["raw"] += int(raw_ok)
            viol["norm"] += int(norm_ok)
        else:
            clean["total"] += 1
            clean["raw"] += int(raw_ok)  # 正常样本命中 = 误报
            clean["norm"] += int(norm_ok)

        details.append((c["text"].replace("\n", "⏎"), label, raw_ok, norm_ok, c.get("note", "")))

    return {"viol": viol, "clean": clean, "details": details}


def evaluate_all(use_llm: bool = True, with_rewrite: bool = False) -> dict:
    """跑完整评测集，写报告，返回指标字典。"""
    rows = _load_labels()
    if not rows:
        raise FileNotFoundError(f"{EVAL_CSV} 为空或不存在；请先放图片并标注 label（1=违规, 0=正常）")

    tp = fp = fn = tn = 0
    llm_calls = 0
    llm_fail = 0
    undetermined = 0
    details = []

    for r in rows:
        path = SAMPLES_DIR / r["filename"]
        if not path.exists():
            details.append((r["filename"], r["label"], "-", "缺文件", "-", "-"))
            continue
        p = _predict(path, use_llm=use_llm)
        if p["undetermined"]:
            undetermined += 1
            details.append((r["filename"], r["label"], "无法判定", p["level"], f"{p['conf']:.2f}", r["note"]))
            continue
        pred, truth = p["pred"], r["label"]
        llm_calls += int(p["llm_used"])
        if use_llm and p["llm_status"] == "llm_failed":
            llm_fail += 1
        if pred == 1 and truth == 1:
            tp += 1
        elif pred == 1 and truth == 0:
            fp += 1
        elif pred == 0 and truth == 1:
            fn += 1
        else:
            tn += 1
        ok = "命中" if pred == truth else "错误"
        details.append((r["filename"], truth, pred, p["level"], f"{p['conf']:.2f} {ok}", r["note"]))

    n = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    acc = (tp + tn) / n if n else 0.0

    lines = ["# 审核评测报告", ""]
    _CN = "一二三四五六"
    _sec = [0]

    def _h(title: str) -> str:
        _sec[0] += 1
        return f"## {_CN[_sec[0] - 1]}、{title}"

    # ---- 判定指标 ----
    if use_llm and llm_fail == len(rows) - undetermined and n:
        lines += [
            "> ⚠️ **LLM 全部调用失败，本次结果实际等同「仅规则」模式。**",
            "> 失败原因见下方「失败样本」小节，不要把这组数当作 LLM 效果。",
            "",
        ]

    lines += [
        _h("判定指标"),
        "",
        f"- 样本数：{n}（违规 {tp + fn} / 正常 {fp + tn}）"
        + (f"，另有 {undetermined} 张无法判定（未计入指标）" if undetermined else ""),
        f"- 判定模式：{'规则 + 大模型' if use_llm else '仅规则'}"
        + (f"（LLM 实际生效 {llm_calls}/{n}，失败 {llm_fail} 条退回规则）" if use_llm else ""),
        "",
        "| 指标 | 数值 |",
        "|---|---|",
        f"| 准确率 accuracy | {acc:.2%} |",
        f"| 精确率 precision | {precision:.2%} |",
        f"| 召回率 recall | {recall:.2%} |",
        f"| F1 | {f1:.2%} |",
        "",
        f"混淆矩阵：TP={tp} FP={fp} FN={fn} TN={tn}",
        "",
        "| 文件 | 真实 | 预测 | 风险级 | 置信度 / 结果 | 备注 |",
        "|---|---|---|---|---|---|",
    ]
    lines += [f"| {d[0]} | {d[1]} | {d[2]} | {d[3]} | {d[4]} | {d[5]} |" for d in details]

    # ---- 失败样本（把静默失败摆到台面上）----
    fail_rows = [d for d in details if d[3] == "-" or "失败" in str(d[4])]
    if use_llm:
        lines += ["", _h("失败样本"), ""]
        if llm_fail == 0 and not fail_rows:
            lines.append("- 无。所有样本均正常完成判定。")
        else:
            lines.append(f"- LLM 调用失败 {llm_fail} 条。失败样本已在上表标注，判定层日志同时输出告警。")

    # ---- 改写指标 ----
    if with_rewrite:
        lines += ["", _h("改写复检指标"), ""]
        if not use_llm:
            lines.append("- 未启用（改写需要 LLM，`--no-llm` 模式下跳过）。")
        else:
            rs = _rewrite_stats(rows, use_llm=True)
            if rs["executed"] == 0:
                lines.append(f"- 违规样本 {rs['total']} 条，改写**一次都没执行成功**，改写效果无数据。")
            else:
                lines += [
                    f"- 违规样本 {rs['total']} 条，改写实际执行 {rs['executed']} 条（执行率 {rs['exec_rate']:.0%}）",
                    f"- 改写后零残留 {rs['clean']} 条（**零残留率 {rs['clean_rate']:.0%}**）",
                    f"- 平均残留违禁词 {rs['avg_residual']:.2f} 个/条",
                    "",
                    "| 文件 | 改写前命中 | 改写后 | 结论 | 首个修改说明 |",
                    "|---|---|---|---|---|",
                ]
                lines += [f"| {d[0]} | {d[1]} | {d[2]} | {d[3]} | {d[4]} |" for d in rs["details"]]

    # ---- OCR 噪声鲁棒性 ----
    noise = _noise_compare()
    if noise and noise["viol"]["total"]:
        v, c = noise["viol"], noise["clean"]
        lines += ["", _h("文本归一化的增量价值（同一批脏文本，开 / 关对比）"), ""]
        lines.append(
            f"- 违规文本 {v['total']} 条，召回：不做归一化 **{v['raw'] / v['total']:.0%}** "
            f"→ 归一化后 **{v['norm'] / v['total']:.0%}**"
        )
        if c["total"]:
            lines.append(
                f"- 正常文本 {c['total']} 条，误报：不做归一化 {c['raw']}/{c['total']} "
                f"→ 归一化后 {c['norm']}/{c['total']}（归一化不应引入新的误报）"
            )
        lines += [
            "- 噪声指：跨行断词、字间空格、全角数字符号、大小写变体。这些在真实广告图上很常见。",
            "- 注意：这组用例直接喂文本，不走 OCR——RapidOCR 自己会抹平全角与空格，用图片测不出来。",
            "",
            "| 文本 | 真实 | 原文匹配 | 归一化后 | 噪声类型 |",
            "|---|---|---|---|---|",
        ]
        lines += [
            f"| {d[0]} | {d[1]} | {'✅' if d[2] else '❌'} | {'✅' if d[3] else '❌'} | {d[4]} |"
            for d in noise["details"]
        ]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "n": n,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "llm_calls": llm_calls,
        "llm_fail": llm_fail,
        "undetermined": undetermined,
        "report_path": str(REPORT_PATH),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true", help="只用规则层，不调用大模型")
    ap.add_argument("--with-rewrite", action="store_true", help="额外评测改写效果（更慢、更费额度）")
    args = ap.parse_args()

    m = evaluate_all(use_llm=not args.no_llm, with_rewrite=args.with_rewrite)
    print(
        f"样本 {m['n']}  准确率 {m['accuracy']:.0%}  精确率 {m['precision']:.0%}  "
        f"召回率 {m['recall']:.0%}  F1 {m['f1']:.0%}"
    )
    if args.no_llm is False:
        print(f"LLM 生效 {m['llm_calls']}/{m['n']}，失败 {m['llm_fail']} 条")
    if m["undetermined"]:
        print(f"无法判定 {m['undetermined']} 条（未计入指标）")
    print(f"报告已写入 {m['report_path']}")


if __name__ == "__main__":
    main()
