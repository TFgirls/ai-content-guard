"""真实商品图评测。

和 `src/evaluate.py` 的区别：那边测的是脚本生成的合成图（白底黑字、标签由程序构造），
这里测的是从电商页面截下来的真实图（复杂背景、艺术字、多种排版）。

存在的意义：合成图上的指标只能证明链路正确，证明不了真实效果。
真实图上掉多少、掉在哪，只有这份数据能回答。

用法：
    python tools/eval_real.py                 # 仅规则层（不花额度）
    python tools/eval_real.py --llm           # 规则 + 大模型
    python tools/eval_real.py --detail        # 打印每张的判定与漏检原因

图片不进版本库（`data/real/images/` 已在 .gitignore），
标注文件 `data/real/labels.csv` 会入库，OCR 文本可从 `scan_result.json` 复盘。
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import BASE_DIR  # noqa: E402
from src.pipeline import detect  # noqa: E402

DATA_DIR = BASE_DIR / "data" / "real"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="启用大模型判定（会消耗 API 额度）")
    ap.add_argument("--detail", action="store_true", help="打印逐张判定结果")
    args = ap.parse_args()

    labels = list(csv.DictReader(open(DATA_DIR / "labels.csv", encoding="utf-8-sig")))
    tp = fp = fn = tn = 0
    undetermined = 0
    misses, false_alarms = [], []

    for row in labels:
        img = DATA_DIR / "images" / row["filename"]
        if not img.exists():
            print(f"[跳过] {row['filename']} 图片不存在")
            continue

        a = detect(img.read_bytes(), use_llm=args.llm)
        if a.undetermined:
            undetermined += 1

        truth = int(row["violation"])
        pred = 1 if a.is_violation else 0
        if pred == 1 and truth == 1:
            tp += 1
        elif pred == 1 and truth == 0:
            fp += 1
            false_alarms.append((row, [h["match"] for h in a.rule_hits]))
        elif pred == 0 and truth == 1:
            fn += 1
            misses.append(row)
        else:
            tn += 1

        if args.detail:
            flag = "命中" if pred == truth else "错误"
            print(
                f"[{flag}] {row['filename']} 真值={truth} 预测={pred} "
                f"风险={a.judgment.risk_level} 命中={[h['match'] for h in a.rule_hits]}"
            )

    n = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    print()
    print(f"真实商品图评测  样本 {n}  模式={'规则 + 大模型' if args.llm else '仅规则'}")
    print("-" * 58)
    print(f"  违规 {tp + fn} 张 / 正常 {tn + fp} 张")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  精确率 {precision:.1%}   召回率 {recall:.1%}   F1 {f1:.1%}")
    print(f"  无法判定（OCR 无文字）{undetermined} 张，会强制转人工")

    if false_alarms:
        print("\n误报（判违规但标注正常）：")
        for row, hits in false_alarms:
            print(f"  {row['filename']}  命中 {hits}")
            print(f"     {row['note']}")

    if misses:
        print(f"\n漏检（标注违规但未判出）：{len(misses)} 张")
        for row in misses:
            print(f"  {row['filename']}  [{row['category']}] {row['note']}")


if __name__ == "__main__":
    main()
