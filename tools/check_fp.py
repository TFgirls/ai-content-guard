"""规则层误报检查：拿合法文案喂规则层，看有多少条被误判。

用途：`docs/评测方法论.md` 里写过「10 条合法文案误报 7 条」，但当时没有留下
可复现的数据与脚本，数字无法核对。这个脚本配合 `data/eval/fp_cases.json`
把这件事固定下来，改动规则或词表后可以随时重跑，看误报有没有变化。

用法：
    python tools/check_fp.py            # 逐条打印命中情况
    python tools/check_fp.py --quiet    # 只输出汇总

退出码：有误报时返回 1，便于接 CI。
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import BASE_DIR  # noqa: E402
from src.rules import scan_rules  # noqa: E402

CASES_PATH = BASE_DIR / "data" / "eval" / "fp_cases.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true", help="只输出汇总")
    args = ap.parse_args()

    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]
    hit = 0
    for c in cases:
        hits = scan_rules(c["text"])
        if hits:
            hit += 1
            if not args.quiet:
                words = "、".join(h["match"] for h in hits)
                print(f"[误报] {c['id']}  陷阱词={c['trap']:<6} 命中：{words}")
                print(f"       {c['text']}")
        elif not args.quiet:
            print(f"[正确] {c['id']}  陷阱词={c['trap']}")

    n = len(cases)
    print()
    print(f"合法文案 {n} 条，被规则层误判 {hit} 条（误报率 {hit / n:.0%}）")
    print("说明：这些文案不含真实违规，正确行为是零命中。")
    return 1 if hit else 0


if __name__ == "__main__":
    raise SystemExit(main())
