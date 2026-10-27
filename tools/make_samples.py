"""生成演示/评测用的样例广告图与标注文件。

两套数据：

- 主评测集（labels.csv）：12 条清晰正面案例 + 6 条边界案例，指标与前版可直接对比。
- 噪声用例（noise_cases.json）：专测文本归一化的增量价值——跨行断词、字间空格、
  全角符号、大小写变体，外加两条「不该误报」的对照样本。这组直接喂文本，
  不打进主指标。

这些仍是合成图（白底黑字），仅用于跑通链路与做基准评测；
正式投递前建议替换为真实电商图，并人工复核标签。

用法：
    python tools/make_samples.py
"""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent.parent
SAMPLES = BASE / "data" / "samples"
EVAL_CSV = BASE / "data" / "eval" / "labels.csv"
NOISE_CASES_PATH = BASE / "data" / "eval" / "noise_cases.json"

FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
]

# (文件名, 标签, 文案, 主色, 备注)
CASES = [
    ("ad_01.png", 1, "全网最低价！限时秒杀", "#E24B4A", "极限词"),
    ("ad_02.png", 1, "三天根治颈椎病，无效退款", "#A32D2D", "医疗功效+虚假承诺"),
    ("ad_03.png", 1, "日入过万，躺赚不是梦", "#BA7517", "虚假承诺"),
    ("ad_04.png", 1, "加微信 abc123，一手货源", "#993C1D", "导流"),
    ("ad_05.png", 1, "国家级认证，销量第一", "#D4537E", "极限词"),
    ("ad_06.png", 1, "100%有效，假一赔十", "#854F0B", "虚假承诺"),
    ("ad_07.png", 0, "秋季新款，舒适透气，支持七天无理由退换", "#0F6E56", "正常"),
    ("ad_08.png", 0, "人体工学设计，久坐不累", "#185FA5", "正常"),
    ("ad_09.png", 0, "顺丰包邮，48 小时内发货", "#3B6D11", "正常"),
    ("ad_10.png", 0, "精选面料，亲肤柔软", "#534AB7", "正常"),
    ("ad_11.png", 0, "活动价 199 元，库存有限", "#5F5E5A", "正常"),
    ("ad_12.png", 0, "支持多种支付方式，售后无忧", "#0F6E56", "正常"),
    # 变体/隐含违规：词表命中不了，用于检验 LLM 能否补位
    ("hard_01.png", 1, "三天见效，不吃药也能好", "#A32D2D", "隐含疗效，需语义理解"),
    ("hard_02.png", 1, "老中医推荐，腰不疼了", "#993C1D", "隐含疗效，需语义理解"),
    ("hard_03.png", 1, "无效包退，放心下单", "#854F0B", "变体承诺"),
    ("hard_04.png", 1, "全网底价，错过再等一年", "#D4537E", "变体极限词"),
    # 易误报的正常样本
    ("hard_05.png", 0, "第一人民医院附近可自提", "#185FA5", "含「第一」但非极限词"),
    ("hard_06.png", 0, "最近上新，欢迎选购", "#3B6D11", "含「最近」易误伤"),
]

# 噪声用例：直接构造「OCR 可能输出的脏文本」，不走图片。
# 原因：RapidOCR 自己会把全角符号和字间空格规范化掉，
# 用图片造噪声测不出来，只有直接喂脏文本才暴露归一化的价值。
NOISE_CASES = [
    {"text": "三天根治颈椎病，无效退\n款", "target": "无效退款", "label": 1, "note": "跨行断词"},
    {"text": "无效 退款，放心下单", "target": "无效退款", "label": 1, "note": "字间空格"},
    {"text": "１００％有效，假一赔十", "target": "100%", "label": 1, "note": "全角符号"},
    {"text": "vx：abc1234 一手货源", "target": "vx", "label": 1, "note": "小写 vx"},
    {"text": "Vx：abc1234 一手货源", "target": "vx", "label": 1, "note": "首字母大写"},
    {"text": "加v：wxid8899", "target": "加V", "label": 1, "note": "词表只有「加V」"},
    {"text": "订单号 1380013800012345 已发货", "target": "13800138000", "label": 0, "note": "长数字串，不该切出手机号"},
    {"text": "第一人民医院附近可自提", "target": "第一品牌", "label": 0, "note": "含「第一」但非极限词"},
    {"text": "最 近上新，欢迎选购", "target": "", "label": 0, "note": "去空格不该造出新违禁词"},
]


def _font(size: int):
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    # 原来静默回退到 load_default()，中文全变方块、OCR 必然失败，
    # 但错误会推迟到评测阶段才暴露。这里直接失败，早报早改。
    raise RuntimeError(
        "未找到可用的中文字体，生成的样例图会变成方块、OCR 必然失败。"
        f"请在本机确认下列任一字体存在，或补充路径：{FONT_CANDIDATES}"
    )


def _render(name: str, text: str, color: str):
    font = _font(30)
    small = _font(18)
    body = text.split("\n")
    height = max(300, 150 + 52 * (len(body) - 1) + 90)
    img = Image.new("RGB", (640, height), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 640, 14], fill=color)
    for i, line in enumerate(body):
        d.text((32, 70 + i * 52), line, fill="#1A1A1A", font=font)
    d.text((32, height - 60), "示例广告图 · 仅用于演示", fill="#888780", font=small)
    img.save(SAMPLES / name)


def _write_csv(path: Path, cases):
    rows = ["filename,label,note"] + [f"{n},{l},{note}" for n, l, _, _, note in cases]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def make():
    SAMPLES.mkdir(parents=True, exist_ok=True)
    EVAL_CSV.parent.mkdir(parents=True, exist_ok=True)

    for name, _, text, color, _ in CASES:
        _render(name, text, color)

    _write_csv(EVAL_CSV, CASES)
    NOISE_JSON = NOISE_CASES_PATH
    NOISE_JSON.write_text(
        json.dumps(NOISE_CASES, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"主评测集 {len(CASES)} 张 -> {EVAL_CSV}")
    print(f"噪声用例 {len(NOISE_CASES)} 条 -> {NOISE_JSON}")
    print(f"图片目录 -> {SAMPLES}")


if __name__ == "__main__":
    make()
