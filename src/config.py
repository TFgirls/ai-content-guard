import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

# 允许通过环境变量换模型 / 换网关（Ollama、vLLM、任何 OpenAI 兼容端点）
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

DATA_DIR = BASE_DIR / "data"
SAMPLES_DIR = DATA_DIR / "samples"
EVAL_CSV = DATA_DIR / "eval" / "labels.csv"
NOISE_CASES = DATA_DIR / "eval" / "noise_cases.json"
LEXICON_PATH = DATA_DIR / "lexicon.json"
REPORT_DIR = BASE_DIR / "reports"
REPORT_PATH = REPORT_DIR / "eval_report.md"

# 低于该置信度的违规判定标为「待人工复核」
REVIEW_CONFIDENCE = float(os.getenv("REVIEW_CONFIDENCE", "0.75"))

# OCR 平均置信度低于该值，视为图片质量可疑，结果一并转人工
OCR_LOW_SCORE = float(os.getenv("OCR_LOW_SCORE", "0.60"))

# Agent 循环：改写最多尝试几轮（超过则转人工）
MAX_REWRITE_ROUNDS = int(os.getenv("MAX_REWRITE_ROUNDS", "3"))
