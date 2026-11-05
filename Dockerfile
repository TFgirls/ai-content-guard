# 内容合规审核 API 的镜像。
#
# 只容器化 FastAPI 服务，Streamlit 界面仍留在本地跑——界面是演示用的，
# 容器化它只会让镜像多背一套前端依赖，没有实际收益。
#
# 构建：docker build -t ai-content-guard:latest .
# 运行：docker run --rm -p 8000:8000 --env-file .env ai-content-guard:latest

FROM python:3.12-slim

# RapidOCR 依赖 opencv，而 slim 基础镜像不带这些系统库，
# 缺了会在 import cv2 时报 "libGL.so.1: cannot open shared object file"。
# 装完立刻清 apt 缓存，避免把 20MB 的包索引留在镜像里。
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先只拷依赖清单并安装。这样改代码不会让依赖层失效，
# 重新构建时能命中缓存，不必重装 800MB 的包。
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# 再拷代码。data/lexicon.json 是规则层的必需文件，
# rules.py 里直接 open() 读它，缺了会在第一次调 /review 时抛 FileNotFoundError。
#
# 不拷 app.py：那是 Streamlit 入口，镜像只跑 API，带上它反而让人误以为
# 容器里也能开界面。
COPY src/ ./src/
COPY data/lexicon.json ./data/lexicon.json

# 用非 root 用户跑。容器默认是 root，一旦应用被攻破影响面更大。
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"

# 密钥不写进镜像（.dockerignore 已排除 .env），运行时通过 --env-file 或
# docker compose 的 env_file 注入。写进镜像等于把 key 固化进每一层。
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
