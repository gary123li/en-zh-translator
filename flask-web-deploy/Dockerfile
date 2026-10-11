# 中英互译 Flask 服务 - 通用 Docker 镜像
# 适配 Zeabur / HuggingFace Spaces(Docker) / Render / Railway 等所有支持 Docker 的 PaaS.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_ENDPOINT=https://hf-mirror.com \
    MODEL_DIR=/app/models

WORKDIR /app

# CPU 版 torch 单独走官方 CPU 索引(比 CUDA 版小 ~4 倍, 省内存)
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt

COPY . .

# 构建期下载模型: 设了 MODEL_ARCHIVE_URL 或 HF_MODEL_REPO 才下载;
# 都没设则跳过, 运行时 download_models 会退回官方基线(首次在线下载约 300MB/方向).
# 想"模型直接打进镜像"就删掉 .dockerignore 里的 models/ 等三行, 把模型放进 models/ 目录.
RUN if [ -n "$MODEL_ARCHIVE_URL" ] || [ -n "$HF_MODEL_REPO" ]; then \
        python download_models.py; \
    else \
        echo "[build] 未设置模型来源, 运行时将使用官方基线或在线下载"; \
    fi

HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=5 \
    CMD python -c "import urllib.request,os,sys; p=os.environ.get('PORT','8080'); sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:%s/healthz'%p, timeout=10).status==200 else 1)"

EXPOSE 8080
CMD ["python", "app.py"]
