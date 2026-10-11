#!/bin/sh
# 一些非 Docker 平台(或想显式控制启动)时使用.
# app.py 内部已会在启动时调用 download_models 准备模型, 这里再调一次只是保险.
set -e
if [ -n "$MODEL_DIR" ]; then
  python download_models.py || echo "[warn] 模型准备失败, 继续启动(未就绪方向会退回官方基线)"
fi
exec python app.py
