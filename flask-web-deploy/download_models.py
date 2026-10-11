#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把两个方向的模型准备到 MODEL_DIR 下(构建期自动跑, 也可以手动跑)

目录约定:
    $MODEL_DIR/en2zh/   config.json + model.safetensors + source.spm + target.spm + ...
    $MODEL_DIR/zh2en/   同上

获取来源(优先级从高到低, 已存在的方向会跳过):
    1. MODEL_ARCHIVE_URL : 一个 tar/tar.gz 直链(腾讯云 COS / 阿里 OSS / Cloudflare R2 都行)
                           解开后里面应当就是 en2zh/ 和 zh2en/ 两个目录
    2. HF_MODEL_REPO     : 你的 HuggingFace 仓库(用 tools/upload_models_to_hf.py 上传)
    3. 官方基线          : Helsinki-NLP/opus-mt-en-zh 与 Helsinki-NLP/opus-mt-zh-en
                           (没上传微调模型也能跑, 效果=微调前的基线)

环境变量:
    MODEL_DIR         默认 /app/models
    MODEL_ARCHIVE_URL 模型包直链
    HF_MODEL_REPO     形如 gary/marian-en-zh
    HF_TOKEN          私有仓库需要
    HF_ENDPOINT       默认 huggingface.co; 连不上会自动切 hf-mirror.com 重试
"""

import os
import sys
import shutil
import tarfile
import tempfile
import urllib.request

MODEL_DIR = os.environ.get('MODEL_DIR', '/app/models')
ARCHIVE_URL = os.environ.get('MODEL_ARCHIVE_URL', '').strip()
HF_REPO = os.environ.get('HF_MODEL_REPO', '').strip()
HF_TOKEN = os.environ.get('HF_TOKEN', '').strip() or None
DIRECTIONS = ('en2zh', 'zh2en')

# 官方基线兜底
BASELINE = {'en2zh': 'Helsinki-NLP/opus-mt-en-zh',
            'zh2en': 'Helsinki-NLP/opus-mt-zh-en'}

ENDPOINTS = [os.environ.get('HF_ENDPOINT', 'https://huggingface.co').strip()]
if 'hf-mirror.com' not in ENDPOINTS[0]:
    ENDPOINTS.append('https://hf-mirror.com')     # 兜底: 国内网络


def log(msg):
    print(f'[models] {msg}', flush=True)


def exists(d):
    return os.path.exists(os.path.join(MODEL_DIR, d, 'config.json'))


def from_archive(url):
    """下载 tar 包并解开到 MODEL_DIR"""
    log(f'下载模型包: {url}')
    tmp = tempfile.mkdtemp()
    try:
        fn = os.path.join(tmp, 'models.tar')
        urllib.request.urlretrieve(url, fn)
        with tarfile.open(fn) as tf:
            tf.extractall(tmp)
        # 包里可能直接是 en2zh/ zh2en/, 也可能外面还套了一层目录(比如 models/en2zh/)
        roots = [tmp]
        for root, dirs, _ in os.walk(tmp):
            for d in dirs:
                if d in DIRECTIONS and root not in roots:
                    roots.append(root)
        got = 0
        for d in DIRECTIONS:
            if exists(d):
                continue
            for r in roots:
                src = os.path.join(r, d)
                if os.path.exists(os.path.join(src, 'config.json')):
                    shutil.move(src, os.path.join(MODEL_DIR, d))
                    log(f'  {d} <- 压缩包')
                    got += 1
                    break
        return got
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def from_hf(repo, token=None):
    """从 HF 仓库下载 en2zh/ zh2en/ 两个子目录"""
    from huggingface_hub import snapshot_download

    got = 0
    for d in DIRECTIONS:
        if exists(d):
            continue
        for ep in ENDPOINTS:
            os.environ['HF_ENDPOINT'] = ep
            tmp = tempfile.mkdtemp()
            try:
                snapshot_download(
                    repo_id=repo, repo_type='model', token=token,
                    allow_patterns=[f'{d}/*'], local_dir=tmp,
                )
                src = os.path.join(tmp, d)
                if os.path.exists(os.path.join(src, 'config.json')):
                    shutil.move(src, os.path.join(MODEL_DIR, d))
                    log(f'  {d} <- {repo} (via {ep})')
                    got += 1
                break
            except Exception as e:
                log(f'  {d} 从 {ep} 下载失败: {e}')
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
    return got


def from_baseline():
    """官方基线兜底: 保证两个方向一定有模型"""
    from huggingface_hub import snapshot_download

    got = 0
    for d in DIRECTIONS:
        if exists(d):
            continue
        for ep in ENDPOINTS:
            os.environ['HF_ENDPOINT'] = ep
            try:
                snapshot_download(repo_id=BASELINE[d], repo_type='model',
                                  local_dir=os.path.join(MODEL_DIR, d))
                log(f'  {d} <- {BASELINE[d]} (via {ep})')
                got += 1
                break
            except Exception as e:
                log(f'  {d} 从 {ep} 下载失败: {e}')
                continue
    return got


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 每次容器启动都会跑一次, 全齐就直接退出(零开销)
    if all(exists(d) for d in DIRECTIONS):
        log(f'两个方向模型都已就绪, 跳过 ({MODEL_DIR})')
        return 0

    log(f'MODEL_DIR = {MODEL_DIR}')
    if ARCHIVE_URL:
        from_archive(ARCHIVE_URL)
    if HF_REPO:
        from_hf(HF_REPO, HF_TOKEN)

    missing = [d for d in DIRECTIONS if not exists(d)]
    if missing:
        log(f'仍缺少 {missing}, 用官方基线补齐(首次约 300MB/方向)')
        from_baseline()

    missing = [d for d in DIRECTIONS if not exists(d)]
    if missing:
        log(f'!! 以下方向最终没拿到模型: {missing} (服务会退回在线加载, 首次请求会慢)')
    else:
        for d in DIRECTIONS:
            size = sum(os.path.getsize(os.path.join(dp, f))
                       for dp, _, fs in os.walk(os.path.join(MODEL_DIR, d))
                       for f in fs) / 1e6
            log(f'{d}: OK  {size:.0f} MB')
    return 0


if __name__ == '__main__':
    sys.exit(main())
