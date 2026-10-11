#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把本地微调模型打包成 models.tar.gz —— 供 Zeabur / 任意 PaaS 用
MODEL_ARCHIVE_URL 拉取(方案 A: 对象存储直链).

约定: 包内根目录直接是 en2zh/ 和 zh2en/ 两个目录, 与 download_models.py
里的 from_archive() 期望完全一致:
    en2zh/ <- ../model_marian_v2   (英译中, 微调 v2)
    zh2en/ <- ../model_marian_zh2en (中译英)

注意: 你本地目录名叫 model_marian_v2 / model_marian_zh2en, 但云端
download_models 找的是 en2zh / zh2en, 本脚本负责在打包时完成重命名,
所以你不用手动 cp.

用法:
    cd flask-web-deploy
    python pack_models.py
输出:
    ../models.tar.gz  (day06 根目录, 与 flask-web-deploy 同级, 不进 git)
"""
import os
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
DAY06 = os.path.abspath(os.path.join(HERE, '..'))

# (本地源目录, 包内目录名)
SRC_MAP = [
    (os.path.join(DAY06, 'model_marian_v2'),   'en2zh'),
    (os.path.join(DAY06, 'model_marian_zh2en'), 'zh2en'),
]

OUT = os.path.join(DAY06, 'models.tar.gz')


def main():
    # 0. 前置检查
    for src, name in SRC_MAP:
        if not os.path.isdir(src):
            print(f'[pack] 错误: 源目录不存在 -> {src}', file=sys.stderr)
            sys.exit(1)
        if not os.path.exists(os.path.join(src, 'config.json')):
            print(f'[pack] 警告: {src} 下没有 config.json, 可能不是完整模型',
                  file=sys.stderr)

    print(f'[pack] 输出: {OUT}')
    raw_total = 0
    n_files = 0
    with tarfile.open(OUT, 'w:gz', compresslevel=9) as tf:
        for src, name in SRC_MAP:
            print(f'[pack] 打包 {name} <- {src}')
            for root, _dirs, files in os.walk(src):
                rel = os.path.relpath(root, src)
                arc_root = name if rel == '.' else os.path.join(name, rel)
                for f in files:
                    fp = os.path.join(root, f)
                    arc = os.path.join(arc_root, f)
                    tf.add(fp, arc)
                    raw_total += os.path.getsize(fp)
                    n_files += 1

    size_mb = os.path.getsize(OUT) / 1e6
    print(f'[pack] 完成! 共 {n_files} 个文件, 原始 {raw_total/1e6:.0f} MB -> '
          f'压缩包 {size_mb:.0f} MB')
    print(f'[pack] 下一步: 把 models.tar.gz 上传到 COS/OSS/R2 拿公开 URL, '
          f'部署时填 MODEL_ARCHIVE_URL')


if __name__ == '__main__':
    main()
