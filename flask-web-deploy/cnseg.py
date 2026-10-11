# -*- coding: utf-8 -*-
"""
中文侧预处理(【关键】训练和推理必须用同一套, 否则模型看到的分布不一致)

为什么需要这个文件:
    Helsinki-NLP/opus-mt-zh-en 是"中译英"方向的 MarianMT, 它的 source.spm(中文)
    是在 OPUS 语料上训的, 而 OPUS 的中文语料普遍是"按词切分"的('我 爱 你' 而不是 '我爱你').
    所以中文送进模型前最好先分词, 这样更贴近模型预训练时见过的样子.

    【最重要的一条原则】不管用哪种切分方式, 训练语料和线上推理必须用同一个函数:
        训练: 08.重建中译英语料.py  -> segment_cn(中文) 写进 cmn-eng-v2.txt
        推理: web/app.py            -> segment_cn(用户输入的中文)
    只要两边一致, 用哪种方式都能学得动; 一旦不一致, 模型等于在考没练过的题型.

分词策略(自动降级, 不会因为缺包报错):
    1. 装了 jieba    -> 正常分词: '我喜欢读书' -> '我 喜欢 读书'(推荐, 和 OPUS 分布最接近)
    2. 没装 jieba    -> 原样返回连续中文: MarianTokenizer 的 spm 会自己切子词
                        (效果略差一点点, 但训练和推理依然是一致的, 不会翻车)

想装 jieba(很小, 纯 Python):
    pip install jieba -i https://pypi.tuna.tsinghua.edu.cn/simple
"""

try:
    import jieba                      # 可选依赖
    import logging
    logging.getLogger('jieba').setLevel(logging.ERROR)   # 关掉 jieba 的 "Building prefix dict" 刷屏
    HAS_JIEBA = True
except Exception:
    jieba = None
    HAS_JIEBA = False


def segment_cn(text):
    """
    把一句中文处理成模型期望的输入形态.
    - 有 jieba: 按词切分, 词之间加空格
    - 无 jieba: 原样返回
    """
    if not text:
        return text
    if not HAS_JIEBA:
        return text.strip()
    # jieba.cut 返回生成器; 用 lcut 直接拿 list, 代码更短
    words = [w for w in jieba.lcut(text.strip()) if w.strip()]
    return ' '.join(words)


def describe():
    """给日志用的一句话说明当前用的是哪种模式"""
    return 'jieba 分词' if HAS_JIEBA else '不分词(未装 jieba, 由 sentencepiece 自行切子词)'


if __name__ == '__main__':
    tests = [
        '这是小菜一碟。',
        '2023年，苹果公司的营收达到3830亿美元，同比增长8%。',
        '由清华大学张教授领导的研究团队上个月发表了一篇论文。',
    ]
    print(f'当前模式: {describe()}')
    for t in tests:
        print(f'  原文  : {t}')
        print(f'  处理后: {segment_cn(t)}')
