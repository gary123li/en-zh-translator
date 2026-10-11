# -*- coding: utf-8 -*-
"""
习语 / 固定表达 兜底词典(在翻译前后做字符串处理, 不占推理时间)

两个方向各一份:
    apply_idioms(en_text)     -> 英译中 用(英文习语 -> 正确中文)
    apply_idioms_cn(cn_text)  -> 中译英 用(中文成语/客套话 -> 地道英文)

为什么需要它:
    实测结论(见 07 脚本的回归测试对比):
        预训练模型: It's a piece of cake.  ->  这是小菜一碟        ✅
        v2 微调后 : It's a piece of cake.  ->  这是一个蛋糕。      ❌
    原因: 新闻语料(news-commentary)里几乎没有习语, 微调等于"用 0 样本"把预训练
    好不容易学会的习语先验冲刷掉了. 这是"语料分布问题", 靠加数据很难治(除非手工
    攒几千条习语句对), 所以用一个小词典硬兜底, 性价比最高.

两层机制:
    第1层 整句精确匹配: 整句就是一句习语 -> 直接返回中文, 不经过模型(零风险)
    第2层 短语改写    : 习语嵌在长句里 -> 把英文习语换成"字面表达", 再交给模型翻译
                        例如: "a piece of cake" -> "very easy" -> 模型译"这很容易"
                        这样至少不会译成"一块蛋糕"

接入方式(web/app.py 的 do_translate 里已经接好):
    from idioms import apply_idioms
    cn, en2 = apply_idioms(en_sentence)
    out = cn if cn else translator.translate(en2)
"""

import re

# ==================== 第 1 层: 整句精确匹配 ====================
# key 是"归一化"后的英文(小写 / 统一撇号 / 去掉结尾标点和引号)
SENTENCE_IDIOMS = {
    "it's a piece of cake": '这是小菜一碟。',
    "it is a piece of cake": '这是小菜一碟。',
    "a piece of cake": '小菜一碟。',
    "don't cry over spilled milk": '覆水难收，别为无法挽回的事后悔。',
    "do not cry over spilled milk": '覆水难收，别为无法挽回的事后悔。',
    "don't cry over spilt milk": '覆水难收，别为无法挽回的事后悔。',
    "let's call it a day": '今天就到此为止吧。',
    "let us call it a day": '今天就到此为止吧。',
    "let's call it a night": '今晚就到此为止吧。',
    "break a leg": '祝你好运。',
    "long time no see": '好久不见。',
    "so far so good": '到目前为止一切顺利。',
    "better late than never": '迟做总比不做好。',
    "no pain no gain": '不付出就没有收获。',
    "practice makes perfect": '熟能生巧。',
    "easy come easy go": '来得容易，去得也快。',
    "it's raining cats and dogs": '外面正下着倾盆大雨。',
    "speak of the devil": '说曹操，曹操到。',
    "when pigs fly": '那是不可能的事。',
    "it's not rocket science": '这又不是什么高深的事。',
    "hang in there": '再坚持一下。',
    "the ball is in your court": '现在轮到你做决定了。',
    "you can't have your cake and eat it too": '鱼和熊掌不可兼得。',
    "that's the way the cookie crumbles": '事已至此，只能接受。',
    "we're all wasting time": '我们都在浪费时间。',
    "i don't know what you're talking about": '我不知道你在说什么。',
}

# ==================== 第 2 层: 短语级英文改写 ====================
# 元组成员: (匹配习语的正则, 替换成的字面表达)
# 顺序有讲究: 长的/更具体的放前面
PHRASE_REWRITES = [
    (r"\bcry(ing)? over spil(l|t)ed milk\b", "regret something that cannot be undone"),
    (r"\ba piece of cake\b", "very easy"),
    (r"\bkill(ing|ed)? two birds with one stone\b", "achieve two goals at once"),
    (r"\bhit(ting)? the nail on the head\b", "be exactly right"),
    (r"\bcost(s|ing)? an arm and a leg\b", "be very expensive"),
    (r"\bonce in a blue moon\b", "very rarely"),
    (r"\bwhen pigs fly\b", "never"),
    (r"\bspill(ed|ing)? the beans\b", "reveal a secret"),
    (r"\bbreak(ing)? the ice\b", "start a friendly conversation"),
    (r"\b(bite|biting|bit) the bullet\b", "endure something painful"),
    (r"\bthe last straw\b", "the final and worst problem"),
    (r"\bmiss(ed|ing)? the boat\b", "miss the opportunity"),
    (r"\bcut(ting)? corners\b", "do something poorly to save money"),
    (r"\bby the book\b", "strictly according to the rules"),
    (r"\bin the long run\b", "over a long period of time"),
    (r"\bon the same page\b", "in agreement"),
    (r"\bunder the weather\b", "slightly ill"),
    (r"\bget(ting)? out of hand\b", "become impossible to control"),
    (r"\bhit the books\b", "study hard"),
    (r"\bcall(ing)? it a day\b", "stop working on this"),
    (r"\bhang(ing)? in there\b", "keep trying"),
    (r"\bno pain,? no gain\b", "there is no reward without effort"),
    (r"\bspeak of the devil\b", "the person we were just talking about"),
    (r"\bmiss the point\b", "fail to understand the main idea"),
    (r"\bget cold feet\b", "become too nervous to act"),
    (r"\bhead over heels\b", "very deeply"),
]


def norm_key(s):
    """归一化: 小写 + 统一撇号 + 去掉首尾引号/标点/多余空格"""
    s = s.strip().lower().replace('\u2019', "'").replace('\u2018', "'")
    s = s.strip('"').strip('\u201c').strip('\u201d').strip()
    s = re.sub(r'[.!?,;:\s]+$', '', s)
    return re.sub(r'\s+', ' ', s).strip()


def apply_idioms(en_text):
    """
    对一句英文做习语处理.
    返回: (中文译文 or None, 处理后的英文)
        - 第 1 层命中: 返回 (中文, 原句), 调用方直接用中文, 不用再过模型
        - 第 2 层命中: 返回 (None, 改写后的英文), 调用方拿改写后的英文去翻译
        - 都没命中  : 返回 (None, 原句)
    """
    if not en_text:
        return None, en_text

    # 第 1 层: 整句
    hit = SENTENCE_IDIOMS.get(norm_key(en_text))
    if hit:
        return hit, en_text

    # 第 2 层: 短语改写
    out = en_text
    for pattern, repl in PHRASE_REWRITES:
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
    return (None, out) if out != en_text else (None, en_text)


# ==================== 中译英方向: 中文 -> 英文 ====================
# 同样的思路: 成语和客套话在平行语料里几乎没有, 模型直译出来会非常生硬
# ('这是小菜一碟' -> 'This is a small dish of cake'), 所以先兜底.

# 第 1 层: 整句精确匹配(中文客套话 / 高频口语句)
SENTENCE_IDIOMS_CN = {
    '你好': 'Hello.',
    '你好吗': 'How are you?',
    '早上好': 'Good morning.',
    '下午好': 'Good afternoon.',
    '晚上好': 'Good evening.',
    '晚安': 'Good night.',
    '再见': 'Goodbye.',
    '谢谢': 'Thank you.',
    '谢谢你': 'Thank you.',
    '非常感谢': 'Thank you very much.',
    '不客气': "You're welcome.",
    '对不起': "I'm sorry.",
    '不好意思': 'Excuse me.',
    '没关系': "It doesn't matter.",
    '没问题': 'No problem.',
    '我不知道': "I don't know.",
    '我明白了': 'I see.',
    '请稍等': 'Please wait a moment.',
    '多少钱': 'How much is it?',
    '厕所在哪里': 'Where is the restroom?',
    '我不会说英语': "I don't speak English.",
    '你能帮我一下吗': 'Could you give me a hand?',
    '很高兴认识你': 'Nice to meet you.',
    '好久不见': 'Long time no see.',
    '一路平安': 'Have a safe trip.',
    '新年快乐': 'Happy New Year.',
    '生日快乐': 'Happy birthday.',
    '恭喜发财': 'Wish you prosperity.',
    '马到成功': 'Wish you every success.',
    '一帆风顺': 'May everything go smoothly.',
    '这是小菜一碟': "It's a piece of cake.",
    '小菜一碟': "It's a piece of cake.",
    '加油': 'Come on, you can do it!',
    '我等你': "I'll wait for you.",
    '救命': 'Help!',
}

# 第 2 层: 成语/俗语 -> 白话中文(改写后再交给模型, 至少不会译成字面意思)
PHRASE_REWRITES_CN = [
    (r'小菜一碟', '非常容易'),
    (r'覆水难收', '事情已经无法挽回'),
    (r'画蛇添足', '多此一举'),
    (r'对牛弹琴', '白费口舌'),
    (r'守株待兔', '妄想不劳而获'),
    (r'亡羊补牢', '出了问题及时补救'),
    (r'破釜沉舟', '下定决心不留退路'),
    (r'一箭双雕', '一举两得'),
    (r'入乡随俗', '适应当地的风俗习惯'),
    (r'雪中送炭', '在别人困难时给予帮助'),
    (r'井底之蛙', '见识短浅的人'),
    (r'半途而废', '中途就放弃了'),
    (r'全力以赴', '尽自己最大的努力'),
    (r'莫名其妙', '让人完全无法理解'),
    (r'不可思议', '令人难以置信'),
    (r'乱七八糟', '非常混乱'),
    (r'马马虎虎', '勉强过得去'),
    (r'一如既往', '和过去完全一样'),
]


def norm_key_cn(s):
    """中文归一化: 去掉所有标点/空格(中文不靠空格分词), 繁简暂不做转换"""
    s = s.strip()
    s = re.sub(r'[\s，。！？；：、“”‘’（）《》\.,!?;:\'"()<>\[\]]+', '', s)
    return s


def apply_idioms_cn(cn_text):
    """
    对一句中文做习惯用语处理.
    返回: (英文译文 or None, 处理后的中文)
        - 第 1 层命中: 返回 (英文, 原句), 调用方直接用英文, 不用再过模型
        - 第 2 层命中: 返回 (None, 改写后的中文), 调用方拿改写后的中文去翻译
        - 都没命中  : 返回 (None, 原句)
    """
    if not cn_text:
        return None, cn_text

    hit = SENTENCE_IDIOMS_CN.get(norm_key_cn(cn_text))
    if hit:
        return hit, cn_text

    out = cn_text
    for pattern, repl in PHRASE_REWRITES_CN:
        out = re.sub(pattern, repl, out)
    return (None, out) if out != cn_text else (None, cn_text)


if __name__ == '__main__':
    # 自测: python idioms.py
    tests = [
        "It's a piece of cake.",
        "Don't cry over spilled milk.",
        "Let's call it a day.",
        "He said, \"I can't believe it!\" and then walked away.",
        "The new system is expensive, but in the long run it will save us money.",
    ]
    for t in tests:
        cn, en2 = apply_idioms(t)
        print(f'原句  : {t}')
        print(f'处理后: {en2}')
        print(f'直出  : {cn}')
        print('-' * 50)

    print('\n[中译英]')
    for t in ['这是小菜一碟。', '好久不见。', '他总是半途而废。', '请问去最近的地铁站怎么走？']:
        en, cn2 = apply_idioms_cn(t)
        print(f'原句  : {t}')
        print(f'处理后: {cn2}')
        print(f'直出  : {en}')
        print('-' * 50)
