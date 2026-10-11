# -*- coding: utf-8 -*-
"""
中英互译 Web 翻译服务(Flask)

支持两个方向, 页面上可随时切换(也可以"自动识别"输入语言):
    英译中 en2zh : Helsinki-NLP/opus-mt-en-zh  (05/07 脚本产出 model_marian / model_marian_v2)
    中译英 zh2en : Helsinki-NLP/opus-mt-zh-en  (09 脚本产出 model_marian_zh2en)

==================== 本地跑 ====================
    pip install flask -i https://pypi.tuna.tsinghua.edu.cn/simple
    python app.py
然后浏览器打开: http://127.0.0.1:7860

==================== 部署到公网(GitHub + PaaS) ====================
代码推到 GitHub, 由 Zeabur / HuggingFace Spaces / Render 等从仓库构建。
关键环境变量:
    MODEL_DIR      模型根目录; 设了它就走"云端布局": $MODEL_DIR/en2zh、 $MODEL_DIR/zh2en
                  没设则退回本地 day06 布局(见下方路径段), 方便本机调试。
    MODEL_ARCHIVE_URL  模型 tar.gz 直链(COS/OSS/R2), 启动时自动下载解开(见 download_models.py)
    HF_MODEL_REPO      你的 HuggingFace 模型仓库(形如 gary/marian-en-zh)
    PORT               PaaS 注入的端口(默认 7860)
    HOST               绑定地址(默认 0.0.0.0)
    QUANTIZE=1         int8 动态量化, 内存减半、更快, BLEU 掉一点点
模型不会进 Git(太大), 走上面两种方式之一拉取, 都没有就退回官方基线(首次在线下载约 300MB/方向)。
详见 README.md。

可选参数:
    python app.py --port 8080                # 换端口
    python app.py --dir zh2en                # 默认方向(en2zh / zh2en / auto)
    python app.py --model pretrained         # 指定默认模型(仅对默认方向生效)
    python app.py --device cpu               # 强制 CPU
"""

import os

# 【关键】必须在 import transformers 之前设置! 否则国内网络连 huggingface.co 会一直超时.
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')

import sys
import re
import time
import argparse
import threading

# windows 控制台默认 gbk, 打印中文会乱码
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

try:
    from flask import Flask, request, jsonify, render_template
except ImportError:
    print('=' * 60)
    print('缺少 Flask! 请先执行(只需一次):')
    print('  pip install flask -i https://pypi.tuna.tsinghua.edu.cn/simple')
    print('=' * 60)
    sys.exit(1)

try:
    from idioms import apply_idioms, apply_idioms_cn      # 习语兜底词典(和 app.py 同目录)
except ImportError:
    def apply_idioms(en_text):           # 找不到就退化成"不处理", 不影响主流程
        return None, en_text

    def apply_idioms_cn(cn_text):
        return None, cn_text

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
except ImportError:
    print('=' * 60)
    print('缺少 transformers / torch! 请先执行:')
    print('  pip install transformers sentencepiece -i https://pypi.tuna.tsinghua.edu.cn/simple')
    print('=' * 60)
    sys.exit(1)


# ==================== 路径与配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # .../flask-web-deploy
DAY06_DIR = os.path.dirname(BASE_DIR)                          # 仅本地调试时找 day06 下的模型/cnseg

# 中文预处理模块 cnseg.py: 训练和推理必须共用同一套切分逻辑(已随本目录一起发布)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
try:
    from cnseg import segment_cn, describe as seg_describe
except ImportError:
    def segment_cn(text):            # 兜底: 找不到就原样返回, 不影响主流程
        return text

    def seg_describe():
        return '不分词(未找到 cnseg.py)'

# ---- 模型路径: 部署 / 本地两套布局 ----
# 设了 MODEL_DIR 就走云端布局(download_models.py 把 en2zh/ zh2en 放在这里);
# 没设就退回 day06 本地布局, 方便本机直接 python app.py 调试。
DEPLOY_MODEL_DIR = os.environ.get('MODEL_DIR')

if DEPLOY_MODEL_DIR:
    FINETUNED_V2_DIR = os.path.join(DEPLOY_MODEL_DIR, 'en2zh')   # 云端: 英译中微调在 en2zh/
    FINETUNED_DIR    = None                                       # 云端不传 v1
    ZH2EN_DIR        = os.path.join(DEPLOY_MODEL_DIR, 'zh2en')     # 云端: 中译英微调在 zh2en/
else:
    FINETUNED_DIR    = os.path.join(DAY06_DIR, 'model_marian')      # 英译中微调 v1 (旧语料: 全小写/无数字)
    FINETUNED_V2_DIR = os.path.join(DAY06_DIR, 'model_marian_v2')  # 英译中微调 v2 (07 脚本产出: 温和清洗语料)
    ZH2EN_DIR        = os.path.join(DAY06_DIR, 'model_marian_zh2en')  # 中译英微调 (09 脚本产出)

PRETRAINED_EN2ZH = 'Helsinki-NLP/opus-mt-en-zh'                  # 英译中预训练
PRETRAINED_ZH2EN = 'Helsinki-NLP/opus-mt-zh-en'                  # 中译英预训练

MAX_INPUT_TOKENS = 256     # 单句送进模型的最大 token 数
GEN_MAX_LEN = 200          # 单句最多生成多少个 token
MAX_SENT_WORDS = 60        # 英文侧: 超过这个单词数的长句, 再按逗号切一刀, 防止翻译不完整
MAX_SENT_CHARS = 80        # 中文侧: 超过这个字数的长句, 再按逗号切一刀


def _model_exists(d):
    """config.json 存在才认为该模型可用(路径为 None 时直接 False)"""
    return bool(d) and os.path.exists(os.path.join(d, 'config.json'))


# ==================== 两个方向各自的模型 ====================
# 结构: DIRECTIONS[方向][模型key] = {path, label, available}
MODELS = {
    'en2zh': {
        # v2 放第一个: 存在时它就是默认模型, 页面上排在下拉框最上面
        'finetuned_v2': {
            'path': FINETUNED_V2_DIR,
            'label': '微调模型 v2 (model_marian_v2 · 温和清洗语料)',
            'available': _model_exists(FINETUNED_V2_DIR),
        },
        'finetuned': {
            'path': FINETUNED_DIR,
            'label': '微调模型 v1 (model_marian · 旧语料)',
            'available': _model_exists(FINETUNED_DIR),
        },
        'pretrained': {
            'path': PRETRAINED_EN2ZH,
            'label': '预训练模型 (Helsinki-NLP/opus-mt-en-zh)',
            'available': True,
        },
    },
    'zh2en': {
        'finetuned': {
            'path': ZH2EN_DIR,
            'label': '微调模型 (model_marian_zh2en · 09 脚本产出)',
            'available': _model_exists(ZH2EN_DIR),
        },
        'pretrained': {
            'path': PRETRAINED_ZH2EN,
            'label': '预训练模型 (Helsinki-NLP/opus-mt-zh-en)',
            'available': True,
        },
    },
}

DIRECTION_LABELS = {'en2zh': '英译中', 'zh2en': '中译英', 'auto': '自动识别'}
# 每个方向里"优先当默认"的顺序(存在就选它)
PREFERRED = {'en2zh': ['finetuned_v2', 'finetuned', 'pretrained'],
             'zh2en': ['finetuned', 'pretrained']}


# ==================== 翻译器 ====================
class Translator:
    """对单个模型的封装: 懒加载 + 线程安全的 generate"""

    def __init__(self, path, device):
        self.path = path
        self.device = torch.device(device)
        self.tokenizer = None
        self.model = None
        self.lock = threading.Lock()   # 同一时刻只允许一个推理, 避免多线程抢资源
        self.loaded = False
        self.error = None

    def load(self):
        if self.loaded:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.path).to(self.device)
        self.model.eval()
        # 清掉 config 里默认的 max_length=512, 否则和 max_new_tokens 冲突, 每次都会刷警告
        self.model.generation_config.max_length = None
        # 可选 int8 动态量化: 内存减半、推理更快, BLEU 掉一点点(仅 CPU 生效)
        if os.environ.get('QUANTIZE') == '1' and self.device.type == 'cpu':
            try:
                from torch.ao.quantization import quantize_dynamic
                self.model = quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
                print(f'  [quantize] {self.path} -> int8 动态量化完成', flush=True)
            except Exception as e:
                print(f'  [quantize] 量化失败, 退回 fp32: {e}', flush=True)
        self.loaded = True

    def params(self):
        return sum(p.numel() for p in self.model.parameters()) / 1e6

    def translate_one(self, text, num_beams=4):
        """翻译一句(中文的话, 调用方已经做过 segment_cn 了)"""
        inputs = self.tokenizer(
            text, return_tensors='pt', truncation=True, max_length=MAX_INPUT_TOKENS
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=GEN_MAX_LEN,
                num_beams=num_beams,
                early_stopping=False,      # 关掉: beam 分数接近时会在第一个 </s> 就停, 导致输出为空
                no_repeat_ngram_size=2,    # 避免重复词语
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    def translate(self, text, num_beams=4):
        with self.lock:
            return self.translate_one(text, num_beams)


class TranslatorHub:
    """管理多个模型: 按需加载, 加载后常驻内存"""

    def __init__(self, device):
        self.device = device
        self.cache = {}
        self.lock = threading.Lock()

    def get(self, direction, key):
        """按 (方向, 模型key) 取翻译器; 两个方向的模型各自缓存, 互不覆盖"""
        if direction not in MODELS or key not in MODELS[direction]:
            raise KeyError(f'{direction}/{key}')
        if not MODELS[direction][key]['available']:
            # 后台下载可能刚完成, 重新扫一次磁盘再决定
            if _model_exists(MODELS[direction][key]['path']):
                MODELS[direction][key]['available'] = True
                DEFAULT_MODEL[direction] = pick_default(direction)
            else:
                raise KeyError(f'{direction}/{key} 不可用')

        ck = f'{direction}:{key}'
        with self.lock:
            t = self.cache.get(ck)
            if t is None:
                t = Translator(MODELS[direction][key]['path'], self.device)
                self.cache[ck] = t
            if not t.loaded:
                t.load()
            return t


# ==================== 文本切分 ====================
def split_paragraphs(text):
    """按空行/换行切出段落(保留原文段落结构)"""
    return [p.strip() for p in re.split(r'[\r\n]+', text.strip()) if p.strip()]


def split_sentences_en(para):
    """
    把一段英文切成适合模型翻译的小句.
    MarianMT 是句级模型, 整段长文直接喂进去会丢内容, 所以先分句再逐句翻译.
    """
    sents = []
    # 英文分句: .!? 后面跟空格 + 大写字母/数字/引号
    for seg in re.split(r'(?<=[.!?])\s+(?=["\'(\[]?[A-Z0-9])', para):
        seg = seg.strip()
        if not seg:
            continue
        # 超长句再按逗号/分号/冒号切一刀
        if len(seg.split()) > MAX_SENT_WORDS:
            buf = ''
            for piece in re.split(r'(?<=[,;:])\s+', seg):
                if buf and len(buf.split()) + len(piece.split()) > MAX_SENT_WORDS:
                    sents.append(buf.strip())
                    buf = piece
                else:
                    buf = (buf + ' ' + piece).strip()
            if buf:
                sents.append(buf)
        else:
            sents.append(seg)
    return sents


def split_sentences_cn(para):
    """
    把一段中文切成小句: 按 。！？；… 切, 超长句再按 ，、： 切一刀.
    英文那套正则在中文上完全不管用(没有空格和大写), 所以要单独写一份.
    """
    sents = []
    for seg in re.split(r'(?<=[。！？!?；;…])', para):
        seg = seg.strip()
        if not seg:
            continue
        if len(seg) > MAX_SENT_CHARS:
            buf = ''
            for piece in re.split(r'(?<=[，,、：:])', seg):
                piece = piece.strip()
                if not piece:
                    continue
                if buf and len(buf) + len(piece) > MAX_SENT_CHARS:
                    sents.append(buf)
                    buf = piece
                else:
                    buf = (buf + piece)
            if buf:
                sents.append(buf)
        else:
            sents.append(seg)
    return [s for s in sents if s.strip()]


# ==================== 自动识别语言 ====================
# 判定阈值(启动时可用 --zh-ratio / --min-cn 调整)
ZH_RATIO = 0.30   # 主规则: 汉字数 >= ASCII 字母数 * ZH_RATIO -> 中译英
MIN_CN = 1        # 兜底: 判为中译英所需的最少汉字数(英文里夹的一两个汉字不算)


def _is_han(ch):
    """汉字: 基本区 + 扩展A + 兼容汉字(不含日文假名/韩文)"""
    o = ord(ch)
    return 0x4e00 <= o <= 0x9fff or 0x3400 <= o <= 0x4dbf or 0xf900 <= o <= 0xfaff


def _is_cjk_punct(ch):
    """中文/全角标点: ，。、；：「」《》…… 等(排除全角数字与全角字母)"""
    o = ord(ch)
    return (0x3000 <= o <= 0x303f or 0xff01 <= o <= 0xff20
            or 0xff3b <= o <= 0xff40 or 0xff5b <= o <= 0xff65)


def detect_direction(text):
    """自动识别: 汉字占比高就是"中译英", 否则"英译中"

    主规则: 汉字数 >= ASCII 字母数 * ZH_RATIO(默认 0.3), 且至少有 MIN_CN 个汉字。
    之所以不是"谁多谁赢", 是因为真实输入常是中英混排, 只要中文能成句就该按中文处理;
    少量汉字往往只是英文句子里的引语或专名, 仍应走英译中。

    在主规则之外补了三条兜底, 覆盖纯统计会判错的边界:
      1. 一个 ASCII 字母都没有, 但有汉字/中文标点  -> 必然不是英文, 判中译英
         (解决了"全是中文标点分隔""汉字在扩展区"等原本会掉到 en2zh 的情况)
      2. 一个汉字都没有                            -> 英译中(纯英文/纯数字/假名/谚文等)
      3. 汉字数 < MIN_CN                           -> 英译中(英文里夹的一两个汉字不算数)
    """
    han = sum(1 for ch in text if _is_han(ch))
    letters = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    cjk = han + sum(1 for ch in text if _is_cjk_punct(ch))

    if letters == 0 and cjk > 0:        # 兜底1: 没有英文字母, 但有中文痕迹
        return 'zh2en'
    if han < MIN_CN:                    # 兜底2/3: 汉字太少直接按英文处理
        return 'en2zh'
    return 'zh2en' if han >= letters * ZH_RATIO else 'en2zh'


def do_translate(hub, direction, key, text, num_beams):
    """段落 -> 分句 -> 逐句翻译 -> 拼回段落结构(两个方向各自的预处理)"""
    paras = split_paragraphs(text)
    if not paras:
        return '', 0

    t = hub.get(direction, key)
    out_paras, n_sent = [], 0
    for para in paras:
        if direction == 'zh2en':
            sents = split_sentences_cn(para)
            pieces = []
            for s in sents:
                # 先过中文习惯用语词典: 整句命中就直接出英文, 不再过模型
                en_hit, s_to_translate = apply_idioms_cn(s)
                if en_hit:
                    pieces.append(en_hit)
                else:
                    # 【关键】和 08/09 训练时同一套中文预处理
                    out = t.translate(segment_cn(s_to_translate), num_beams=num_beams)
                    if out:
                        pieces.append(out)
            n_sent += len(sents)
            # 英文句子之间要有空格
            if pieces:
                out_paras.append(' '.join(pieces))
        else:
            sents = split_sentences_en(para)
            pieces = []
            for s in sents:
                # 先过习语词典: 整句命中就直接出中文, 否则用改写后的英文再翻译
                cn_hit, s_to_translate = apply_idioms(s)
                out = cn_hit if cn_hit else t.translate(s_to_translate, num_beams=num_beams)
                if out:
                    pieces.append(out)
            n_sent += len(sents)
            # 中文句子顺序拼接即可(模型输出的标点本身就是中文标点)
            if pieces:
                out_paras.append(''.join(pieces))
    return '\n'.join(out_paras), n_sent


# ==================== Flask ====================
app = Flask(__name__, template_folder='templates', static_folder='static')
HUB = None
DEFAULT_DIR = 'en2zh'                 # 默认方向(auto / en2zh / zh2en)
# auto 模式启动时没有文本, 需要先"绑"到某个方向上预热; 同时也是下拉框初始值.
# 取 zh2en(中译英), 因为中文微调模型 (model_marian_zh2en) 体积小且启动快.
DEFAULT_AUTO_DIR = 'zh2en'
DEFAULT_MODEL = {'en2zh': 'pretrained', 'zh2en': 'pretrained'}
# 束搜索宽度(前端已去掉"解码精度"下拉, 固定按这个值请求)
DEFAULT_BEAMS = 8


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/healthz')
def healthz():
    """PaaS 健康检查端点(返回 200 即健康)"""
    return 'ok', 200


@app.route('/api/models')
def api_models():
    """返回两个方向各自的模型列表(前端切换方向时据此刷新下拉框)"""
    refresh_models()      # 让前端始终反映后台下载完成后的最新可用性
    return jsonify({
        # 前端拿到 current_dir 后会用它预热模型下拉; auto 时回退到默认绑定方向
        'current_dir': DEFAULT_DIR if DEFAULT_DIR in MODELS else DEFAULT_AUTO_DIR,
        'default_auto_dir': DEFAULT_AUTO_DIR,
        'current': {d: DEFAULT_MODEL[d] for d in MODELS},
        'device': str(HUB.device) if HUB else 'cpu',
        'labels': DIRECTION_LABELS,
        # 把自动识别的阈值下发给前端, 保证前后端判定规则完全一致
        'detect': {'zh_ratio': ZH_RATIO, 'min_cn': MIN_CN},
        'models': {
            d: [{'key': k, 'label': v['label'], 'available': v['available']}
                for k, v in models.items()]
            for d, models in MODELS.items()
        },
    })


@app.route('/api/translate', methods=['POST'])
def api_translate():
    data = request.get_json(force=True) or {}
    text = (data.get('text') or '').strip()
    if not text:
        return jsonify({'ok': False, 'error': '请输入要翻译的内容'})

    try:
        num_beams = int(data.get('num_beams', DEFAULT_BEAMS))
    except (TypeError, ValueError):
        num_beams = DEFAULT_BEAMS
    num_beams = max(1, min(8, num_beams))

    # 方向: 前端传 en2zh / zh2en / auto, auto 就按输入语言自动判断
    direction = data.get('dir') or DEFAULT_DIR
    if direction not in MODELS:                 # auto 或非法值 -> 自动识别
        direction = detect_direction(text)
    real_dir = direction

    key = data.get('model') or DEFAULT_MODEL[direction]
    if key not in MODELS[direction] or not MODELS[direction][key]['available']:
        key = DEFAULT_MODEL[direction]

    start = time.time()
    try:
        result, n_sent = do_translate(HUB, direction, key, text, num_beams)
    except Exception as e:
        return jsonify({'ok': False, 'error': f'翻译失败: {e}'})

    return jsonify({
        'ok': True,
        'result': result,
        'sentences': n_sent,
        'dir': real_dir,
        'dir_label': DIRECTION_LABELS[real_dir],
        'model': MODELS[direction][key]['label'],
        'device': str(HUB.device) if HUB else 'cpu',
        'elapsed': round(time.time() - start, 2),
    })


# ==================== 启动 ====================
def parse_args():
    p = argparse.ArgumentParser(description='中英互译 Web 翻译服务')
    p.add_argument('--host', default=None, help='绑定地址(默认 0.0.0.0; 也可用 HOST 环境变量)')
    p.add_argument('--port', type=int, default=7860, help='端口(默认 7860; PaaS 用 PORT 环境变量覆盖)')
    p.add_argument('--dir', default='auto', choices=['auto', 'en2zh', 'zh2en'],
                   help='默认方向(auto=按输入语言自动判断)')
    p.add_argument('--model', default=None,
                   help='默认使用哪个模型(仅对默认方向生效: en2zh 可填 finetuned_v2/finetuned/pretrained; '
                        'zh2en 可填 finetuned/pretrained)')
    p.add_argument('--device', default=None, help='cuda / cpu, 不填则自动选择')
    p.add_argument('--zh-ratio', type=float, default=ZH_RATIO,
                   help='自动识别阈值: 汉字数 >= ASCII字母数 * 该值 判为中译英(默认 0.3, 调大=更倾向英译中)')
    p.add_argument('--min-cn', type=int, default=MIN_CN,
                   help='自动识别兜底: 判为中译英所需的最少汉字数(默认 1)')
    return p.parse_args()


def lan_ip():
    """本机在局域网里的 IP(手机要访问就得用这个地址), 拿不到就退回 127.0.0.1"""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))     # 不会真的发包, 只为让系统选出出口网卡
        return s.getsockname()[0]
    except Exception:
        return '127.0.0.1'
    finally:
        s.close()


def pick_default(direction):
    """某个方向上的默认模型: 按 PREFERRED 顺序, 存在就选它"""
    for key in PREFERRED[direction]:
        if MODELS[direction][key]['available']:
            return key
    return 'pretrained'


def refresh_models():
    """重新扫描磁盘, 更新各模型可用性并重新选默认(后台下载完成后调用)"""
    for d, models in MODELS.items():
        for k, v in models.items():
            if k == 'pretrained':
                continue
            v['available'] = bool(v['path']) and _model_exists(v['path'])
    for d in MODELS:
        DEFAULT_MODEL[d] = pick_default(d)


def prepare_models():
    """云端启动前, 若设置了 MODEL_DIR 且模型未就绪, 触发 download_models 拉取"""
    if not DEPLOY_MODEL_DIR:
        return
    try:
        import download_models
        print('=' * 60)
        print('  后台准备模型(MODEL_DIR=%s)...' % DEPLOY_MODEL_DIR)
        download_models.main()
        refresh_models()      # 下载完成后刷新可用性, 让微调模型被识别为可用
    except Exception as e:
        print('  [warn] 模型准备脚本异常: %s' % e)
        print('         服务仍会启动, 但未就绪的方向会退回官方基线(首次请求在线下载约 300MB)')


def boot_worker(boot_dir, boot_key):
    """后台线程: 拉模型 + 加载并预热默认模型, 不阻塞端口绑定(避免被 600s 健康检查卡住)"""
    prepare_models()
    if not DEPLOY_MODEL_DIR:
        return
    # 下载+刷新后, 选当前真正该当默认的模型(微调优先)来预热, 让首次请求不卡
    d = boot_dir
    k = pick_default(d)
    if MODELS[d][k]['available']:
        try:
            print('  正在加载默认模型(%s/%s)...' % (d, k), flush=True)
            t0 = time.time()
            HUB.get(d, k)
            print('  模型加载完毕! 参数量: %.1fM, 耗时: %.1fs'
                  % (HUB.get(d, k).params(), time.time() - t0), flush=True)
            warm = 'hello world' if d == 'en2zh' else segment_cn('你好，世界。')
            HUB.get(d, k).translate(warm)
            print('  预热完成!', flush=True)
        except Exception as e:
            print('  [warn] 默认模型加载/预热失败(首次请求会按需加载): %s' % e, flush=True)


def main():
    global HUB, DEFAULT_DIR, DEFAULT_MODEL, ZH_RATIO, MIN_CN

    args = parse_args()
    ZH_RATIO = max(0.05, min(1.0, args.zh_ratio))
    MIN_CN = max(1, args.min_cn)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    HUB = TranslatorHub(device)

    DEFAULT_DIR = args.dir
    for d in MODELS:
        DEFAULT_MODEL[d] = pick_default(d)
    # --model 只覆盖"默认方向"的那个模型
    if args.model:
        d = DEFAULT_DIR if DEFAULT_DIR in MODELS else 'en2zh'
        if args.model in MODELS[d] and MODELS[d][args.model]['available']:
            DEFAULT_MODEL[d] = args.model

    boot_dir = DEFAULT_DIR if DEFAULT_DIR in MODELS else DEFAULT_AUTO_DIR
    boot_key = DEFAULT_MODEL[boot_dir]

    # 云端模型准备 + 加载放到后台线程: 端口立即绑定, /healthz 秒回, 不被 600s 健康检查卡住
    if DEPLOY_MODEL_DIR:
        threading.Thread(target=boot_worker, args=(boot_dir, boot_key), daemon=True).start()

    print('=' * 60)
    print('  中英互译 Web 翻译服务')
    print('=' * 60)
    print(f'  运行环境 : Python {sys.version.split()[0]} | torch {torch.__version__}')
    print(f'  推理设备 : {device}')
    print(f'  默认方向 : {DIRECTION_LABELS[DEFAULT_DIR]}'
          + (f'  (auto 时绑到 {DIRECTION_LABELS[DEFAULT_AUTO_DIR]})' if DEFAULT_DIR == 'auto' else ''))
    print(f'  自动识别 : 汉字占比阈值 {ZH_RATIO} | 最少汉字数 {MIN_CN}  (--zh-ratio / --min-cn 可调)')
    print(f'  中文预处理: {seg_describe()}')
    for d, models in MODELS.items():
        print(f'  [{DIRECTION_LABELS[d]}]')
        for k, v in models.items():
            mark = '*' if k == DEFAULT_MODEL[d] else ' '
            print(f'    {mark} {k:14s} {"可用" if v["available"] else "未找到"}  {v["label"]}')
    if not MODELS['zh2en']['finetuned']['available']:
        print('  提示: 没找到中译英微调模型 (model_marian_zh2en / $MODEL_DIR/zh2en)')
        print('        首次切到「中译英」会自动下载 Helsinki-NLP/opus-mt-zh-en (约 300MB)')
        print('        想微调它: python 09.中译英_语料微调.py')
    print('  模型准备在后台进行中, 端口已就绪, /healthz 立即可用')
    print('  首次访问翻译接口会在需要时按需加载模型(可能稍慢, 属正常)')

    print('=' * 60)
    port = int(os.environ.get('PORT', args.port))
    host = args.host or os.environ.get('HOST', '0.0.0.0')
    print(f'  本机浏览器: http://127.0.0.1:{port}')
    if host in ('0.0.0.0', '::'):
        print(f'  手机访问  : http://{lan_ip()}:{port}   (手机与电脑连同一个 WiFi)')
    else:
        print(f'  手机访问  : 需加 --host 0.0.0.0 重启, 然后用 http://{lan_ip()}:{port} 访问')
    print('  按 Ctrl+C 停止服务')
    print('=' * 60)

    # debug=False + use_reloader=False: 避免 Flask 的重载器把模型加载两遍
    app.run(host=host, port=port, threaded=True, debug=False, use_reloader=False)


if __name__ == '__main__':
    main()
