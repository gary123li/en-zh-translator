# 中英互译 Web 服务（Flask + 微调 MarianMT）

把你本地那版 Flask 翻译服务部署到公网。代码推到 **GitHub**，由 PaaS（Zeabur / HuggingFace Spaces / Render）从仓库构建并分配公网域名。

界面、习语兜底、自动识别、中英文分句逻辑全部沿用你原来的 `web/app.py`，只做了**部署化改造**：
- 模型路径由 `MODEL_DIR` 环境变量驱动（本地仍可用 `day06` 布局直接调试）；
- 监听 `$PORT` / `0.0.0.0`，适配 PaaS；
- 启动前自动拉模型（复用 `download_models.py`）；
- 新增 `/healthz` 健康检查端点；
- 可选 `QUANTIZE=1` 做 int8 量化（内存减半）。

---

## 0. 先把模型托管搞定（最关键）

模型两个方向各 ~300MB（合计 620MB），**不能进 Git**（GitHub 单文件上限 100MB）。必须从下面四条路里选一条，在云端把模型拉下来：

| 方案 | 怎么做 | 什么时候选 |
|---|---|---|
| **A. 对象存储直链** | 把 `en2zh/` `zh2en/` 打包成 `models.tar.gz` 传腾讯云 COS / 阿里云 OSS / Cloudflare R2，拿公开 URL | **国内网络首选**，上传快、下载稳 |
| **B. HuggingFace 仓库** | 把模型传成 HF model repo，运行时 `snapshot_download` 拉 | 能上 HF 就用，最省事 |
| **C. 直接打进镜像** | 把模型放进 `models/` 一起 Docker 构建（删掉 `.dockerignore` 对应三行） | 哪儿都上不去时的兜底，镜像会变大 |
| **D. 先用官方基线** | 什么都不设 | 只想先把流程跑通，效果=微调前 |

> 建议顺序：**先 D 跑通页面** → 再换 A/C 上自己的微调模型。分步好排错。

### A. 对象存储直链（推荐，你已选）

已帮你写好打包脚本 `flask-web-deploy/pack_models.py`，它会把本地 `model_marian_v2` / `model_marian_zh2en` 自动重命名为云端需要的 `en2zh/` / `zh2en/` 并打成 `models.tar.gz`：

```bash
cd D:/Python-Project/nlp/day06/flask-web-deploy
python pack_models.py
# -> 生成 ../models.tar.gz  (约 578MB, 根目录直接是 en2zh/ zh2en/)
```

把 `D:\Python-Project\nlp\day06\models.tar.gz` 上传到 **腾讯云 COS / 阿里云 OSS / Cloudflare R2**（任选一家，设为**公共可读**），拿到形如：
```
https://<bucket>.cos.ap-guangzhou.myqcloud.com/models.tar.gz
```

部署时给 Zeabur 设环境变量：
```
MODEL_ARCHIVE_URL="上面这个 URL"
```

> ⚠️ URL 必须是**公开可读**（COS 设公共读 / R2 用 public bucket / OSS 设公共读 ACL）。否则容器拉不下来，会退回官方基线。
> R2 的 public bucket 直接用 `https://<acctid>.r2.cloudflarestorage.com/<bucket>/models.tar.gz` 这类公开 URL 即可，国内也稳。

### B. HuggingFace 仓库

```bash
pip install huggingface_hub -i https://mirrors.aliyun.com/pypi/simple/
huggingface-cli login
# 把两个方向分别上传到 repo 的 en2zh/ 和 zh2en/ 子目录
python -c "from huggingface_hub import HfApi; api=HfApi(); ... "   # 或用 web 上传
```
部署时设 `HF_MODEL_REPO="你的用户名/marian-en-zh"`（私有仓库再加 `HF_TOKEN`）。

### C. 打进镜像（不用任何外部服务）

```bash
cd D:/Python-Project/nlp/day06/flask-web-deploy
mkdir -p models/en2zh models/zh2en
cp -r ../model_marian_v2/*    models/en2zh/
cp -r ../model_marian_zh2en/* models/zh2en/
```
然后**删掉 `.dockerignore` 里的三行**（`models/` `*.safetensors` `*.spm`），构建时就会被带进镜像。`download_models.py` 检测到已存在会跳过下载。

---

## 1. 推到 GitHub

```bash
cd D:/Python-Project/nlp/day06/flask-web-deploy
git init
git add .
git commit -m "中英互译 Flask 服务 v1"
git branch -M main
git remote add origin https://github.com/<你的用户名>/<仓库名>.git
git push -u origin main
```

> ⚠️ **GitHub 直连被墙**（IP 层阻断，改 hosts 没用）。本机推之前给 git 套代理，任选其一：
> ```bash
> # 方案1: 用加速镜像前缀(公益站, 寿命短, 多备几个)
> git remote set-url origin https://ghproxy.net/https://github.com/<用户名>/<仓库>.git
> # 方案2: 本地加速工具 Watt Toolkit / dev-sidecar 之后, 不用改 URL
> # 方案3: 配置 git 走 HTTP 代理
> git config --global http.proxy http://127.00.1:7890
> ```

---

## 2. 从 GitHub 部署（三选一）

### ① Zeabur（推荐，国内近 + 你已有 2GB 专属服务器）

1. 网页端新建 Project，区域选**香港/东京**。
2. 新建 Service 时选 **Deploy from GitHub**（授权后选这个仓库）。
3. 构建完在 **Resources → Memory** 调到 **1.5GB**（1GB + `QUANTIZE=1` 也行）。
4. **Networking → Public** 生成 `xxx.zeabur.app`（永久公网域名）。
5. 环境变量（构建期+运行期都生效，在 Zeabur 该 Service 的 **Environment** 里填）：
   - `MODEL_ARCHIVE_URL` = 你上传的 `models.tar.gz` 公开 URL（方案 A，必填）
   - `QUANTIZE=1`（int8 量化，省内存，强烈建议开）
   - 可选：`DEFAULT_DIR=auto`

### ② HuggingFace Spaces（Docker 模式，ML 原生）

1. 新建 Space，SDK 选 **Docker**，Repo 选 **"Link to a GitHub repo"**（绑定上面那个仓库）。
2. Space 设置里加环境变量：`MODEL_ARCHIVE_URL` 或 `HF_MODEL_REPO`、`QUANTIZE=1`。
3. 分配的资源：免费档 CPU，内存偏紧——建议 `QUANTIZE=1` 且保持单模型常驻（`MAX_RESIDENT_MODELS` 默认 1）。
4. 部署完成得到 `你的用户名-空间名.hf.space`。
   - HF Spaces 国内偶尔不稳，可配 `HF_ENDPOINT` 已是镜像源（Dockerfile 里已设 `hf-mirror.com`）。

### ③ Render（GitHub 原生，但免费档内存小）

1. New → **Web Service** → 连 GitHub 仓库。
2. Runtime 选 **Docker**，实例类型选 **至少 1GB / 2GB RAM**（免费 512MB 跑不动这个模型）。
3. 在 Environment 里加 `MODEL_ARCHIVE_URL` / `HF_MODEL_REPO` / `QUANTIZE=1`。
4. 得到 `xxx.onrender.com`。免费档会冷启动（首次请求慢，要等模型加载）。

---

## 3. 本地验证（不部署也能跑）

```bash
cd D:/Python-Project/nlp/day06/flask-web-deploy
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu   # 或用自己的 venv
# 本地用 day06 布局直接调试(不设 MODEL_DIR 即可)
python app.py
# 浏览器开 http://127.0.0.1:7860
```

想模拟云端布局本地验证：

```bash
MODEL_DIR=./local_models python app.py
# 首次会触发 download_models 拉模型(需设 MODEL_ARCHIVE_URL / HF_MODEL_REPO 或下载官方基线)
```

---

## 4. 环境变量速查

| 变量 | 默认 | 说明 |
|---|---|---|
| `MODEL_DIR` | 不设 | 设了走云端布局 `$MODEL_DIR/en2zh` `$MODEL_DIR/zh2en`；不设退回本地 `day06` 布局 |
| `MODEL_ARCHIVE_URL` | 空 | 模型 tar.gz 直链（COS/OSS/R2），解开需含 `en2zh/` `zh2en/` |
| `HF_MODEL_REPO` | 空 | 你的 HF 模型仓库，形如 `用户名/marian-en-zh` |
| `HF_TOKEN` | 空 | 私有 HF 仓库需要（fine-grained，部署完可 revoke） |
| `HF_ENDPOINT` | `hf-mirror.com` | 国内镜像源，构建/运行下载模型用 |
| `PORT` | `7860` | PaaS 注入，别手动改 |
| `HOST` | `0.0.0.0` | 绑定地址 |
| `QUANTIZE` | `0` | `1` = int8 动态量化（内存减半、更快） |
| `DEFAULT_DIR` | `auto` | `auto` / `en2zh` / `zh2en` |
| `MAX_RESIDENT_MODELS` | — | 见 `download_models`；本 Flask 版默认单模型常驻 |

---

## 5. 排错

| 现象 | 处理 |
|---|---|
| 页面能开但翻译退回"预训练"效果 | 看启动日志有没有 `[models] en2zh <- <你的来源>`；没有就是没拉到微调模型，检查 `MODEL_ARCHIVE_URL` / `HF_MODEL_REPO` |
| 服务起来就 OOM 重启 | 内存调到 1.5GB，或设 `QUANTIZE=1` |
| 打开域名 502 / 一直转 | 还在加载模型（15~30s），等一下刷新 |
| 第一次翻译特别慢 | 正常，模型懒加载 + 预热；之后再翻就快了 |
| 构建卡在下载模型 | 确认 `HF_ENDPOINT` 是镜像源（Dockerfile 已设），或改用 A 方案直链 |
| `git push` 超时 | GitHub 直连被墙，按第 1 节套代理 |
