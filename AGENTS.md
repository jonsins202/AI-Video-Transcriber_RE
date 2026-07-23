# AI Video Transcriber（RE 版本）

> 面向 AI agent 与开发者的项目说明。
> 本仓库是 [wendy7756/AI-Video-Transcriber](https://github.com/wendy7756/AI-Video-Transcriber) 的二次开发版本（jonsins202 fork）。
> 仓库地址：https://github.com/jonsins202/AI-Video-Transcriber_RE.git

粘贴视频 URL 或上传本地音视频 → ASR 转录 → AI 文本无损优化 + 摘要（+ 翻译）。
支持 YouTube、B站、TikTok、小宇宙、Apple Podcasts 等 30+ 平台。

---

## 与上游的主要差异

1. **ASR 引擎迁移**：`faster-whisper` → **FunASR Paraformer-large**（默认，阿里达摩院中文旗舰 + 自带标点 ct-punc）+ SenseVoice-Small 备选（多语言 + 情绪检测）。切换靠环境变量 `ASR_BACKEND` / `FUNASR_MODEL` 不改代码。
2. **lossless 无损优化引擎**（`backend/lossless_optimizer.py`）：
   - 分段：~300 字目标，边界检测（句号/段落/逗号优先）避免硬切
   - 分批：~2700 字一批 + 200 字上下文，LLM 按 segment id 逐段校对（加标点/修错字，**禁止删改**）
   - **并发**：`LOSSLESS_CONCURRENCY` 默认 32（Semaphore + gather），受 asyncio 线程池 min(32,cpu+4) 限制
   - **429 退避**：`openai.RateLimitError` 指数退避重试（2s x 2^n, cap 20s, jitter, honor Retry-After）
   - **无损保证**：id 逐段对账 + 二分重试 + 三层兜底（`_run_one` → `optimize_transcript` → `assemble_segments`）
3. **段落聚合**（`backend/transcriber.py`）：`FUNASR_PARAGRAPH_PAUSE=1.5s` / `FUNASR_PARAGRAPH_CHARS=300` 解决 Paraformer 句级时间戳导致的每句换行
4. **优化阶段进度回调**（`backend/main.py`）：每批完成更新 progress（55→68），根治"进度卡住假死"
5. **运行日志系统**：`logs/app.log` 按天轮转 30 天 + 任务阶段计时 timeline（`[task:xxx] --> stage` / `<-- stage (Ns)`）
6. **ForcedAligner 开关**：`ENABLE_TIMESTAMPS=False`（关闭可大幅提速，不跑字级对齐）
7. **前端任务耗时展示**（`static/app.js`）：完成后显示总耗时 + 各阶段耗时

## 环境要求

- Python 3.10+
- FFmpeg（系统 PATH）
- NVIDIA GPU + CUDA（推荐；CPU 也能跑但慢）
- 任意 OpenAI 兼容 API key（OpenAI / OpenRouter / 本地 LLM，用于优化 + 摘要 + 翻译）

## 搭建步骤

```bash
git clone https://github.com/jonsins202/AI-Video-Transcriber_RE.git
cd AI-Video-Transcriber_RE

python -m venv venv
venv\Scripts\activate          # Windows PowerShell
# source venv/Scripts/activate # Git Bash

pip install -r requirements.txt
```

## 模型下载

ASR 模型在 `.gitignore`（不进 git）。Paraformer 首次转录时**自动从 ModelScope 下载**（国内源，免代理，约 1.1GB）。SenseVoice 同理，设 `FUNASR_MODEL=sensevoice` 后首次运行自动下载。

如需 `ENABLE_TIMESTAMPS=True`（字级时间戳），还需额外配置（当前默认关闭，不需下载）。

## 关键配置

通过环境变量或 `.env` 文件配置（`backend/` 各模块读取）：

| 变量 | 默认 | 说明 |
|------|------|------|
| `ASR_BACKEND` | `funasr` | ASR 后端（`funasr` / `qwen`） |
| `FUNASR_MODEL` | `paraformer` | FunASR 模型（`paraformer` / `sensevoice`） |
| `FUNASR_PARAGRAPH_PAUSE` | `1.5` | 段落聚合停顿阈值（秒） |
| `FUNASR_PARAGRAPH_CHARS` | `300` | 段落聚合字数上限 |
| `FUNASR_HOTWORD` | （空） | 热词（空格分隔，主要 paraformer 生效） |
| `LOSSLESS_CONCURRENCY` | `32` | 无损优化并发数 |
| `SUMMARY_CONCURRENCY` | `6` | 摘要并发数 |
| `ENABLE_TIMESTAMPS` | `False` | ForcedAligner 字级时间戳开关 |
| `OPENAI_API_KEY` | （UI 填） | API key |
| `OPENAI_BASE_URL` | （UI 填） | API base URL |
| `UPLOAD_MAX_MB` | `200` | 本地上传大小上限（MB） |

## 项目结构

```
backend/
  main.py               # FastAPI 主应用 + 日志系统 + 进度回调 + 阶段计时
  transcriber.py        # ASR 转录（FunASR Paraformer/SenseVoice + 段落聚合 + 热词）
  lossless_optimizer.py # 无损优化引擎（分段/分批/并发/id 对账/保真校验）
  summarizer.py         # AI 文本优化（lossless 并发 + 429 退避）+ 摘要
  translator.py         # LLM 翻译
  video_processor.py    # yt-dlp 下载 + FFmpeg 转码 + cookie/proxy 配置
  llm_sanitize.py       # LLM 输出后处理（去客套话）
static/                 # 前端（HTML + CSS + JS）
tests/                  # unittest 测试（lossless 并发等价、429 退避等）
logs/app.log            # 运行日志（按天轮转 30 天）
temp/                   # 运行产物（音频/转录/摘要，tasks.json）
models/                 # ASR 模型（.gitignore，自动下载）
```

## 启动

```bash
python start.py --prod
```

- 访问 `http://localhost:8000`
- API key 在 UI「AI Settings」里填，或设环境变量 `OPENAI_API_KEY` / `OPENAI_BASE_URL`
- `--prod` 禁用热重载（长任务 SSE 稳定，**长视频必须用**）

> **改代码后必须重启服务** -- `--prod` 模式不热重载，运行中的进程会一直用旧代码。

## 运行日志系统

`logs/app.log` 自动记录全部模块日志，按天轮转保留 30 天。关键格式：

```
[task:1a57b0a5] --> transcribe                    <- 阶段开始
转录完成: 耗时 1214.7s（RTF=0.23）                  <- ASR 性能
[task:1a57b0a5] <-- transcribe (1214.7s)           <- 阶段结束 + 耗时
```

- 查某个任务全流程：`grep "task:1a57b0a5" logs/app.log`
- 每个任务的阶段耗时汇总：`temp/tasks.json` 里的 `timeline` 字段
- 失败的任务：`logger.exception` 记录完整堆栈，直接 grep `Traceback`

## 性能实测

测试环境：RTX 5060 Ti 16GB + CUDA，Paraformer-large，4 小时音频

| 场景 | 耗时 | 说明 |
|------|------|------|
| ASR 转录 | ~2 分钟 | Paraformer CUDA 推理 |
| 无损优化 | ~10-12 分钟 | 32 并发，取决于 LLM 响应速度 |
| 总计 | ~15-20 分钟 | 含优化 + 摘要 |

## 排查速查

| 症状 | 处理 |
|------|------|
| CUDA out of memory | 调小 `ASR_BATCH_SIZE`；CPU 也能跑（慢） |
| yt-dlp 被 B站/抖音 412 限流 | 项目根放 `cookies.txt`（Netscape 格式），或设 `YT_DLP_COOKIES_FROM_BROWSER=chrome` |
| 优化慢/频繁 429 | 日志看"被限流(429)"；降 `LOSSLESS_CONCURRENCY` 到 4-8 |
| 改了代码没生效 | `--prod` 不热重载，重启服务（`Ctrl+C` 重跑 `python start.py --prod`） |
| 摘要功能不可用 | UI「AI Settings」填 API key，或设 `OPENAI_API_KEY` 环境变量 |
| 乱码/ASR 识别错 | 切 Paraformer（默认）；加 `FUNASR_HOTWORD` 热词 |
| 任务失败 | 看 `logs/app.log` 的 `Traceback` |

## 未推送的本地文件（换机器需自备）

| 文件/目录 | 原因 |
|-----------|------|
| `models/` | ASR 模型，自动下载（ModelScope） |
| `cookies.txt` | 登录凭据，敏感 |
| `.env` | API key |
| `venv/` `temp/` `logs/` | 本地运行产物 |

## 测试

```bash
python -m unittest discover -s tests -v
```

测试覆盖：lossless 并发等价性、429 退避恢复/耗尽兜底等。
