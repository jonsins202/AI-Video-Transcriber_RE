# AI Video Transcriber（RE 版本）

> 面向 AI agent 与开发者的项目说明。
> 本仓库是 [wendy7756/AI-Video-Transcriber](https://github.com/wendy7756/AI-Video-Transcriber) 的二次开发版本（jonsins202 fork）。

粘贴视频 URL 或上传本地音视频 → ASR 转录 → AI 文本优化 + 摘要（+ 翻译）。
支持 YouTube、B站、TikTok、Apple Podcasts 等 30+ 平台。

---

## 与上游的主要差异

1. **ASR 引擎迁移**：`faster-whisper` → **Qwen3-ASR-0.6B**（中文效果更好）
2. **batch 推理优化**：删除原手动 60s 分段，改用 qwen_asr 原生 batch 推理（长音频提速）
3. **运行日志系统**：`logs/app.log` 按天轮转 30 天 + 任务阶段计时（timeline）+ 错误堆栈
4. **ForcedAligner 开关**：新增 `ENABLE_TIMESTAMPS`（关闭可大幅提速，见下）

## 环境要求

- Python 3.10+
- FFmpeg（系统 PATH）
- NVIDIA GPU + CUDA（CPU 也能跑但慢）
- 任意 OpenAI 兼容 API key（OpenAI / OpenRouter / 本地 LLM，用于摘要与翻译）

## 搭建步骤

```bash
git clone https://github.com/jonsins202/AI-Video-Transcriber_RE.git
cd AI-Video-Transcriber_RE

python -m venv venv
source venv/Scripts/activate     # Git Bash；PowerShell 用 venv\Scripts\activate

pip install -r requirements.txt
```

## 模型下载（必需，未随仓库推送）

模型在 `.gitignore`（几 GB，不进 git）。当前默认配置（`ENABLE_TIMESTAMPS=False`）**只需 ASR 模型**：

```bash
pip install huggingface_hub

# 必需：Qwen3-ASR-0.6B（约 1.2GB），HuggingFace 仓库 Qwen/Qwen3-ASR-0.6B
huggingface-cli download Qwen/Qwen3-ASR-0.6B --local-dir models/Qwen3-ASR-0.6B

# 可选：Qwen3-ForcedAligner-0.6B（仅 ENABLE_TIMESTAMPS=True 时需要）
# huggingface-cli download Qwen/Qwen3-ForcedAligner-0.6B --local-dir models/Qwen3-ForcedAligner-0.6B
```

国内访问慢可设镜像：`export HF_ENDPOINT=https://hf-mirror.com`

## 关键配置（`backend/transcriber.py` 顶部）

```python
ENABLE_TIMESTAMPS = False   # 字级时间戳开关（False = 转录文本不含 [00:00] 标记，更快）
ASR_BATCH_SIZE = 2          # batch 大小，按显存调整
```

| `ENABLE_TIMESTAMPS` | 音频块大小 | 8GB 显存 | 16GB+ 显存 | 特点 |
|---|---|---|---|---|
| `False`（当前默认） | 1200s/块 | ❌ 必 OOM | batch=2-4 | 快，无时间戳 |
| `True` | 180s/块 | batch=4 | batch=6-8 | 有时间戳；ForcedAligner 在 CPU 是瓶颈 |

> **8GB 显存唯一可行配置**：`ENABLE_TIMESTAMPS=True` + `ASR_BATCH_SIZE=4`（RTF≈0.23）。
> 关 ForcedAligner 路线（1200s/块）单块就要 8.86GB 显存，8GB 必 OOM，需 ≥16GB。

## 启动

```bash
python start.py --prod
```

- 访问 `http://localhost:8000`
- API key 在 UI「AI Settings」里填，或设环境变量 `OPENAI_API_KEY` / `OPENAI_BASE_URL`
- `--prod` 禁用热重载（长任务 SSE 稳定，**长视频必须用**）

> ⚠️ **改代码后必须重启服务** —— `--prod` 模式不热重载，运行中的进程会一直用旧代码。

## 运行日志系统

`logs/app.log` 自动记录全部模块日志，按天轮转保留 30 天。关键格式：

```
[task:1a57b0a5] --> transcribe                    ← 阶段开始
转录完成: 耗时 1214.7s（RTF=0.23, 4.3x 实时）       ← ASR 性能
[task:1a57b0a5] <-- transcribe (1214.7s)           ← 阶段结束 + 耗时
```

- 查某个任务全流程：`grep "task:1a57b0a5" logs/app.log`
- 每个任务的阶段耗时汇总：`temp/tasks.json` 里的 `timeline` 字段
- 失败的任务：`logger.exception` 记录完整堆栈，直接 grep `Traceback`

## 性能基线（RTF = 处理耗时 / 音频时长）

测试音频：B站 Nathan Lambe 专访 87 分钟（`BV1HXTs6bEzH`）

| 配置 | 显存 | 转录 RTF | 实时倍率 |
|---|---|---|---|
| 上游原版（faster-whisper，手动 60s 分段） | 8GB | 0.30-0.32 | 3.1-3.3x |
| `batch=4 + ForcedAligner(CPU)` | 8GB | 0.23 | 4.3x |
| `batch=2 + 关 ForcedAligner` | 16GB+ | 待测 | 预期 0.08-0.12 |

## 排查速查

| 症状 | 处理 |
|---|---|
| ASR `CUDA out of memory` | 调小 `ASR_BATCH_SIZE`；8GB 显存必须设 `ENABLE_TIMESTAMPS=True`（180s 块更省显存） |
| yt-dlp 被 B站/抖音 412 限流 | 项目根放 `cookies.txt`（Netscape 格式），或设 `YT_DLP_COOKIES_FROM_BROWSER=chrome` |
| 任务失败 | 看 `logs/app.log` 的 `Traceback` |
| 改了代码没生效 | `--prod` 不热重载，重启服务（`Ctrl+C` 重跑 `python start.py --prod`） |
| 摘要功能不可用 | UI「AI Settings」填 API key，或设 `OPENAI_API_KEY` 环境变量 |

## 未推送的本地文件（换机器需自备）

| 文件/目录 | 原因 |
|---|---|
| `models/` | 几 GB，单独下载（见上） |
| `cookies.txt` | 登录凭据，敏感 |
| `.env` | API key |
| `venv/` `temp/` `logs/` | 本地运行产物 |
| `启动服务.bat` | 已推送，但内含硬编码路径 `K:\AI-Video-Transcriber`，换机器需改 |

## 项目结构

```
backend/
  main.py              # FastAPI 主应用 + 日志系统 + 阶段计时
  transcriber.py       # Qwen3-ASR 转录（batch 推理，关键配置在此）
  video_processor.py   # yt-dlp 下载 + FFmpeg 转码 + cookie/proxy 配置
  summarizer.py        # LLM 文本优化与摘要
  translator.py        # LLM 翻译
  llm_sanitize.py      # LLM 输出后处理（去客套话）
static/                # 前端（HTML + JS）
logs/app.log           # 运行日志（自动生成）
temp/                  # 音频/转录产物（运行时生成）
models/                # ASR 模型（需单独下载）
```
