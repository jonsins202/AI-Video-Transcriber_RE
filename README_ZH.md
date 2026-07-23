<div align="center">

# AI Video Transcriber (RE)

中文 | [English](README.md)

一款开源的 AI 视频/播客转录与摘要工具：粘贴 URL 或上传本地音视频 → ASR 转录 → AI 无损优化 + 摘要（+ 翻译）。

支持 YouTube、Bilibili、抖音、TikTok、小宇宙、Apple Podcasts、SoundCloud 等 30+ 平台。

![Interface](en_video.png)

</div>

## 与上游的主要差异

本仓库是 [wendy7756/AI-Video-Transcriber](https://github.com/wendy7756/AI-Video-Transcriber) 的二次开发版本（RE 版本）。核心改进：

| 改进点 | 上游原版 | RE 版本 |
|--------|----------|---------|
| ASR 引擎 | Faster-Whisper | **FunASR Paraformer-large**（默认）+ SenseVoice-Small 备选 |
| 中文转录质量 | 一般 | 显著提升（阿里达摩院中文旗舰模型） |
| 文本优化 | 逐句简单校对 | **lossless 无损优化引擎**（并发 + id 对账 + 三层兜底，保证不丢字不改义） |
| 段落排版 | 无 | 段落聚合（`FUNASR_PARAGRAPH_PAUSE` + `CHARS`），解决每句换行问题 |
| 进度反馈 | 优化阶段卡住假死 | 优化阶段实时进度回调（55→68） |
| 运行日志 | 无 | `logs/app.log` 按天轮转 30 天 + 任务阶段计时 timeline |
| ASR 切换 | 改代码 | 环境变量 `ASR_BACKEND` / `FUNASR_MODEL` 即时切换 |

## 功能特性

- **多平台支持**：YouTube、Bilibili、抖音、TikTok、小宇宙、Apple Podcasts 等 30+ 平台
- **本地上传**：支持 `.txt`、`.mp3`、`.mp4`、`.m4a`、`.wav`、`.webm`、`.mkv`、`.ogg`、`.flac`，默认上限 200MB
- **字幕优先**：有原生字幕的平台直接提取，无需下载音频；无字幕时回退 ASR
- **FunASR 中文转录**：Paraformer-large 自带标点模型（ct-punc），中文效果显著优于 Whisper
- **无损优化引擎**：LLM 逐段校对（加标点/修错字，禁止删改），32 并发 + 429 退避 + id 对账保真
- **段落聚合**：根据停顿阈值和字数上限智能合并句级时间戳为自然段落
- **多语言摘要**：支持多种语言的智能摘要生成
- **条件式翻译**：摘要语言与转录语言不一致时自动生成翻译
- **自定义 AI 模型**：在 UI 中配置任意 OpenAI 兼容接口（OpenAI、OpenRouter、本地 LLM 等）
- **前端耗时展示**：完成后显示总耗时 + 各阶段耗时
- **移动适配**：完美支持移动设备

## 快速开始

### 环境要求

- Python 3.10+
- FFmpeg（系统 PATH 中）
- NVIDIA GPU + CUDA（推荐；CPU 也能跑但慢）
- 任意 OpenAI 兼容 API key（用于优化 + 摘要 + 翻译）

### 安装

```bash
git clone https://github.com/jonsins202/AI-Video-Transcriber_RE.git
cd AI-Video-Transcriber_RE

python -m venv venv
venv\Scripts\activate          # Windows PowerShell
# source venv/Scripts/activate # Git Bash

pip install -r requirements.txt
```

### 模型下载

ASR 模型在 `.gitignore` 中，需单独下载。Paraformer 首次转录时也会自动从 ModelScope 下载（国内源，免代理，约 1.1GB）。如需手动预下载：

```bash
# FunASR Paraformer-large（默认，推荐）
# 首次运行时自动下载，也可手动：
# ModelScope 国内源自动，HuggingFace 需代理

# FunASR SenseVoice-Small（备选，多语言 + 情绪检测）
# 设置 FUNASR_MODEL=sensevoice 时自动下载
```

### 启动

```bash
python start.py --prod
```

- 访问 `http://localhost:8000`
- API key 在页面 **AI Settings** 中填写，或设环境变量 `OPENAI_API_KEY` / `OPENAI_BASE_URL`
- `--prod` 禁用热重载，保持长任务 SSE 连接稳定（**长视频必须用**）

> **改代码后必须重启服务** -- `--prod` 模式不热重载。

## 使用指南

1. **选择输入方式**
   - **URL**：粘贴 YouTube、Bilibili 等链接
   - **本地文件**：拖放或点击上传
2. **选择摘要语言**：下拉菜单选择输出语言
3. **（可选）配置 AI 模型**：点击 **AI Settings**，填 API Base URL 和 Key，点 **Fetch** 拉取模型列表
4. **点击 Transcribe**
   - **字幕模式**（绿色）：有原生字幕，秒级完成
   - **ASR 模式**（橙色）：无字幕，下载音频后转录
5. **查看结果**：优化后的转录文本 + AI 摘要，语言不一致时自动显示翻译
6. **下载**：保存 Markdown 格式文件（转录 / 翻译 / 摘要）

## 技术架构

### 后端

| 组件 | 技术 |
|------|------|
| Web 框架 | FastAPI |
| ASR 转录 | FunASR Paraformer-large（默认）/ SenseVoice-Small |
| 无损优化 | `lossless_optimizer.py`（分段/分批/并发/id 对账） |
| 视频下载 | yt-dlp + FFmpeg |
| AI 文本处理 | OpenAI 兼容 API（优化 + 摘要 + 翻译） |

### 前端

HTML5 + CSS3 + JavaScript（Marked.js + Font Awesome）

### 项目结构

```
backend/
  main.py               # FastAPI 主应用 + 日志系统 + 进度回调 + 阶段计时
  transcriber.py        # ASR 转录（FunASR Paraformer/SenseVoice + 段落聚合 + 热词）
  lossless_optimizer.py # 无损优化引擎（分段/分批/并发/id 对账/保真校验）
  summarizer.py         # AI 文本优化（lossless 并发 + 429 退避）+ 摘要
  translator.py         # LLM 翻译
  video_processor.py    # yt-dlp 下载 + FFmpeg 转码
  llm_sanitize.py       # LLM 输出后处理
static/                 # 前端（HTML + CSS + JS）
tests/                  # unittest 测试
logs/app.log            # 运行日志（按天轮转 30 天）
temp/                   # 运行产物（音频/转录/摘要）
models/                 # ASR 模型（.gitignore，需单独下载）
```

## 配置选项

以下变量通过环境变量或 `.env` 文件配置：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ASR_BACKEND` | `funasr` | ASR 后端（`funasr` / `qwen`） |
| `FUNASR_MODEL` | `paraformer` | FunASR 模型（`paraformer` / `sensevoice`） |
| `FUNASR_PARAGRAPH_PAUSE` | `1.5` | 段落聚合停顿阈值（秒） |
| `FUNASR_PARAGRAPH_CHARS` | `300` | 段落聚合字数上限 |
| `FUNASR_HOTWORD` | （空） | 热词（空格分隔，主要 paraformer 生效） |
| `LOSSLESS_CONCURRENCY` | `32` | 无损优化并发数（遇 429 可降到 4-8） |
| `SUMMARY_CONCURRENCY` | `6` | 摘要并发数 |
| `ENABLE_TIMESTAMPS` | `False` | ForcedAligner 字级时间戳开关 |
| `OPENAI_API_KEY` | （UI 填） | API key |
| `OPENAI_BASE_URL` | （UI 填） | API base URL |
| `UPLOAD_MAX_MB` | `200` | 本地上传单文件大小上限（MB） |

## 性能实测

测试环境：RTX 5060 Ti 16GB + CUDA，Paraformer-large

| 场景 | 耗时 | 说明 |
|------|------|------|
| ASR 转录 | ~2 分钟 / 4 小时音频 | Paraformer CUDA 推理 |
| 无损优化 | ~10-12 分钟 / 4 小时音频 | 32 并发，取决于 LLM 响应速度 |
| 总计 | ~15-20 分钟 / 4 小时音频 | 含优化 + 摘要 |

## 常见问题

### Q: 转录出现乱码 / 识别错误多？
默认使用 Paraformer（中文效果好）。如有问题，可加 `FUNASR_HOTWORD` 热词提高专有名词识别率。

### Q: 优化很慢 / 频繁 429？
日志中出现"被限流(429)"说明 LLM API 限流。将 `LOSSLESS_CONCURRENCY` 降到 4-8 可缓解。

### Q: CUDA out of memory？
调小 `ASR_BATCH_SIZE`。CPU 模式也能跑，只是慢。

### Q: yt-dlp 被 B站/抖音限流（412）？
在项目根目录放 `cookies.txt`（Netscape 格式），或设 `YT_DLP_COOKIES_FROM_BROWSER=chrome`。

### Q: AI 优化 / 摘要功能不可用？
在页面 **AI Settings** 中填写 API key，或设 `OPENAI_API_KEY` 环境变量。

### Q: 改了代码没生效？
`--prod` 不热重载，需要重启服务：`Ctrl+C` 后重跑 `python start.py --prod`。

### Q: 任务失败了？
查看 `logs/app.log` 中的 `Traceback`，或查看 `temp/tasks.json` 中对应任务的状态。

## 运行日志

`logs/app.log` 自动记录全部模块日志，按天轮转保留 30 天。日志格式：

```
[task:1a57b0a5] --> transcribe                    <-- 阶段开始
转录完成: 耗时 1214.7s（RTF=0.23）                  <-- ASR 性能
[task:1a57b0a5] <-- transcribe (1214.7s)           <-- 阶段结束 + 耗时
```

- 查某个任务全流程：`grep "task:1a57b0a5" logs/app.log`
- 失败任务：grep `Traceback` 查看完整堆栈

## 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 致谢

- [wendy7756/AI-Video-Transcriber](https://github.com/wendy7756/AI-Video-Transcriber) -- 上游原版项目
- [FunASR](https://github.com/modelscope/FunASR) -- 阿里达摩院语音识别框架
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) -- 强大的视频下载工具
- [FastAPI](https://fastapi.tiangolo.com/) -- 现代化的 Python Web 框架
- [OpenAI](https://openai.com/) -- 智能文本处理 API

## 联系方式

如有问题或建议，请 [提交 Issue](https://github.com/jonsins202/AI-Video-Transcriber_RE/issues)。
