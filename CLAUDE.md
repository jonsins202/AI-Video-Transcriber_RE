# AI Video Transcriber（RE 版本）

> 本仓库是 [wendy7756/AI-Video-Transcriber](https://github.com/wendy7756/AI-Video-Transcriber) 的二次开发版本（jonsins202 fork）。
> 仓库地址：https://github.com/jonsins202/AI-Video-Transcriber_RE.git

粘贴视频 URL 或上传本地音视频 → ASR 转录 → AI 文本无损优化 + 摘要（+ 翻译）。
支持 YouTube、B站、TikTok、小宇宙、Apple Podcasts 等 30+ 平台。

---

## 与上游的主要差异

1. **ASR 引擎**：`faster-whisper` → **FunASR Paraformer-large**（默认）+ SenseVoice-Small 备选。环境变量 `ASR_BACKEND`/`FUNASR_MODEL` 切换。
2. **lossless 无损优化引擎**（`backend/lossless_optimizer.py`）：~300 字分段 + ~2700 字分批 + 32 并发 + 429 退避 + id 对账三层兜底，保证不丢字不改义。
3. **段落聚合**（`backend/transcriber.py`）：`FUNASR_PARAGRAPH_PAUSE`/`CHARS` 解决句级时间戳每句换行。
4. **优化进度回调**（`backend/main.py`）：每批完成更新 progress，根治卡住假死。
5. **运行日志**：`logs/app.log` 按天轮转 30 天 + 阶段计时 timeline。
6. **ForcedAligner 开关**：`ENABLE_TIMESTAMPS=False` 默认关闭。

## 环境要求

- Python 3.10+
- FFmpeg（系统 PATH）
- NVIDIA GPU + CUDA（推荐；CPU 也能跑但慢）
- 任意 OpenAI 兼容 API key（优化 + 摘要 + 翻译）

## 启动

```bash
python start.py --prod
```

- 访问 `http://localhost:8000`
- API key 在 UI「AI Settings」填，或设 `OPENAI_API_KEY` / `OPENAI_BASE_URL`
- `--prod` 禁用热重载，长视频 SSE 稳定
- **改代码后必须重启服务**

## 关键配置

环境变量或 `.env`：

| 变量 | 默认 | 说明 |
|------|------|------|
| `ASR_BACKEND` | `funasr` | ASR 后端（`funasr`/`qwen`） |
| `FUNASR_MODEL` | `paraformer` | FunASR 模型（`paraformer`/`sensevoice`） |
| `FUNASR_PARAGRAPH_PAUSE` | `1.5` | 段落聚合停顿阈值（秒） |
| `FUNASR_PARAGRAPH_CHARS` | `300` | 段落聚合字数上限 |
| `FUNASR_HOTWORD` | （空） | 热词（空格分隔，主要 paraformer 生效） |
| `LOSSLESS_CONCURRENCY` | `32` | 无损优化并发数（429 时降到 4-8） |
| `SUMMARY_CONCURRENCY` | `6` | 摘要并发数 |
| `ENABLE_TIMESTAMPS` | `False` | ForcedAligner 字级时间戳 |
| `OPENAI_API_KEY` | （UI 填） | API key |
| `OPENAI_BASE_URL` | （UI 填） | API base URL |
| `UPLOAD_MAX_MB` | `200` | 上传大小上限（MB） |

## 项目结构

```
backend/
  main.py               # FastAPI 主应用 + 日志 + 进度回调 + 阶段计时
  transcriber.py        # ASR 转录（FunASR Paraformer/SenseVoice + 段落聚合 + 热词）
  lossless_optimizer.py # 无损优化引擎（分段/分批/并发/id 对账/保真校验）
  summarizer.py         # AI 优化（lossless 并发 + 429 退避）+ 摘要
  translator.py         # LLM 翻译
  video_processor.py    # yt-dlp 下载 + FFmpeg 转码
  llm_sanitize.py       # LLM 输出后处理
static/                 # 前端（HTML + CSS + JS）
tests/                  # unittest 测试
logs/app.log            # 运行日志（按天轮转 30 天）
temp/                   # 运行产物（tasks.json + 音频/转录）
models/                 # ASR 模型（.gitignore，自动下载）
```

## 开发指南

### 改代码后
1. `Ctrl+C` 停止服务
2. 改代码
3. 重启：`python start.py --prod`

### 运行测试
```bash
python -m unittest discover -s tests -v
```

测试覆盖：lossless 并发等价性、429 退避恢复/耗尽兜底。

### 查看日志
```bash
# 查某个任务全流程
grep "task:1a57b0a5" logs/app.log

# 查失败堆栈
grep "Traceback" logs/app.log
```

### 模型管理
- Paraformer 首次转录自动从 ModelScope 下载（约 1.1GB，国内免代理）
- SenseVoice 同理，设 `FUNASR_MODEL=sensevoice` 后自动下载
- 模型在 `models/` 目录，`.gitignore` 不进 git

## 排查速查

| 症状 | 处理 |
|------|------|
| CUDA out of memory | 调小 `ASR_BATCH_SIZE`；CPU 也行（慢） |
| yt-dlp 412 限流 | 放 `cookies.txt`；或设 `YT_DLP_COOKIES_FROM_BROWSER=chrome` |
| 优化慢/429 | 降 `LOSSLESS_CONCURRENCY` 到 4-8 |
| 改代码没生效 | `--prod` 不热重载，重启服务 |
| 摘要不可用 | UI 填 API key，或设 `OPENAI_API_KEY` |
| ASR 识别错 | 切 Paraformer（默认）；加 `FUNASR_HOTWORD` |
| 任务失败 | 看 `logs/app.log` 的 `Traceback` |

## 未推送的本地文件

| 文件/目录 | 原因 |
|-----------|------|
| `models/` | ASR 模型，自动下载 |
| `cookies.txt` | 登录凭据 |
| `.env` | API key |
| `venv/` `temp/` `logs/` | 本地运行产物 |
