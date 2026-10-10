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
   - **调用成本控制**：批量大小/输出上限由 `LOSSLESS_BATCH_CHARS`/`LOSSLESS_MAX_TOKENS` 控制；优化/清理结果落盘 `temp/optimize_cache/`（`optimize_cache.py`，键=正文哈希+模型+档位，同键重跑零 LLM 调用）
3. **口语清理模式（可选）**（`backend/summarizer.py` `cleanup_transcript_body`）：
   - UI「口语清理」勾选（默认关），管线变为 转录 → 无损优化 → 口语清理（删呃/嗯/啊等填充词、顺句）→ 翻译/摘要
   - 复用 lossless 引擎（分段/分批/并发/id 对账/三层兜底），只换 prompt + 校验档位
   - **非逐字**：`CLEANUP_LIMITS`（单跨度 ≤40、删 ≤30%、改 ≤40%、插 ≤10% + 首尾锚点防截断）；任何未验证段回退优化稿
   - 启用时无损优化稿另存 `temp/verbatim_*.md`，审计 `cleanup_audit_*.json`
4. **段落聚合**（`backend/transcriber.py`）：`FUNASR_PARAGRAPH_PAUSE=1.5s` / `FUNASR_PARAGRAPH_CHARS=300` 解决 Paraformer 句级时间戳导致的每句换行
5. **优化阶段进度回调**（`backend/main.py`）：每批完成更新 progress（优化 55→63、清理 65→72、翻译 75、摘要 85），根治"进度卡住假死"
6. **运行日志系统**：`logs/app.log` 按天轮转 30 天 + 任务阶段计时 timeline（`[task:xxx] --> stage` / `<-- stage (Ns)`）
7. **ForcedAligner 开关**：`ENABLE_TIMESTAMPS=False`（关闭可大幅提速，不跑字级对齐）
8. **前端任务耗时展示**（`static/app.js`）：完成后显示总耗时 + 各阶段耗时
9. **媒体缓存**（`backend/media_cache.py`）：同 URL 重复测试不同 LLM 时免重复下载/转录
   - 两级缓存：raw 级（零网络，直接进优化管线，任务记录 `cache_hit="raw"`）/ audio 级（跳过下载重转录，`cache_hit="audio"`）
   - 索引 `temp/media_cache.json` 映射 url → 音频/raw 文件（文件名含随机 uuid 无法反查，靠索引）
   - **ASR 指纹失效**：`ASR_BACKEND`/`FUNASR_MODEL`/`FUNASR_HOTWORD`/`FUNASR_LANG`/段落参数/`FUNASR_SPK` 任一变化，raw 自动重转录；音频不受影响；字幕来源固定 key `"subtitle"`
   - 文件缺失惰性清理；`MEDIA_CACHE=0` 关闭；强制重新下载 = 删 `temp/media_cache.json`（或对应缓存文件）
10. **说话人分离**（`backend/transcriber.py`，diarization）：`FUNASR_SPK=cam++` 开启（仅 Paraformer 路径，SenseVoice 无时间戳不支持）
    - cam++ 声纹嵌入 + 自动聚类，逐句输出说话人标签；段落聚合规则：**说话人变化 > 停顿 > 字数**，段落带「说话人A：」前缀
    - `FUNASR_SPK_NUM=2` 可固定人数（UI「说话人数」输入框按请求覆盖，参数优先于环境变量）；日志输出"识别到 N 个说话人"
    - 优化/清理 prompt 已加"保留说话人标签"约束；媒体缓存指纹已含开关
    - **说话人命名面板**（前端 + `POST /api/task/{id}/speakers`）：完成的文稿含「说话人X：」标签时，结果区出现命名面板，填真名一键替换（可重复修改、清空回退标签；文稿文件同步重写，摘要不回改）
11. **云端 ASR 后端**（`backend/transcriber.py` `_DashScopeBackend`）：`ASR_BACKEND=dashscope` 启用，免 GPU 依赖
    - 流程：本地音频 → OSS 上传 → 签名 URL（6h 有效）提交异步转写 → 轮询任务 → 下载结果 JSON → 删除 OSS 对象（OSS 桶另配 1 天生命周期兜底）
    - 默认模型 `qwen-audio-3.0-asr-flash-filetrans`（≤12h/2GB）；源站直链（B站/YouTube）有防盗链/签名过期问题**不可直接传**，必须经 OSS 中转
    - 说话人分离：`diarization_enabled` **默认开启**（云端自动聚类识别人数，无需预填；`DASHSCOPE_DIARIZATION=0` 关闭；UI 说话人数按请求传 `speaker_count`）。官方建议 ≤2h，超长自动切片：ffmpeg 切成 ≤110 分钟片（`DASHSCOPE_CHUNK_MAX_MIN`，重叠 30s `DASHSCOPE_CHUNK_OVERLAP_S`），并发转写后合并——接缝按重叠区中点去重，说话人跨片用接缝带文本相似度（≥0.5）匹配重映射，匹配不到分配新编号
    - 已显式关闭系统敏感词过滤（`special_word_filter.system_reserved_filter=false`），否则文稿会被打码 `*`；`transcription_url` 仅 24h 有效需立即落盘
    - 媒体缓存指纹已含 `DASHSCOPE_ASR_MODEL`/`DASHSCOPE_DIARIZATION`/`DASHSCOPE_LANGUAGE_HINTS`

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
| `ASR_BACKEND` | `funasr` | ASR 后端（`funasr` / `qwen` / `dashscope`） |
| `FUNASR_MODEL` | `paraformer` | FunASR 模型（`paraformer` / `sensevoice`） |
| `FUNASR_PARAGRAPH_PAUSE` | `1.5` | 段落聚合停顿阈值（秒） |
| `FUNASR_PARAGRAPH_CHARS` | `300` | 段落聚合字数上限 |
| `FUNASR_HOTWORD` | （空） | 热词（空格分隔，主要 paraformer 生效） |
| `FUNASR_SPK` | （空） | 说话人分离模型（填 `cam++` 开启；仅 paraformer 路径） |
| `FUNASR_SPK_NUM` | （空） | 固定说话人数（已知人数时设，聚类更稳；UI「说话人数」输入框按请求覆盖此值） |
| `DASHSCOPE_API_KEY` | （必填） | 百炼 API Key（仅 `ASR_BACKEND=dashscope` 时） |
| `DASHSCOPE_WORKSPACE` | （必填） | 百炼业务空间 ID（决定接入点域名，如 `llm-xxxx`） |
| `DASHSCOPE_REGION` | `cn-beijing` | DashScope 地域 |
| `DASHSCOPE_ASR_MODEL` | `qwen-audio-3.0-asr-flash-filetrans` | 云端 ASR 模型（异步文件转写，≤12h/2GB） |
| `DASHSCOPE_DIARIZATION` | `1` | 云端说话人分离（默认开，自动聚类人数，无需预填；`=0` 关闭） |
| `DASHSCOPE_CHUNK_MAX_MIN` | `110` | 开分离时单片上限（分钟，官方建议 ≤2h）；超长自动带重叠切片并发转写合并 |
| `DASHSCOPE_LANGUAGE_HINTS` | `zh,en` | 云端识别语言提示（逗号分隔） |
| `OSS_BUCKET` 等四项 | （必填） | 云端 ASR 中转桶（`OSS_BUCKET`/`OSS_ENDPOINT`/`OSS_ACCESS_KEY_ID`/`OSS_ACCESS_KEY_SECRET`） |
| `SKIP_OPTIMIZE` | （空） | `=1` 跳过无损优化阶段（云端 ASR 输出质量高时省一次 LLM 调用；口语清理/摘要不受影响） |
| `LOSSLESS_CONCURRENCY` | `32` | 无损优化并发数 |
| `LOSSLESS_BATCH_CHARS` | `5000` | 无损优化单批字数（原 2700；配合 `LOSSLESS_MAX_TOKENS` 提升可近半减少调用次数，质量由 id 对账+二分+回退兜底） |
| `LOSSLESS_MAX_TOKENS` | `8000` | 无损优化单次调用输出上限（模型不支持 8k 输出时请调低，否则全部回退原文，看 audit `fallback_api`） |
| `OPTIMIZE_CACHE` | `1` | 优化结果缓存：同 (正文, 模型, 档位) 跳过优化/清理阶段；改 prompt 需 bump `optimize_cache.py` 里 profile 版本号 |
| `SUMMARY_CONCURRENCY` | `6` | 摘要并发数 |
| `ENABLE_TIMESTAMPS` | `False` | ForcedAligner 字级时间戳开关 |
| `OPENAI_API_KEY` | （UI 填） | API key |
| `OPENAI_BASE_URL` | （UI 填） | API base URL |
| `UPLOAD_MAX_MB` | `200` | 本地上传大小上限（MB） |
| （UI 勾选）`oral_cleanup` | `false` | 口语清理模式：删填充词/顺句（非逐字，走 CLEANUP 校验档） |
| `MEDIA_CACHE` | `1` | 媒体缓存开关：同 URL 复用音频/raw 转录，`0` 关闭 |
| `PORT` | `9000` | 服务监听端口（`.env` / 环境变量可改；临时换端口用 `python start.py --prod --port N`） |

## 项目结构

```
backend/
  main.py               # FastAPI 主应用 + 日志系统 + 进度回调 + 阶段计时
  transcriber.py        # ASR 转录（FunASR Paraformer/SenseVoice + Qwen3-ASR 本地 + DashScope 云端 + 段落聚合 + 热词）
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

- 访问 `http://localhost:9000`
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
| 对话音频分不清谁在说 | 设 `FUNASR_SPK=cam++`（双人访谈再加 `FUNASR_SPK_NUM=2`） |
| 想强制重新下载/转录 | 删 `temp/media_cache.json`（或对应 `cache_raw_*.md`/`audio_*.wav`） |
| 优化全回退原文（audit 全 `fallback_api`） | 模型不支持 8k 输出，调低 `LOSSLESS_MAX_TOKENS` |
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

测试覆盖：lossless 并发等价性、429 退避恢复/耗尽兜底、CLEANUP 档校验（放行删填充词/拦截改写截断）、口语清理管线回退、媒体缓存（索引/失效/任务流接线）、说话人段落聚合等。
