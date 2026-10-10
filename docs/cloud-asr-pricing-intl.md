# 云端 ASR API 调研（国际厂商）

> 查询日期：2026-10-10。所有价格以官方定价页/官方文档为准，均已用 FetchURL 逐页核对原文（例外处已注明"未能核实"）。
> 汇率换算按 **1 USD ≈ 7.1 CNY**。
> 背景：本项目（AI Video Transcriber）当前用本地 FunASR Paraformer-large（中文为主，英文/多语言兼有，音频常 1~4 小时）。本文评估改用云端 ASR API 的可行性，核心维度：**价格** 与 **中文准确率**，其次：长音频、说话人分离（diarization）、标点、合规。

---

## 0. 结论速览（TL;DR）

| 结论 | 说明 |
|------|------|
| **最便宜** | Groq `whisper-large-v3-turbo` **$0.04/hr（¥0.28/hr）**，其次 AssemblyAI Universal-2 $0.15/hr、Google V2 dynamic batch $0.18/hr |
| **准确率（官方口径）** | OpenAI `gpt-transcribe` 官方宣称比 whisper-1 错误率低约 52%（第三方转述，OpenAI 未公布 WER 明细表）；Groq 官方标 whisper-large-v3 WER 10.3%、turbo 12%（英文为主的综合基准）；Deepgram/AssemblyAI 官方只给相对排名不给公开 WER 表 |
| **中文是最大坑** | ① 以上所有厂商的准确率宣传几乎都是**英文基准**，中文（尤其带口音、中英混说）需自测；② Deepgram Nova-3 的 `multi` 多语模式仅 10 语言子集，**中文不在其中**，中文实际还在用 Nova-2 代际（第三方集成商实测"静默降级"）；③ 本仓库现用的 Paraformer-large 是阿里中文旗舰模型，中文 CER 上云端通用模型大概率**不如现状** |
| **合规坑** | 全部为境外服务（美国为主），音频出境涉及 PIPL/数据出境评估；Groq/OpenAI 无中国区域；Deepgram 有 EU 端点但无亚太合规优势 |
| **长音频坑** | OpenAI 单文件上限 25MB（≈ 压缩后 1~2 小时勉强、无损格式远远不够），需自行切片；Groq 免费层 25MB / 付费 100MB，官方建议自行 chunking；Deepgram/AssemblyAI/AWS/Azure/Google 的 batch 接口天然支持小时级长音频（URL/S3/GCS 提交） |
| **whisper-1 将停服** | OpenAI 官方弃用公告：`whisper-1`、`gpt-4o-transcribe`、`gpt-4o-mini-transcribe`、`gpt-4o-transcribe-diarize` 均于 **2027-02-26 停服**，替代为 `gpt-transcribe` / `gpt-live-transcribe`。新接入不要选旧模型 |

**对本项目的建议**：若目标是中文长音频转录，云端国际 API 没有"既更准又更便宜"的选项——本地 Paraformer 中文准确率仍是强项，迁移的主要收益是免 GPU 运维和提速。若真要迁，性价比首选 **Groq turbo（¥0.28/hr）** 或 **AssemblyAI（¥1.07/hr 起，中文在 18 语言旗舰列表内）**；准确率优先则实测 **OpenAI gpt-transcribe（¥1.92/hr）** 与 AssemblyAI Universal-3.5 Pro 的中文效果再定。

---

## 1. OpenAI

来源：[官方定价页 Transcription models 表](https://platform.openai.com/docs/pricing)、[Speech-to-text 官方指南](https://platform.openai.com/docs/guides/speech-to-text)、[官方弃用公告](https://developers.openai.com/api/docs/deprecations)（均 2026-10-10 fetch 原文核对）

| 模型 | 价格 | 换算 $/hr | 换算 ¥/hr | 备注 |
|------|------|-----------|-----------|------|
| `gpt-transcribe` | $0.0045/min | $0.27 | ¥1.92 | **当前官方推荐**的文件转录模型 |
| `gpt-4o-transcribe` | $0.006/min | $0.36 | ¥2.56 | 已公告 2027-02-26 停服 |
| `gpt-4o-mini-transcribe` | $0.003/min | $0.18 | ¥1.28 | 已公告 2027-02-26 停服 |
| `whisper-1` | （旧价 $0.006/min，已从新定价表移除） | ~$0.36 | ~¥2.56 | 已公告 2027-02-26 停服；目前仍是**唯一**支持 `timestamp_granularities`（词级时间戳）和 `/audio/translations`（翻译成英文）的模型 |
| `gpt-4o-transcribe-diarize` | 定价页本次未列出（第三方一致引用 $0.006/min，**未能核实**） | ~$0.36? | ~¥2.56? | 说话人分离专用模型；同样 2027-02-26 停服 |
| `gpt-live-transcribe` | $0.017/min | $1.02 | ¥7.24 | 实时流式（Realtime API），非批量场景 |

- **准确性**：官方 2026-07 发布 `gpt-transcribe` 时宣称较 whisper-1 错误率降低约 52%（[第三方报道](https://alphasignal.ai/news/openai-replaces-whisper-with-gpt-transcribe-slashing-errors-by-52)，OpenAI 未公布逐语言 WER 表，**具体数字未能从官方一手页面核实**）。
- **文件上限**：单文件 **25 MB**（官方指南原文），超出需自行切片（官方建议用 PyDub 切，避免句中断开）。对 1~4 小时音频是明显工程负担。
- **语言**：支持 100+ 语言，中文 `zh`/`zh-cn`/`zh-tw`/`zh-hk`、`cmn`、`yue` 均在官方语言代码示例中。`gpt-transcribe` 支持多语言提示（`languages` 复数参数）。
- **说话人分离**：仅 `gpt-4o-transcribe-diarize` 支持（`diarized_json`，支持已知说话人参考音频 2~10s×4 人）；且该模型**已列入停服**。`gpt-transcribe` 不支持 diarization。
- **标点**：自带。
- **免费额度**：无转录专用免费额度（新账号试用金政策随时变动，未核实）。
- **合规**：美国服务，无中国区域；数据出境需自行评估。

## 2. Groq

来源：[Groq 官方 Speech-to-Text 文档](https://console.groq.com/docs/speech-to-text)（2026-10-10 fetch 原文核对）、[whisper-large-v3 模型页](https://console.groq.com/docs/model/whisper-large-v3)、[whisper-large-v3-turbo 模型页](https://console.groq.com/docs/model/whisper-large-v3-turbo)

| 模型 | 价格（官方表格） | 换算 ¥/hr | 官方 WER | 官方实时倍速 |
|------|------------------|-----------|----------|--------------|
| `whisper-large-v3` | $0.111/hr | ¥0.79 | 10.3% | 189x |
| `whisper-large-v3-turbo` | **$0.04/hr** | **¥0.28** | 12% | 216x |

- distil-whisper-large-v3-en：官方文档当前模型表中**已不列出**（早期 $0.02/hr 左右，仅英文，本次未能核实是否仍可用）。
- **WER 口径注意**：官方 10.3%/12% 是综合基准（以英文数据集为主），**中文 WER 会显著更高**（Whisper 系模型中文是弱项，这同时是 Groq 方案的最大风险）。
- **文件上限**：免费层 25MB，付费（dev tier）100MB；官方建议用 `url` 参数传大文件或自行 chunking（官方给了 ffmpeg 压缩命令与切片教程）。最短计费 10 秒。
- **说话人分离**：**不支持**。
- **语言**：Multilingual（Whisper 系 99 语言，含中文）。
- **免费额度**：有免费层（速率/文件大小受限）。
- **合规**：美国服务。

## 3. Deepgram

来源：[官方定价页](https://deepgram.com/pricing)（2026-10-10 fetch 原文核对，流式价格直接确认）；预录价格官方定价页表格为动态加载，数值以 [Deepgram 官方博客](https://deepgram.com/learn/voice-intent-detection-guide)（"STT pricing starts at $0.0043/min for Nova-3"）和 [官方对比文](https://deepgram.com/learn/why-enterprises-are-moving-to-streaming-and-why-whisper-can-t-keep-up)（"$0.0043/min (Mono) and $0.0052/min (Multi)"）交叉确认。

| 模型/模式 | 价格 | 换算 $/hr | 换算 ¥/hr |
|-----------|------|-----------|-----------|
| Nova-3 Monolingual 预录 | $0.0043/min | $0.26 | ¥1.83 |
| Nova-3 Multilingual 预录 | $0.0052/min | $0.31 | ¥2.22 |
| Nova-3 Monolingual 流式（促销价，常规 $0.0077） | $0.0048/min | $0.29 | ¥2.05 |
| Nova-3 Multilingual 流式（促销价，常规 $0.0092） | $0.0058/min | $0.35 | ¥2.47 |

- **中文重大警示**：Nova-3 的 `multi` 多语模式只有 **10 语言子集**（英/西/法/德/印地/俄/葡/日/意/荷级别），**中文不在其中**——第三方集成商实测中文会"静默降级"（[LiveKit 社区](https://community.livekit.io/t/docs-clarify-which-languages-multi-mode-supports-for-deepgram-models/1850)）；另有厂商集成说明"Chinese Mandarin still uses Nova 2"（[Affinity 更新日志](https://support.affinity.co/s/article/Affinity-in-2025)）。官方定价页口径为"Nova 支持 45+ 语言"。**中文用户不建议选 Deepgram。**
- **说话人分离**：支持（Speaker Diarization，预录含在价内；流式为 add-on，第三方引用 +$0.0020/min，**该加价未能从官方页核实**）。
- **长音频**：预录接口支持 URL 提交，无 25MB 级别的硬限制，适合小时级音频。
- **免费额度**：**$200 试用金**（官方定价页，1 年有效）。
- **合规**：SOC 2 Type 1&2、HIPAA、GDPR、CCPA、PCI；有 EU 端点（api.eu.deepgram.com）。美国公司，无中国区域。

## 4. AssemblyAI

来源：[官方定价页](https://www.assemblyai.com/pricing)（2026-10-10 fetch 原文核对）、[官方支持语言文档](https://www.assemblyai.com/docs/pre-recorded-audio/supported-languages)、[官方多语言博客](https://www.assemblyai.com/blog/multilingual-speech-to-text-api)

| 模型 | 价格 | 换算 ¥/hr | 说明 |
|------|------|-----------|------|
| Universal-3.5 Pro（预录旗舰） | $0.21/hr | ¥1.49 | 18 语言旗舰精度（官方列表**含 Mandarin**），原生 code-switching（中英混说友好），最强 diarization |
| Universal-2 | $0.15/hr | ¥1.07 | 99 语言（含中文），12.5M 小时训练数据 |
| Universal-3.6 Pro Realtime（流式） | $0.45/hr | ¥3.20 | 实时场景 |

- **免费额度**：官方定价页注明免费层含 **185 小时预录转录** + 333 小时流式；第三方一致引用新账号 $50 免费额度（未逐条核实具体形态）。
- **说话人分离**：支持（官方称 Universal-3.5 Pro 是"our most accurate speaker diarization yet"）。
- **长音频**：异步 batch 天然面向长录音（会议/播客场景），无 OpenAI 式 25MB 限制。
- **准确率**：官方自称"most accurate"但无公开逐语言 WER 表；第三方基准普遍将其列英文第一梯队，**中文需自测**。
- **合规**：美国服务，无中国区域。

## 5. 三大云（Google / Azure / AWS）

### Google Cloud Speech-to-Text

来源：[官方定价页](https://cloud.google.com/speech-to-text/pricing)（2026-10-10 fetch 原文核对）

| 方案 | 价格 | 换算 $/hr | 换算 ¥/hr |
|------|------|-----------|-----------|
| V2 API Standard（Chirp 等），0~50 万分钟/月 | $0.016/min | $0.96 | ¥6.82 |
| V2 **Dynamic Batch**（低优先级批量，折扣价） | $0.003/min | $0.18 | ¥1.28 |
| V1 含数据日志 | $0.016/min（60 分钟/月免费后） | $0.96 | ¥6.82 |
| V1 不含数据日志 | $0.024/min | $1.44 | ¥10.22 |

- **免费额度**：V1 每月 60 分钟免费。
- **中文**：支持 zh-CN / zh-TW / zh-HK / yue（粤语），Chirp 模型多语言能力强，是三大云里中文口碑相对最好的。
- **说话人分离**：支持（diarization）。
- **长音频**：长音频走 GCS URI + LongRunningRecognize，无时长硬上限（小时级没问题）。
- **合规**：全球区域可选（含香港/台湾区域的基础设施，但仍是美国公司服务）。

### Azure Speech to Text

来源：[官方定价页](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/speech-services/)（2026-10-10 fetch——**价格数字为页面动态加载，本次未能从官方页核实具体金额**；下列数字为多来源一致引用：real-time $1.00/hr、Fast Transcription $0.36/hr、Batch $0.18/hr，见 [tokenprice.fyi](https://tokenprice.fyi/models/azure-ai-speech)、[VexaScribe 对比](https://vexascribe.com/compare/best-transcription-api-for-developers)，标"未能核实"）。免费层 **F0：每月 5 小时实时转录免费**（此项官方页原文确认）。

| 方案 | 价格（未能核实，第三方一致值） | 换算 ¥/hr |
|------|------------------------------|-----------|
| Standard Real-time | ~$1.00/hr（$0.0167/min） | ~¥7.10 |
| Fast Transcription | ~$0.36/hr | ~¥2.56 |
| Batch | ~$0.18/hr（$0.003/min） | ~¥1.28 |

- **中文**：支持 zh-CN 等 100+ 语言，普通话质量在三大云里中上。
- **说话人分离**：支持（Diarization 为 enhanced add-on，实时需另计费）。
- **长音频**：Batch 接口（REST v3.2+）面向长音频。
- 另有 Azure OpenAI 版 whisper 与新的 MAI-Transcribe 系列（价格表已在官方页列出但金额动态加载未核实）。

### AWS Amazon Transcribe

来源：[官方定价页](https://aws.amazon.com/transcribe/pricing/)（2026-10-10 fetch 原文核对；分层价格表为动态加载，以下取自官方页面**定价示例**原文，US East N. Virginia）

| 方案 | 官方示例价格 | 换算 $/hr | 换算 ¥/hr |
|------|--------------|-----------|-----------|
| Batch | $0.006/min | $0.36 | ¥2.56 |
| Streaming | $0.01/min | $0.60 | ¥4.26 |

> 注：官方注明为分层定价、随区域/用量递减，大用量可谈折扣；双声道对话不按声道双倍计费。
> PII 脱敏 add-on：T1 +$0.0024/min。

- **免费额度**：每月 60 分钟，限注册后 12 个月。
- **中文**：官方语言表含 zh-CN（简体）、zh-HK/yue-HK（粤语）等（[官方支持语言文档](https://docs.aws.amazon.com/transcribe/latest/dg/supported-languages.html)）。中文普通话质量一般，业界口碑弱于 Google/Azure。
- **说话人分离**：支持（含在标准价内，官方原文明确 "includes ... speaker diarization"）。
- **长音频**：Batch 走 S3，适合小时级音频。
- **合规**：美国公司，区域众多；中国区域（宁夏/北京）为隔离运营实体，账号不互通。

## 6. 第三方权威 Benchmark

- **HuggingFace Open ASR Leaderboard**（[Space](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)）：开源 ASR 最常被引用的公开榜单。数据集以**英文为主**（LibriSpeech、GigaSpeech、Earnings22、SPGISpeech、TED-LIUM、VoxPopuli、AMI 等），2025 年论文版（[arXiv:2510.06961](https://arxiv.org/abs/2510.06961)）扩展到多语言与长音频评测。**榜单页面为 JS 动态渲染，本次未能抓取实时排名数值**；其历史结论：开源侧 NVIDIA Parakeet / Canary 系列英文 WER 优于 whisper-large-v3，whisper-large-v3 在 Groq 官方口径为 10.3% WER。
- **中文基准**：上述榜单对中文覆盖很弱。中文 ASR 公开对比主要看各模型论文自报（Paraformer、SenseVoice、Qwen-Audio 等在 AISHELL/WenetSpeech 上的 CER），**没有跨"国际云 API vs 中文开源模型"的权威一手榜单**——这是本项目做决策前必须自行实测的原因。
- Groq 官方表格（来源见上）：whisper-large-v3 WER 10.3%、turbo 12%（综合，偏英文口径）。
- OpenAI `gpt-transcribe` "较 whisper-1 错误率 -52%"：OpenAI 发布口径的第三方转述（[AlphaSignal](https://alphasignal.ai/news/openai-replaces-whisper-with-gpt-transcribe-slashing-errors-by-52)），官方未公布逐语言明细，**未能一手核实**。

## 7. 对比总表

| 服务 | 价格 $/hr | 价格 ¥/hr | 免费额度 | 中文支持 | 说话人分离 | 长音频/文件上限 |
|------|-----------|-----------|----------|----------|-----------|------------------|
| Groq whisper-large-v3-turbo | **0.04** | **0.28** | 免费层（25MB/限速） | 支持但 Whisper 中文偏弱 | ❌ | 25MB（免费）/100MB（付费），需自行切片 |
| Groq whisper-large-v3 | 0.111 | 0.79 | 同上 | 同上 | ❌ | 同上 |
| AssemblyAI Universal-2 | 0.15 | 1.07 | 185 小时预录 | 99 语言含中文 | ✅ | batch 面向长音频，无小文件限制 |
| Google V2 Dynamic Batch | 0.18 | 1.28 | V1 60 分钟/月 | zh-CN/TW/HK/粤语，三大云中较好 | ✅ | GCS 长音频无硬上限 |
| Azure Batch | ~0.18（未核实） | ~1.28 | 每月 5 小时（F0） | zh-CN 等 100+，中上 | ✅（add-on） | batch 面向长音频 |
| OpenAI gpt-4o-mini-transcribe | 0.18 | 1.28 | 无 | 支持 | ❌ | **单文件 25MB**（2027-02 停服） |
| AssemblyAI Universal-3.5 Pro | 0.21 | 1.49 | 同上 | 18 语言旗舰含 Mandarin，原生中英混说 | ✅（最强档） | 同上 |
| Deepgram Nova-3 预录 | 0.26~0.31 | 1.83~2.22 | $200 试用金 | ⚠️ multi 模式不含中文，中文仍是 Nova-2 代际 | ✅ | URL 提交，适合长音频 |
| OpenAI gpt-transcribe | 0.27 | 1.92 | 无 | 支持（zh/cmn/yue） | ❌（diarize 模型将停服） | **单文件 25MB**，需自行切片 |
| AWS Transcribe Batch | 0.36 | 2.56 | 60 分钟/月×12 月 | zh-CN/粤语，口碑一般 | ✅（含在价内） | S3 batch，适合长音频 |
| Azure Real-time | ~1.00（未核实） | ~7.10 | 同上 | 同上 | ✅（add-on） | — |
| Google V2 Standard | 0.96 | 6.82 | 同上 | 同上 | ✅ | — |
| 参照：本地 Paraformer-large | ≈0（电费/GPU 折旧） | ≈0 | — | **中文旗舰，当前基准** | cam++（已支持） | 无限制 |

## 8. 数据合规注意（中文用户视角）

- 以上全部为**境外服务**（主要美国），上传中文音视频 = 数据出境。若内容涉及个人信息，需评估《个人信息保护法》出境合规（单独同意、出境评估/标准合同等）；普通公开视频转录风险低，但隐私内容（会议、访谈）需留意。
- 无任何一家提供中国大陆区域；Azure/AWS 的中国区域为隔离运营（21Vianet / 宁夏西云），账号、API、定价均不互通。
- 数据驻留：Deepgram 有 EU 端点；OpenAI 提供部分模型的区域处理（data residency，加价 10%）但不含中国；Google/Azure/AWS 可选亚太区域（新加坡/东京/香港等）降低延迟但法律主体仍为境外。
- 网络可达性：从中国大陆直连 OpenAI/Groq/Deepgram/AssemblyAI/Google API 均需自行解决网络问题，且涉及 ToS 风险（OpenAI 不向中国大陆提供服务）。

---

*未能核实项汇总：Azure 具体价格数字（官方页动态加载）；Deepgram 流式 diarization 加价；OpenAI `gpt-4o-transcribe-diarize` 官方单价；gpt-transcribe 错误率 -52% 的官方一手出处；HF Open ASR Leaderboard 实时排名数值；Groq distil-whisper 当前可用性。*
