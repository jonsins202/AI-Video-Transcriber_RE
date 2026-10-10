# 阿里百炼（DashScope）ASR 模型代际调研：paraformer-v2 过时了吗？

> 查询日期：2026-10-10。来源均为官方一手资料：阿里云百炼官方文档/定价页（help.aliyun.com）、ModelScope 官方模型卡、QwenLM 官方 GitHub。
> 问题背景：有观点认为 paraformer-v2 是 2024 年的模型、可能过时。本文核实百炼当前在售 ASR lineup、官方推荐关系、准确率数据，并给选型结论。

**一句话结论：是的，paraformer-v2 已被官方明确标记为"较早一代"。** 百炼官方语音识别选型页原文：*"Paraformer 是较早一代的 ASR 模型……若您的业务允许，建议迁移到前文推荐的 Fun-ASR 或 Qwen-ASR。"*（[语音识别-选型](https://help.aliyun.com/zh/model-studio/asr-model)，查询日期 2026-10-10）。它没下架、仍便宜且每月送 10 小时，但已不是推荐项。

---

## 1. 当前在售 ASR 模型 lineup（按代际，新→旧）

来源：[百炼-语音识别选型](https://help.aliyun.com/zh/model-studio/asr-model)、[模型调用价格](https://help.aliyun.com/zh/model-studio/model-pricing)、[非实时语音识别用户指南](https://help.aliyun.com/zh/model-studio/non-realtime-speech-recognition-user-guide)（均查询于 2026-10-10）

### ① Qwen-Audio-3.x-ASR（当前官方首推，最新代）

| 模型 ID | 模式 | 计费（华北2-北京） | 说话人分离 | 精度增强 | 时长/大小上限 |
|---|---|---|---|---|---|
| `qwen-audio-3.1-asr-flash-filetrans` | 非实时（HTTP 异步） | **按 token**：输入 0.8 元/百万 + 输出 2.7 元/百万（免费 100 万 token/90 天） | ✅ | 热词 + Prompt 上下文 | 12 小时 / 2GB |
| `qwen-audio-3.0-asr-flash-filetrans` | 非实时（HTTP 异步） | 0.00022 元/秒 = 0.792 元/小时 | ✅ | 热词 + Prompt 上下文 | 12 小时 / 2GB |
| `qwen-audio-3.1-asr-flash-streaming` | 实时（WebSocket） | 按 token：输入 0.8 + 输出 2.7 元/百万 | ❌ | 热词 + Prompt 上下文 | 不限 |
| `qwen-audio-3.1-asr-flash` | 非实时同步（≤5 分钟） | 按 token：同上 | ✅ | 热词 + Prompt 上下文 | 5 分钟 |

- 支持语言：30 个语种 + 中文 8 大方言体系/20+ 地区口音（选型页列明，含粤语、吴语、闽南语、客家话等）。
- **3.1 是官方"推荐模型"栏里唯一的文件转写模型**；注意 3.1 系按 token 计费（与 3.0 按秒计费不同）。音频 token 折算规则：官方"千问Audio"计费说明为**每秒音频 = 25 token**（[模型调用价格](https://help.aliyun.com/zh/model-studio/model-pricing)）；据此粗算输入侧约 0.072 元/小时（90000 token × 0.8 元/百万），加上输出文本 token（普通话语音每小时约 1~3 万汉字，输出 token 约 2~6 万 × 2.7 元/百万 ≈ 0.05~0.16 元），**合计约 0.12~0.23 元/小时**——若此折算适用，3.1 比 paraformer-v2 还便宜，但 token 计费用量随输出文本量波动，需实测对账（官方未在 ASR 小节单独复述 25 token/s 规则，此换算标注为推算）。

### ② Fun-ASR（通义"百聆"，2025-09 云栖大会发布）

| 模型 ID | 模式 | 计费 | 说话人分离 | 时长上限 |
|---|---|---|---|---|
| `fun-asr`（= 快照 2025-11-07） | 非实时（HTTP 异步） | 0.00022 元/秒 = **0.792 元/小时**，RPM 600 | ✅ | 12 小时 / 2GB |
| `fun-asr-mtl` | 非实时（多语种） | 同上 | — | 同上 |
| `fun-asr-realtime` | 实时（WebSocket） | 0.00033 元/秒 = 1.188 元/小时 | ❌ | 不限 |
| `fun-asr-flash-2026-06-15` | 非实时同步（≤5 分钟） | 0.00022 元/秒 | ❌ | 5 分钟 |
| `fun-asr-flash-8k-realtime` | 实时 8k 电话信道 | 0.00022 元/秒 | ❌ | 不限 |

- 发布时间线：通义百聆语音模型家族（Fun-ASR + Fun-CosyVoice）于 **2025-09-24 云栖大会**发布（[雪球快讯](https://xueqiu.com/8181523997/354219536)，转引自官方发布）；技术报告 arXiv:2509.12508（2025-09）。API 快照：2025-08-25、2025-11-07；官方模型页注明当前 `fun-asr` 主版本 = 2025-11-07 快照（[fun-asr 模型页](https://help.aliyun.com/zh/model-studio/fun-asr)，查询日期 2026-10-10）。
- 语言：`fun-asr` 主版本 30 个语种 + 中文 7 大方言体系 + 20+ 地区口音（模型页原文）。
- 开源对应物：Fun-ASR-Nano-2512（0.8B，2025-12 开源，[ModelScope 模型卡](https://www.modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512)）；**API 上的 `fun-asr` 对应闭源 7.7B 完整版**。

### ③ Qwen-ASR（qwen3-asr-flash 系，2025-09-08 发布）

| 模型 ID | 模式 | 计费 | 说话人分离 | 情感识别 | 时长上限 |
|---|---|---|---|---|---|
| `qwen3-asr-flash-filetrans` | 非实时（HTTP 异步） | 0.00022 元/秒 = **0.792 元/小时** | ❌ | ✅（固定开启，7 类情绪） | 12 小时 / 2GB |
| `qwen3-asr-flash-realtime` | 实时（WebSocket） | 0.00033 元/秒 = 1.188 元/小时 | ❌ | ✅ | 不限 |
| `qwen3-asr-flash` | 非实时同步 / OpenAI 兼容 | 0.00022 元/秒 | ❌ | ✅ | 5 分钟 / 10MB |

- 语言：30 个语种（中英日韩德法西葡阿意俄印印尼泰土乌越捷丹菲芬冰马挪波瑞典）+ 中文方言（四川话、闽南语、吴语、粤语）。
- 特色：支持任意格式文本上下文注入（专名/背景定制）、歌声识别、自动语种判断、静音/噪声过滤。发布宣传称在方言、多语种、关键信息、歌词识别上错误率低于 GPT-4o-Transcribe、Gemini-2.5-Pro、**Paraformer-v1**、豆包 Doubao-ASR（[智东西报道](https://www.sohu.com/a/970627093_122476352)，转引自阿里官方发布稿，2025-09）。
- 开源对应物：Qwen3-ASR-0.6B/1.7B + Qwen3-ForcedAligner-0.6B，**2026-01-29 开源**（[QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)），支持 30 语种 + 22 种中文方言、流式/离线统一、长音频转写、字级时间戳。**本仓库 `models/Qwen3-ASR-0.6B` 已下载该模型，`ASR_BACKEND=qwen` 可本地使用。**

### ④ Paraformer（上一代，官方建议迁移）

| 模型 ID | 模式 | 计费 | 说话人分离 | 语言 |
|---|---|---|---|---|
| `paraformer-v2` | 非实时（HTTP） | **0.00008 元/秒 = 0.288 元/小时**；免费 **10 小时/月（每月 1 日发放，当月有效）** | ✅ | 中文（含 19 种方言口音）+ 英/日/韩/德/法/俄 |
| `paraformer-8k-v2` | 非实时 8k | 同上 | — | 中文普通话 |
| `paraformer-realtime-v2` / `-8k-v2` | 实时 | 0.00024 元/秒 = 0.864 元/小时；10 小时/月免费 | — | 同上 |
| `paraformer-v1` / `paraformer-mtl-v1` | 非实时 | 定价页仍有条目 | — | 更老，选型页已不再列出 |

- paraformer-v2 即本项目本地 FunASR Paraformer-large 同源的 2024 代云端版本。官方定位原文见上方"一句话结论"。
- 说话人分离限制：全部支持 diarization 的模型（qwen-audio-3.x-filetrans、fun-asr、paraformer-v2）**官方均建议开启说话人分离时音频 ≤ 2 小时**，否则可能失败或超时（[非实时语音识别用户指南](https://help.aliyun.com/zh/model-studio/non-realtime-speech-recognition-user-guide)）。

### ⑤ Gummy / 语音翻译系（非纯 ASR，仅备注）

`gummy-translation-realtime`、`qwen3-livetranslate-flash` 为语音翻译/同传模型，按 token 计费，不适用于纯转写场景。

---

## 2. 准确率对比（官方/一手 benchmark 数字）

### Fun-ASR 官方模型卡：行业测试集 WER（%），含 Paraformer v2 直接对比

来源：[ModelScope Fun-ASR-Nano-2512 模型卡](https://www.modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512)（查询日期 2026-10-10；表中 Fun-ASR = 闭源 7.7B，即百炼 API `fun-asr` 对应版本）

| 测试集 | Paraformer v2 (0.2B) | Fun-ASR-Nano (0.8B) | Fun-ASR (7.7B) | Seed-ASR（豆包） |
|---|---|---|---|---|
| 近场 Nearfield | 8.11 | 7.79 | **6.31** | 7.20 |
| 远场 Farfield | 9.55 | 5.79 | **4.34** | 4.59 |
| 复杂背景 Complex Background | 15.19 | 14.59 | **11.45** | 12.90 |
| 英语 English General | 19.48 | 15.28 | **13.73** | 15.65 |
| 方言 Dialect | 41.16 | 28.18 | **15.21** | 29.45 |
| 口音 Accent | 17.80 | 12.90 | **10.31** | 10.23 |
| **平均 Average** | **23.49** | **16.72** | **12.70** | 15.95 |

→ **官方行业集上，paraformer-v2 平均 WER 约为 fun-asr 的 1.85 倍（23.49% vs 12.70%）**；方言/口音差距最大。注意这是阿里自有行业测试集，口径对自家新模型有利，但量级差距（非零点几个点）有参考价值。

### Qwen3-ASR 官方 benchmark（开源 1.7B，[GitHub QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)，查询日期 2026-10-10）

中文公开集（WER%）：WenetSpeech net **4.97** / meeting **5.88**、AISHELL-2 **2.71**、SpeechIO **2.88**、Fleurs-zh **2.41**——全面优于同表的 GPT-4o-Transcribe、Gemini-2.5-Pro、Doubao-ASR、Whisper-large-v3。内部集：对话普通话 6.54、绕口令 2.44、极端噪声 16.17、中文方言对话 15.94。语种识别准确率平均 97.9%（Whisper-large-v3 为 94.1%）——**对中英混说（code-switching）场景，Qwen3-ASR 与 Fun-ASR 官方均明确宣传支持自由切换/混合识别**；paraformer 系官方语言列表为单语种+方言枚举，未宣传混说能力。

### 百炼官方文档未给出 qwen3-asr-flash API 版的公开 CER/WER 表；以上为同家族开源模型/模型卡口径，API 版能力 ≥ 开源版（阿里惯例），但严格说不完全等同，建议实测。

---

## 3. 选型结论（2026-10-10，阿里百炼范围）

需求画像：**中文为主、中英夹杂、约 60 小时/月、单文件 1~4 小时、最好带说话人分离**。

| 方案 | 月成本（60h） | 说话人分离 | 准确率（官方口径） | 备注 |
|---|---|---|---|---|
| **`fun-asr`（首选）** | **~47.5 元** | ✅（≤2h 建议，超长需切片） | 行业集平均 WER 12.70% | 30 语种、中英混说、标点/ITN 优化、RPM 600、12h 上限 |
| `qwen3-asr-flash-filetrans` | ~47.5 元 | ❌ | 同家族 1.7B 中文公开集 WER 2.4~5.9% | 需要情感标签/不要说话人分离时选它 |
| `qwen-audio-3.1-asr-flash-filetrans` | 按 token 计费，粗算 ~7~14 元（推算，未实测） | ✅ | 官方当前首推 | token 计费需实测对账；上下文/热词能力最全 |
| `paraformer-v2`（省钱兜底） | ~17.3 元（另有 10h/月免费，实际更低） | ✅ | 行业集平均 WER 23.49% | 官方已标记"较早一代"，仅作成本敏感兜底 |
| 本地 `Qwen3-ASR-0.6B`（仓库已有） | 0 元 | ❌（无 diarization） | 中文集 WER 2.9~6.9%（官方） | 已在本仓库 models/ 下，`ASR_BACKEND=qwen` |

**结论：**

1. 用户质疑成立——paraformer-v2 确为 2024 代模型，百炼官方 2026 年的选型页已明确建议迁往 Fun-ASR / Qwen-ASR。继续用它的唯一理由是价格（0.288 元/小时 + 每月 10h 免费）。
2. 对该需求画像，**`fun-asr` 是最均衡的选择**：0.792 元/小时（60h ≈ 47.5 元/月）、官方行业集准确率明显优于 paraformer-v2（尤其口音/方言/中英混说）、支持说话人分离、单文件 12 小时上限覆盖 1~4h 音频。月成本与 qwen3-asr-flash-filetrans 相同，但多说话人分离。
3. 说话人分离 + >2h 音频是全线共同短板（官方建议 ≤2h），若 4 小时访谈必须分人，需要在管道层按 VAD/静音切片后分片提交再拼接——本项目已有段落聚合与时间戳处理，改造成本可控。
4. `qwen-audio-3.1-asr-flash-filetrans` 按 token 计费且官方首推，粗算可能是最便宜的云端选项，但 token 折算对输出文本量敏感，**建议先用 100 万 token 免费额度跑几条真实音频对账后再定**。
5. 别忘了非云端选项：仓库里已有 Qwen3-ASR-0.6B（2026-01 开源，中文公开集 WER 与商用 API 同档），零边际成本；其短板是无说话人分离、长音频需自行切片、吞吐受本地 GPU 限制。
