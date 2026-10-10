<div align="center">

# AI Video Transcriber (RE)

English | [中文](README_ZH.md)

An open-source AI tool to transcribe, losslessly optimize, and summarize videos/podcasts.
Paste a URL from YouTube, Bilibili, TikTok, Apple Podcasts, and 30+ platforms, or upload a local audio/video file.

A re-edited fork (RE = Re-Edit) of [wendy7756/AI-Video-Transcriber](https://github.com/wendy7756/AI-Video-Transcriber).

![Interface](en_video.png)

</div>

## What's different from upstream

This fork replaces the original ASR and optimization pipeline with a Chinese-first, high-fidelity stack:

- **ASR**: Faster-Whisper → **FunASR Paraformer-large** (default, DAMO Academy Chinese flagship with built-in punctuation) or **SenseVoice-Small** (multilingual + emotion). Switchable via env var, no code change.
- **Lossless optimization engine** (`backend/lossless_optimizer.py`): LLM proofreads segment-by-segment (add punctuation, fix typos, **never delete or rewrite**). 32-way concurrent, exponential 429 backoff, per-segment id reconciliation guarantees zero text loss.
- **Paragraph aggregation**: merges sentence-level timestamps into natural paragraphs (`FUNASR_PARAGRAPH_PAUSE` / `FUNASR_PARAGRAPH_CHARS`).
- **Real-time optimization progress** (55→68), fixing the "stuck/hung" false impression.
- **Structured logging**: `logs/app.log` daily rotation (30 days) + per-task stage timing timeline.
- **Frontend timing display**: total + per-stage elapsed time after completion.

## Quick Start

### Prerequisites

- Python 3.10+
- FFmpeg (in system PATH)
- NVIDIA GPU + CUDA (recommended; CPU works but slower)
- Any OpenAI-compatible API key (for optimization + summary + translation)

### Install & Run

```bash
git clone https://github.com/jonsins202/AI-Video-Transcriber_RE.git
cd AI-Video-Transcriber_RE

python -m venv venv
venv\Scripts\activate          # Windows PowerShell
# source venv/Scripts/activate # Git Bash

pip install -r requirements.txt

python start.py --prod
```

- Visit `http://localhost:9000`
- Enter your API key in the UI **AI Settings** panel, or set `OPENAI_API_KEY` / `OPENAI_BASE_URL` env vars.
- The Paraformer model (~1.1GB) auto-downloads from ModelScope on first transcription (China-friendly, no proxy needed).
- `--prod` disables hot-reload for stable SSE on long tasks. **Restart the service after code changes.**

## Key Configuration

Configure via env vars or `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `ASR_BACKEND` | `funasr` | ASR backend (`funasr` / `qwen`) |
| `FUNASR_MODEL` | `paraformer` | FunASR model (`paraformer` / `sensevoice`) |
| `FUNASR_PARAGRAPH_PAUSE` | `1.5` | Paragraph aggregation pause threshold (seconds) |
| `FUNASR_PARAGRAPH_CHARS` | `300` | Paragraph aggregation max char count |
| `FUNASR_HOTWORD` | (empty) | Hot words (space-separated, mainly paraformer) |
| `LOSSLESS_CONCURRENCY` | `32` | Lossless optimization concurrency |
| `SUMMARY_CONCURRENCY` | `6` | Summary concurrency |
| `ENABLE_TIMESTAMPS` | `False` | ForcedAligner word-level timestamps toggle |
| `OPENAI_API_KEY` | (UI) | API key |
| `OPENAI_BASE_URL` | (UI) | API base URL |

## Performance (measured)

RTX 5060 Ti 16GB + CUDA, Paraformer-large:

| Stage | Time | Notes |
|-------|------|-------|
| ASR | ~2 min / 4h audio | Paraformer CUDA |
| Lossless optimization | ~10-12 min / 4h audio | 32 concurrent, depends on LLM |
| Total | ~15-20 min / 4h audio | incl. optimization + summary |

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| CUDA out of memory | Lower `ASR_BATCH_SIZE`; CPU also works (slower) |
| yt-dlp rate-limited (412) | Add `cookies.txt`, or set `YT_DLP_COOKIES_FROM_BROWSER=chrome` |
| Optimization slow / frequent 429 | Lower `LOSSLESS_CONCURRENCY` to 4-8 |
| Code changes not taking effect | `--prod` no hot-reload; restart the service |
| Summary unavailable | Fill API key in UI, or set `OPENAI_API_KEY` |
| Garbled / wrong ASR | Use Paraformer (default); add `FUNASR_HOTWORD` |

## Acknowledgments

- [wendy7756/AI-Video-Transcriber](https://github.com/wendy7756/AI-Video-Transcriber) -- upstream project
- [FunASR](https://github.com/modelscope/FunASR) -- DAMO Academy speech recognition
- [yt-dlp](https://github.com/yt-dlp/yt-dlp), [FastAPI](https://fastapi.tiangolo.com/), [OpenAI](https://openai.com/)

---

For full documentation, see [README_ZH.md (中文)](README_ZH.md). For issues, please [submit an Issue](https://github.com/jonsins202/AI-Video-Transcriber_RE/issues).
