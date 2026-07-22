import os
import re
import logging
import asyncio
from dataclasses import dataclass, field
from typing import Optional, List

logger = logging.getLogger(__name__)


# Qwen3-ASR 语言名称 → ISO 代码映射（_QwenBackend 用 + 兼容外部 import）
LANG_NAME_TO_ISO = {
    "chinese": "zh", "english": "en", "japanese": "ja", "korean": "ko",
    "spanish": "es", "french": "fr", "german": "de", "italian": "it",
    "portuguese": "pt", "russian": "ru", "arabic": "ar", "hindi": "hi",
    "thai": "th", "vietnamese": "vi", "turkish": "tr", "polish": "pl",
    "dutch": "nl", "swedish": "sv", "danish": "da", "norwegian": "no",
    "cantonese": "zh-yue", "shanghainese": "zh-wuu", "sichuanese": "zh-sic",
}


# ----------------------------------------------------------------------
# 统一数据结构（两套后端都产出 TranscriptResult，门面统一拼 markdown）
# ----------------------------------------------------------------------

@dataclass
class Segment:
    start_fmt: str   # "MM:SS" 或 "HH:MM:SS"
    end_fmt: str
    text: str


@dataclass
class TranscriptResult:
    text: str
    detected_language: Optional[str] = None
    language_probability: Optional[float] = None
    segments: List[Segment] = field(default_factory=list)


def _format_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _format_time_ms(ms) -> str:
    return _format_time(float(ms) / 1000.0)


def _make_backend(model_path, forced_aligner_path, device):
    name = os.getenv("ASR_BACKEND", "funasr").strip().lower()
    if name == "qwen":
        return _QwenBackend(model_path, forced_aligner_path, device)
    return _FunASRBackend()


# ======================================================================
# 门面类（公开契约不变，main.py 零改动）
# ======================================================================

class Transcriber:
    """音频转录器门面。根据 ASR_BACKEND 委托给 FunASR（默认）或 Qwen3-ASR。"""

    def __init__(self, model_path: Optional[str] = None,
                 forced_aligner_path: Optional[str] = None,
                 device: str = "cuda:0"):
        self.last_detected_language: Optional[str] = None
        self._backend = _make_backend(model_path, forced_aligner_path, device)
        logger.info(f"ASR 后端: {self._backend.__class__.__name__}")

    async def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        """转录音频，返回 Markdown 字符串。"""
        import time
        try:
            is_url = audio_path.startswith(("http://", "https://", "ftp://"))
            if not is_url and not os.path.exists(audio_path):
                raise Exception(f"音频文件不存在: {audio_path}")

            # 探测音频时长（仅读 header），用于 RTF 日志
            duration = 0.0
            if not is_url:
                try:
                    import soundfile as sf
                    info = sf.info(audio_path)
                    if info.samplerate:
                        duration = info.frames / float(info.samplerate)
                except Exception:
                    pass

            logger.info(f"开始转录音频: {audio_path}（{duration:.1f}s）")
            t_start = time.perf_counter()
            result: TranscriptResult = await asyncio.to_thread(
                self._backend.transcribe_sync, audio_path, language
            )
            t_elapsed = time.perf_counter() - t_start

            self.last_detected_language = result.detected_language
            if duration > 0 and t_elapsed > 0:
                rtf = t_elapsed / duration
                logger.info(
                    f"转录完成: 耗时 {t_elapsed:.1f}s（RTF={rtf:.3f}，{1/rtf:.1f}x 实时），"
                    f"语言: {result.detected_language}"
                )
            else:
                logger.info(f"转录完成: 耗时 {t_elapsed:.1f}s，语言: {result.detected_language}")

            return self._build_markdown(result)

        except Exception as e:
            logger.error(f"转录失败: {str(e)}")
            raise Exception(f"转录失败: {str(e)}")

    def get_detected_language(self, transcript_text: Optional[str] = None) -> Optional[str]:
        if self.last_detected_language:
            return self.last_detected_language
        if transcript_text and "**Detected Language:**" in transcript_text:
            for line in transcript_text.split("\n"):
                if "**Detected Language:**" in line:
                    lang = line.split("Detected Language:", 1)[-1].strip().lstrip("*").strip()
                    return lang if lang and lang != "—" and lang != "unknown" else None
        return None

    def get_supported_languages(self) -> list:
        return list(set(LANG_NAME_TO_ISO.values()))

    def _build_markdown(self, result: TranscriptResult) -> str:
        """组装为与原实现兼容的 Markdown（main.py 依赖 Detected Language 行）。"""
        lines = [
            "# Video Transcription", "",
            f"**Detected Language:** {result.detected_language or 'unknown'}",
            f"**Language Probability:** "
            f"{result.language_probability if result.language_probability is not None else '—'}",
            "", "## Transcription Content", "",
        ]
        if result.segments:
            for seg in result.segments:
                lines.append(f"**[{seg.start_fmt} - {seg.end_fmt}]**")
                lines.append("")
                lines.append(seg.text)
                lines.append("")
        else:
            lines.append(result.text.strip())
            lines.append("")
        return "\n".join(lines)


# ======================================================================
# FunASR 后端（默认）：SenseVoice-Small / Paraformer-large
# ======================================================================

# SenseVoice 前缀标签里的语言码 → ISO
_SENSEVOICE_LANG_TAG = {
    "zh": "zh", "en": "en", "ja": "ja", "ko": "ko",
    "yue": "zh-yue", "auto": None,
}


class _FunASRBackend:
    """FunASR 后端。模型由 FUNASR_MODEL 选择：sensevoice（默认）/ paraformer / 自定义 repo id。"""

    def __init__(self):
        self.model = None
        self._kind = None

    def _load(self):
        if self.model is not None:
            return
        from funasr import AutoModel

        kind = (os.getenv("FUNASR_MODEL", "sensevoice").strip().lower() or "sensevoice")
        device = os.getenv("FUNASR_DEVICE", "cuda").strip() or "cuda"
        vad = os.getenv("FUNASR_VAD_MODEL", "fsmn-vad").strip() or None

        kwargs = dict(device=device, disable_update=True)
        if vad:
            kwargs["vad_model"] = vad

        if kind in ("sensevoice", "sensevoice-small"):
            kwargs["model"] = "iic/SenseVoiceSmall"
            logger.info(f"加载 FunASR SenseVoice-Small（device={device}, vad={vad}）")
        elif kind in ("paraformer", "paraformer-zh"):
            kwargs["model"] = "paraformer-zh"
            kwargs["punc_model"] = "ct-punc"   # Paraformer 需要标点模型补标点
            logger.info(f"加载 FunASR Paraformer-large（device={device}, vad={vad}, punc=ct-punc）")
        else:
            kwargs["model"] = kind   # 允许直接填 ModelScope repo id
            logger.info(f"加载 FunASR 模型: {kind}（device={device}, vad={vad}）")

        self.model = AutoModel(**kwargs)
        self._kind = kind

    def transcribe_sync(self, audio_path: str, language: Optional[str]) -> TranscriptResult:
        self._load()

        try:
            batch_size_s = int(os.getenv("FUNASR_BATCH_SIZE_S", "300"))
        except (ValueError, TypeError):
            batch_size_s = 300
        gen_kwargs = dict(input=audio_path, batch_size_s=batch_size_s)
        lang_arg = (language or os.getenv("FUNASR_LANG", "")).strip() or None
        if lang_arg:
            gen_kwargs["language"] = lang_arg
        hotword = os.getenv("FUNASR_HOTWORD", "").strip()
        if hotword:
            gen_kwargs["hotword"] = hotword
        # Paraformer 需要 sentence_timestamp=True 才会聚合出 sentence_info（句级时间戳）
        if self._kind in ("paraformer", "paraformer-zh"):
            gen_kwargs["sentence_timestamp"] = True

        res = self.model.generate(**gen_kwargs)
        if not res:
            raise Exception("FunASR 返回空结果")
        r = res[0]

        raw_text = (r.get("text") or "").strip()
        # 清洗 SenseVoice 前缀标签：<|zh|><|HAPPY|><|BGM|><|woitn|> 等
        text = re.sub(r"<\|[^|]*\|>", "", raw_text).strip()
        lang = self._parse_language(raw_text, r)
        # Paraformer 不返回语言字段/标签：paraformer-zh 是中文模型，兜底 zh
        if lang is None and self._kind in ("paraformer", "paraformer-zh"):
            lang = lang_arg or "zh"
        elif lang is None and lang_arg:
            lang = lang_arg

        # Paraformer 返回 sentence_info（带毫秒时间戳）；SenseVoice 无（纯文本）
        segments: List[Segment] = []
        si = r.get("sentence_info")
        if si:
            for s in si:
                seg_text = (s.get("sentence") or s.get("text") or "").strip()
                if not seg_text:
                    continue
                segments.append(Segment(
                    start_fmt=_format_time_ms(s.get("start", 0)),
                    end_fmt=_format_time_ms(s.get("end", 0)),
                    text=seg_text,
                ))

        if not text and segments:
            text = " ".join(s.text for s in segments)

        return TranscriptResult(text=text, detected_language=lang, segments=segments)

    @staticmethod
    def _parse_language(raw_text: str, result) -> Optional[str]:
        # SenseVoice：<|zh|> 前缀标签
        m = re.search(r"<\|([a-zA-Z]{2,5})\|>", raw_text or "")
        if m:
            tag = m.group(1).lower()
            if tag in _SENSEVOICE_LANG_TAG:
                return _SENSEVOICE_LANG_TAG[tag]   # auto → None（未知），已知语种 → ISO
            return tag   # 未知语种原样返回（如 fr/de）
        # Paraformer：result["language"]
        lang = result.get("language") if hasattr(result, "get") else None
        if lang:
            return str(lang).lower()[:5]
        return None


# ======================================================================
# Qwen3-ASR 后端（备选）：搬迁自原 transcriber.py，懒导入
# ======================================================================

_ENABLE_TIMESTAMPS = False   # 关：长音频按 1200s/块切分，无 CPU 对齐瓶颈
_ASR_BATCH_SIZE = 2


class _QwenBackend:
    """Qwen3-ASR 后端（备选）。懒导入 torch/qwen_asr。"""

    def __init__(self, model_path=None, forced_aligner_path=None, device="cuda:0"):
        self.device = device
        self.model = None
        self._timestamps_available = False

        if model_path:
            self.model_path = model_path
        else:
            self.model_path = os.getenv(
                "QWEN3_ASR_MODEL_PATH",
                os.path.join(os.path.dirname(__file__), "..", "models", "Qwen3-ASR-0.6B"),
            )
        self.model_path = os.path.normpath(os.path.abspath(self.model_path))

        if forced_aligner_path:
            self.forced_aligner_path = forced_aligner_path
        else:
            env_fa = os.getenv("QWEN3_FORCED_ALIGNER_PATH", "")
            if env_fa:
                self.forced_aligner_path = os.path.normpath(os.path.abspath(env_fa))
            else:
                default_fa = os.path.join(
                    os.path.dirname(__file__), "..", "models", "Qwen3-ForcedAligner-0.6B"
                )
                default_fa = os.path.normpath(os.path.abspath(default_fa))
                self.forced_aligner_path = default_fa if os.path.isdir(default_fa) else None

    def _load_model(self):
        if self.model is not None:
            return
        import torch
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        logger.info(f"正在加载 Qwen3-ASR 模型: {self.model_path}")
        if not os.path.isdir(self.model_path):
            raise Exception(f"模型路径不存在: {self.model_path}")

        load_kwargs = {
            "dtype": torch.bfloat16,
            "device_map": self.device,
            "max_inference_batch_size": _ASR_BATCH_SIZE,
            # qwen_asr 默认 max_new_tokens=512 会截断长音频转录，必须放大
            "max_new_tokens": 8192,
        }
        if _ENABLE_TIMESTAMPS and self.forced_aligner_path and os.path.isdir(self.forced_aligner_path):
            logger.info(f"同时加载 ForcedAligner（CPU）: {self.forced_aligner_path}")
            load_kwargs["forced_aligner"] = self.forced_aligner_path
            load_kwargs["forced_aligner_kwargs"] = {"dtype": torch.float32, "device_map": "cpu"}
            self._timestamps_available = True
        else:
            self._timestamps_available = False
            logger.info(f"时间戳关闭（ENABLE_TIMESTAMPS={_ENABLE_TIMESTAMPS}），长音频按 1200s/块切分")

        from qwen_asr import Qwen3ASRModel
        try:
            self.model = Qwen3ASRModel.from_pretrained(self.model_path, **load_kwargs)
            logger.info("Qwen3-ASR 模型加载完成")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise

    def transcribe_sync(self, audio_path: str, language: Optional[str]) -> TranscriptResult:
        self._load_model()
        qwen_lang = self._iso_to_qwen_lang(language) if language else None

        results = self.model.transcribe(
            audio=audio_path,
            language=qwen_lang,
            return_time_stamps=self._timestamps_available,
        )
        if not results:
            raise Exception("Qwen3 转录结果为空")
        result = results[0]
        detected = result.language or "unknown"
        iso = self._qwen_lang_to_iso(detected)

        segments: List[Segment] = []
        if self._timestamps_available and getattr(result, "time_stamps", None) is not None:
            for seg in self._group_timestamps(result.time_stamps):
                segments.append(Segment(
                    start_fmt=seg["start"], end_fmt=seg["end"], text=seg["text"]
                ))

        text = (result.text or "").strip()
        return TranscriptResult(text=text, detected_language=iso, segments=segments)

    # ---- Qwen 语言工具（搬迁自原文件）----
    @staticmethod
    def _iso_to_qwen_lang(iso_code: Optional[str]) -> Optional[str]:
        if not iso_code:
            return None
        iso_lower = iso_code.lower().strip()
        for name, iso in LANG_NAME_TO_ISO.items():
            if iso == iso_lower:
                return name
        return iso_code

    @staticmethod
    def _qwen_lang_to_iso(qwen_lang: Optional[str]) -> Optional[str]:
        if not qwen_lang:
            return None
        first_lang = qwen_lang.split(",")[0].strip().lower()
        return LANG_NAME_TO_ISO.get(first_lang, first_lang)

    _SPLIT_PUNCT = set("。，！？；：、…—\"\"''（）《》【】,.;:!?\"'")

    def _group_timestamps(self, time_stamps) -> list:
        segments = []
        buf_text, buf_start, buf_end = [], None, None
        for seg in time_stamps:
            text = seg.text if hasattr(seg, "text") else str(seg)
            t_start = seg.start_time if hasattr(seg, "start_time") else 0.0
            t_end = seg.end_time if hasattr(seg, "end_time") else 0.0
            if text in self._SPLIT_PUNCT:
                if buf_text:
                    segments.append({
                        "start": _format_time(buf_start or 0),
                        "end": _format_time(buf_end or 0),
                        "text": "".join(buf_text).strip(),
                    })
                    buf_text, buf_start, buf_end = [], None, None
                continue
            if buf_start is None:
                buf_start = t_start
            buf_end = t_end
            buf_text.append(text)
        if buf_text:
            segments.append({
                "start": _format_time(buf_start or 0),
                "end": _format_time(buf_end or 0),
                "text": "".join(buf_text).strip(),
            })
        return segments
