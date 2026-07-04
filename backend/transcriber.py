import os
import torch
import logging
from typing import Optional, List

import soundfile as sf
from qwen_asr import Qwen3ASRModel

logger = logging.getLogger(__name__)

# 是否启用 ForcedAligner 字级时间戳（关闭后转录文本不含 [00:00] 标记，但长音频
# 切分从 180s/块 → 1200s/块，块数大减 + 无 CPU 对齐瓶颈，转录大幅提速）
ENABLE_TIMESTAMPS = False

# ASR batch 推理大小。注：块大小随 ENABLE_TIMESTAMPS 变化 → 激活显存差异巨大：
#   - ENABLE_TIMESTAMPS=False：1200s/块，8GB 建议 2（OOM 则降到 1）
#   - ENABLE_TIMESTAMPS=True：  180s/块，8GB 建议 4-6
ASR_BATCH_SIZE = 2

# Qwen3-ASR 语言名称 → ISO 代码映射
LANG_NAME_TO_ISO = {
    "chinese": "zh",
    "english": "en",
    "japanese": "ja",
    "korean": "ko",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "portuguese": "pt",
    "russian": "ru",
    "arabic": "ar",
    "hindi": "hi",
    "thai": "th",
    "vietnamese": "vi",
    "turkish": "tr",
    "polish": "pl",
    "dutch": "nl",
    "swedish": "sv",
    "danish": "da",
    "norwegian": "no",
    "cantonese": "zh-yue",
    "shanghainese": "zh-wuu",
    "sichuanese": "zh-sic",
}


class Transcriber:
    """音频转录器，使用 Qwen3-ASR 进行语音转文字"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        forced_aligner_path: Optional[str] = None,
        device: str = "cuda:0",
    ):
        """
        初始化转录器

        Args:
            model_path: Qwen3-ASR 模型路径，默认从环境变量 QWEN3_ASR_MODEL_PATH 读取，
                        否则为项目目录下的 models/Qwen3-ASR-0.6B
            forced_aligner_path: Qwen3-ForcedAligner 模型路径（可选），用于字级时间戳
            device: 推理设备 (cuda:0 / cpu)
        """
        if model_path:
            self.model_path = model_path
        else:
            self.model_path = os.getenv(
                "QWEN3_ASR_MODEL_PATH",
                os.path.join(os.path.dirname(__file__), "..", "models", "Qwen3-ASR-0.6B"),
            )
        # 规范化路径
        self.model_path = os.path.normpath(os.path.abspath(self.model_path))

        if forced_aligner_path:
            self.forced_aligner_path = forced_aligner_path
        else:
            env_fa = os.getenv("QWEN3_FORCED_ALIGNER_PATH", "")
            if env_fa:
                self.forced_aligner_path = os.path.normpath(os.path.abspath(env_fa))
            else:
                # 默认路径，但不强制要求存在
                default_fa = os.path.join(
                    os.path.dirname(__file__), "..", "models", "Qwen3-ForcedAligner-0.6B"
                )
                default_fa = os.path.normpath(os.path.abspath(default_fa))
                self.forced_aligner_path = default_fa if os.path.isdir(default_fa) else None

        self.device = device
        self.model = None
        self.last_detected_language = None
        self._timestamps_available = False

    def _load_model(self):
        """延迟加载模型（GPU bf16）"""
        if self.model is not None:
            return

        # 减少 PyTorch CUDA 内存碎片化（长音频推理关键）
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        logger.info(f"正在加载 Qwen3-ASR 模型: {self.model_path}")
        if not os.path.isdir(self.model_path):
            raise Exception(f"模型路径不存在: {self.model_path}")

        load_kwargs = {
            "dtype": torch.bfloat16,
            "device_map": self.device,
            # qwen_asr 默认 max_inference_batch_size=32，8GB 显存必 OOM，限制为 ASR_BATCH_SIZE
            "max_inference_batch_size": ASR_BATCH_SIZE,
        }

        # ForcedAligner 仅在 ENABLE_TIMESTAMPS=True 时加载（字级时间戳）
        # 关闭时：省 ~1.2GB 内存 + 加载时间，且 qwen_asr 内部切分从 180s/块 → 1200s/块
        if ENABLE_TIMESTAMPS and self.forced_aligner_path and os.path.isdir(self.forced_aligner_path):
            logger.info(f"同时加载 ForcedAligner（CPU）: {self.forced_aligner_path}")
            load_kwargs["forced_aligner"] = self.forced_aligner_path
            load_kwargs["forced_aligner_kwargs"] = {
                "dtype": torch.float32,
                "device_map": "cpu",
            }
            self._timestamps_available = True
        else:
            self._timestamps_available = False
            logger.info(
                f"时间戳关闭（ENABLE_TIMESTAMPS={ENABLE_TIMESTAMPS}），"
                f"长音频按 1200s/块切分，无 CPU 对齐瓶颈"
            )

        try:
            self.model = Qwen3ASRModel.from_pretrained(self.model_path, **load_kwargs)
            logger.info("Qwen3-ASR 模型加载完成")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise Exception(f"模型加载失败: {e}")

    async def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        """
        转录音频文件（长音频由 qwen_asr 内部自动切分 + batch 推理）

        Args:
            audio_path: 音频文件路径（支持 wav/mp3/m4a 等）
            language: ISO 语言代码（可选，如 zh/en；不指定则自动检测）

        Returns:
            转录文本（Markdown 格式）
        """
        import asyncio
        import time

        try:
            # URL 不需要检查本地文件是否存在
            is_url = audio_path.startswith(("http://", "https://", "ftp://"))
            if not is_url and not os.path.exists(audio_path):
                raise Exception(f"音频文件不存在: {audio_path}")

            self._load_model()

            # 将 ISO 代码映射为 Qwen3-ASR 的语言名称
            qwen_lang = self._iso_to_qwen_lang(language) if language else None

            # 探测音频时长（仅读 header，不加载音频）用于计算实时率 RTF
            duration = 0.0
            if not is_url:
                try:
                    info = sf.info(audio_path)
                    if info.samplerate:
                        duration = info.frames / float(info.samplerate)
                except Exception:
                    pass

            # 直接传整段音频：qwen_asr 内部按静音边界智能切分（ForcedAligner 模式 180s/块）
            # 并按 max_inference_batch_size 并行推理，无需外部手动分段
            logger.info(
                f"开始转录音频: {audio_path}"
                f"（{duration:.1f}s，batch_size={ASR_BATCH_SIZE}，"
                f"时间戳={'开' if self._timestamps_available else '关'}）"
            )
            t_start = time.perf_counter()

            def _do_transcribe():
                return self.model.transcribe(
                    audio=audio_path,
                    language=qwen_lang,
                    return_time_stamps=self._timestamps_available,
                )

            results: List = await asyncio.to_thread(_do_transcribe)

            t_elapsed = time.perf_counter() - t_start

            if not results:
                raise Exception("转录结果为空")

            result = results[0]
            detected_language = result.language or "unknown"
            self.last_detected_language = self._qwen_lang_to_iso(detected_language)

            # RTF < 1 表示快于实时（如 0.2 = 5x 实时速度）
            rtf = t_elapsed / duration if duration > 0 else 0.0
            if rtf > 0:
                logger.info(
                    f"转录完成: 耗时 {t_elapsed:.1f}s"
                    f"（RTF={rtf:.2f}，{1/rtf:.1f}x 实时），"
                    f"语言: {detected_language} → {self.last_detected_language}"
                )
            else:
                logger.info(
                    f"转录完成: 耗时 {t_elapsed:.1f}s，"
                    f"语言: {detected_language} → {self.last_detected_language}"
                )

            return self._build_markdown(result, self.last_detected_language or detected_language)

        except Exception as e:
            logger.error(f"转录失败: {str(e)}")
            raise Exception(f"转录失败: {str(e)}")


    # ------------------------------------------------------------------
    # Markdown 组装
    # ------------------------------------------------------------------

    def _build_markdown(self, result, detected_language: str) -> str:
        """将转录结果组装为 Markdown 格式。"""
        transcript_lines = [
            "# Video Transcription",
            "",
            f"**Detected Language:** {detected_language}",
            f"**Language Probability:** —",
            "",
            "## Transcription Content",
            "",
        ]

        if self._timestamps_available and result.time_stamps is not None:
            segments = self._group_timestamps(result.time_stamps)
            for seg in segments:
                transcript_lines.append(f"**[{seg['start']} - {seg['end']}]**")
                transcript_lines.append("")
                transcript_lines.append(seg["text"])
                transcript_lines.append("")
        else:
            transcript_lines.append(result.text.strip())
            transcript_lines.append("")

        return "\n".join(transcript_lines)

    def _group_timestamps(self, time_stamps) -> list:
        """
        将逐字/逐词的时间戳按标点合并为句子级片段。

        中英文标点作为分割点，连续的非标点字符合并为一个片段。
        片段使用第一个字的开始时间和最后一个字的结束时间。
        """
        SPLIT_PUNCT = set("。，！？；：、…—\"\"''（）《》【】,.;:!?\"'")

        segments = []
        buf_text = []
        buf_start = None
        buf_end = None

        for seg in time_stamps:
            text = seg.text if hasattr(seg, "text") else str(seg)
            t_start = seg.start_time if hasattr(seg, "start_time") else 0.0
            t_end = seg.end_time if hasattr(seg, "end_time") else 0.0

            if text in SPLIT_PUNCT:
                # 标点：先提交当前缓冲区
                if buf_text:
                    segments.append({
                        "start": self._format_time(buf_start),
                        "end": self._format_time(buf_end),
                        "text": "".join(buf_text).strip(),
                    })
                    buf_text = []
                    buf_start = None
                    buf_end = None
                # 跳过单独的标点片段
                continue
            else:
                if buf_start is None:
                    buf_start = t_start
                buf_end = t_end
                buf_text.append(text)

        # 提交剩余缓冲区
        if buf_text:
            segments.append({
                "start": self._format_time(buf_start),
                "end": self._format_time(buf_end),
                "text": "".join(buf_text).strip(),
            })

        return segments

    def _format_time(self, seconds: float) -> str:
        """将秒数转换为 MM:SS 或 HH:MM:SS 格式"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"

    def _iso_to_qwen_lang(self, iso_code: str) -> Optional[str]:
        """ISO 语言代码 → Qwen3-ASR 语言名称"""
        iso_lower = iso_code.lower().strip()
        for name, iso in LANG_NAME_TO_ISO.items():
            if iso == iso_lower:
                return name  # 保持小写，Qwen3-ASR 要求精确匹配
        # 如果找不到映射，直接返回原始输入（可能已经是 Qwen 格式）
        return iso_code

    def _qwen_lang_to_iso(self, qwen_lang: str) -> Optional[str]:
        """Qwen3-ASR 语言名称 → ISO 代码"""
        if not qwen_lang:
            return None
        # 处理多语言混合的情况，取第一个
        first_lang = qwen_lang.split(",")[0].strip().lower()
        return LANG_NAME_TO_ISO.get(first_lang, first_lang)

    def get_supported_languages(self) -> list:
        """获取支持的语言列表（ISO 代码）"""
        return list(set(LANG_NAME_TO_ISO.values()))

    def get_detected_language(self, transcript_text: Optional[str] = None) -> Optional[str]:
        """
        获取最近一次转录检测到的语言

        Args:
            transcript_text: 转录文本（可选，用于回退解析）

        Returns:
            ISO 语言代码
        """
        if self.last_detected_language:
            return self.last_detected_language

        if transcript_text and "**Detected Language:**" in transcript_text:
            lines = transcript_text.split("\n")
            for line in lines:
                if "**Detected Language:**" in line:
                    lang = line.split(":", 1)[-1].strip()
                    return lang if lang and lang != "—" else None

        return None
