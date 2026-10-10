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


def _speaker_label(spk_id: int) -> str:
    """spk 聚类编号 → 「说话人A」标签（0→A, 1→B, ...）。"""
    if 0 <= spk_id < 26:
        return f"说话人{chr(ord('A') + spk_id)}"
    return f"说话人{spk_id}"


_SPEAKER_LABEL_RE = re.compile(r"说话人([A-Za-z0-9]+)：")


def rename_speaker_labels(text: str, names: dict, previous: Optional[dict] = None) -> tuple:
    """把文稿中的「说话人A：」替换为「名字：」；返回 (新文本, 实际应用的映射)。

    names: {"A": "小辉博士"}，空名/缺项 → 保留（或还原为）「说话人A：」。
    previous: 上一次应用的映射。先把旧名还原成标签再统一替换，
    因此重复修改、清空回退都天然幂等。
    """
    cleaned = {}
    for key, value in (names or {}).items():
        key = (str(key) if key is not None else "").strip()
        value = (str(value) if value is not None else "").strip()
        value = value.replace("：", "").replace(":", "")[:30]
        if key and value:
            cleaned[key] = value

    reverted = text or ""
    for key, old_name in (previous or {}).items():
        if old_name:
            reverted = reverted.replace(f"{old_name}：", f"说话人{key}：")

    def _sub(match: "re.Match") -> str:
        label = match.group(1)
        return f"{cleaned[label]}：" if label in cleaned else match.group(0)

    return _SPEAKER_LABEL_RE.sub(_sub, reverted), cleaned


def _make_backend(model_path, forced_aligner_path, device):
    name = os.getenv("ASR_BACKEND", "funasr").strip().lower()
    if name == "qwen":
        return _QwenBackend(model_path, forced_aligner_path, device)
    if name in ("dashscope", "cloud"):
        return _DashScopeBackend()
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

    async def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        spk_num: Optional[int] = None,
    ) -> str:
        """转录音频，返回 Markdown 字符串。spk_num：固定说话人数（仅 FunASR+FUNASR_SPK 生效）。"""
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
                self._backend.transcribe_sync, audio_path, language, spk_num
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

    def detect_speaker_labels(self, transcript_text: str) -> list:
        """从文稿提取出现的说话人标签键（如 ["A", "B"]），供前端命名面板用。"""
        return sorted(set(_SPEAKER_LABEL_RE.findall(transcript_text or "")))

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
# DashScope 云端 ASR 后端：异步文件转写（qwen-audio-3.0-asr-flash-filetrans）
# 流程：本地音频 → 上传 OSS → 签名 URL 提交异步任务 → 轮询 → 下载结果 JSON
# ======================================================================


class _DashScopeBackend:
    """DashScope 云端 ASR 后端（懒导入 oss2）。本地音频先传 OSS 生成签名 URL 供
    云端拉取（DashScope 异步转写只接受公网 URL），转写完成后删除 OSS 对象。

    环境变量：
      DASHSCOPE_API_KEY / DASHSCOPE_WORKSPACE（必填；WorkspaceId 决定接入点域名）
      OSS_BUCKET / OSS_ENDPOINT / OSS_ACCESS_KEY_ID / OSS_ACCESS_KEY_SECRET（必填）
      DASHSCOPE_ASR_MODEL       默认 qwen-audio-3.0-asr-flash-filetrans
      DASHSCOPE_REGION          默认 cn-beijing
      DASHSCOPE_DIARIZATION     默认 1（开，云端自动聚类识别人数）；=0 关闭
      DASHSCOPE_LANGUAGE_HINTS  默认 zh,en（逗号分隔；空串则不传）
      DASHSCOPE_POLL_INTERVAL   轮询间隔秒，默认 3
      DASHSCOPE_POLL_TIMEOUT    最长等待秒，默认 3600
      DASHSCOPE_CHUNK_MAX_MIN   开说话人分离时单片上限分钟，默认 110（官方建议 ≤2h）
      DASHSCOPE_CHUNK_OVERLAP_S 切片重叠秒，默认 30（用于接缝去重 + 说话人跨片匹配）
    """

    _SIGN_URL_TTL = 6 * 3600  # 签名 URL 有效期，覆盖 12h 上限音频的排队+转写

    def __init__(self):
        self.api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
        workspace = os.getenv("DASHSCOPE_WORKSPACE", "").strip()
        region = os.getenv("DASHSCOPE_REGION", "cn-beijing").strip() or "cn-beijing"
        self.model = os.getenv(
            "DASHSCOPE_ASR_MODEL", "qwen-audio-3.0-asr-flash-filetrans"
        ).strip()
        missing = [k for k, v in (("DASHSCOPE_API_KEY", self.api_key),
                                  ("DASHSCOPE_WORKSPACE", workspace))
                   if not v]
        for k in ("OSS_BUCKET", "OSS_ENDPOINT",
                  "OSS_ACCESS_KEY_ID", "OSS_ACCESS_KEY_SECRET"):
            if not os.getenv(k, "").strip():
                missing.append(k)
        if missing:
            raise Exception(f"DashScope 后端缺少环境变量: {', '.join(missing)}")
        self.base_url = f"https://{workspace}.{region}.maas.aliyuncs.com/api/v1"

    # ---- OSS 上传 / 签名 URL / 清理 ----
    def _oss_bucket(self):
        import oss2
        auth = oss2.Auth(os.getenv("OSS_ACCESS_KEY_ID", "").strip(),
                         os.getenv("OSS_ACCESS_KEY_SECRET", "").strip())
        return oss2.Bucket(auth, os.getenv("OSS_ENDPOINT", "").strip(),
                           os.getenv("OSS_BUCKET", "").strip())

    def _stage_to_oss(self, audio_path: str):
        """上传本地音频到 OSS，返回 (bucket, object_key, 签名 URL)。"""
        import uuid
        ext = os.path.splitext(audio_path)[1] or ".wav"
        key = f"asr-staging/{uuid.uuid4().hex}{ext}"
        bucket = self._oss_bucket()
        size_mb = os.path.getsize(audio_path) / 1024 / 1024
        logger.info(f"上传音频到 OSS: {key}（{size_mb:.1f}MB）")
        bucket.put_object_from_file(key, audio_path)
        url = bucket.sign_url("GET", key, self._SIGN_URL_TTL, slash_safe=True)
        return bucket, key, url

    # ---- DashScope 异步任务 ----
    @staticmethod
    def _language_hints(language: Optional[str]) -> List[str]:
        hints = ([language] if language else None) or [
            h.strip() for h in os.getenv("DASHSCOPE_LANGUAGE_HINTS", "zh,en").split(",")
            if h.strip()
        ]
        return hints
    def _http_json(self, method: str, url: str, payload: Optional[dict] = None,
                   extra_headers: Optional[dict] = None) -> dict:
        import json as _json
        import urllib.request
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}
        headers.update(extra_headers or {})
        data = _json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return _json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:500]
            raise Exception(f"DashScope HTTP {e.code}: {body}")
        except urllib.error.URLError as e:
            raise Exception(f"DashScope 请求失败: {e}")

    @staticmethod
    def _diarization_enabled(spk_num: Optional[int]) -> bool:
        """说话人分离默认开启（云端自动聚类，无需预填人数）；DASHSCOPE_DIARIZATION=0 关闭。"""
        if spk_num:
            return True
        return os.getenv("DASHSCOPE_DIARIZATION", "1").strip().lower() not in ("0", "false")

    def _submit(self, file_url: str, language: Optional[str],
                spk_num: Optional[int]) -> str:
        params: dict = {
            # 关掉系统预置敏感词表，否则识别结果会被打码成 ***
            "special_word_filter": {"system_reserved_filter": False},
        }
        hints = self._language_hints(language)
        if hints:
            params["language_hints"] = hints
        if self._diarization_enabled(spk_num):
            params["diarization_enabled"] = True
            if spk_num:
                params["speaker_count"] = spk_num
        payload = {
            "model": self.model,
            "input": {"file_urls": [file_url]},
            "parameters": params,
        }
        logger.info(f"提交 DashScope 转写任务: model={self.model}, params={params}")
        resp = self._http_json(
            "POST", f"{self.base_url}/services/audio/asr/transcription",
            payload, {"X-DashScope-Async": "enable"})
        task_id = (resp.get("output") or {}).get("task_id")
        if not task_id:
            raise Exception(f"DashScope 任务提交失败: {resp}")
        return task_id

    def _wait_task(self, task_id: str) -> dict:
        import time
        try:
            interval = max(1, float(os.getenv("DASHSCOPE_POLL_INTERVAL", "3")))
        except (ValueError, TypeError):
            interval = 3.0
        try:
            deadline = time.monotonic() + float(os.getenv("DASHSCOPE_POLL_TIMEOUT", "3600"))
        except (ValueError, TypeError):
            deadline = time.monotonic() + 3600.0
        while True:
            resp = self._http_json("GET", f"{self.base_url}/tasks/{task_id}")
            output = resp.get("output") or {}
            status = output.get("task_status")
            if status == "SUCCEEDED":
                return output
            if status in ("FAILED", "CANCELED", "UNKNOWN"):
                raise Exception(f"DashScope 任务{status}: {output.get('code')} "
                                f"{output.get('message')}")
            if time.monotonic() > deadline:
                raise Exception(f"DashScope 任务超时（task_id={task_id}）")
            time.sleep(interval)

    def _fetch_result(self, output: dict) -> dict:
        import json as _json
        import urllib.request
        results = output.get("results") or []
        if not results:
            raise Exception(f"DashScope 结果为空: {output}")
        item = results[0]
        if item.get("subtask_status") != "SUCCEEDED":
            raise Exception(f"DashScope 子任务失败: {item.get('code')} "
                            f"{item.get('message')}")
        url = item.get("transcription_url")
        if not url:
            raise Exception(f"DashScope 结果缺少 transcription_url: {item}")
        # transcription_url 仅 24h 有效，立即下载
        with urllib.request.urlopen(url, timeout=120) as resp:
            return _json.loads(resp.read().decode("utf-8"))

    # ---- 结果解析（复用 FunASR 段落聚合规则）----
    @staticmethod
    def _collect_sentences(data: dict) -> tuple:
        """解析结果 JSON → (全文 text, si 句级列表, 语言投票)。si 适配 _paragraphs_from_sentence_info。"""
        transcripts = data.get("transcripts") or []
        text_parts: List[str] = []
        si: List[dict] = []
        lang_votes: dict = {}
        for tr in transcripts:
            t = (tr.get("text") or "").strip()
            if t:
                text_parts.append(t)
            for s in tr.get("sentences") or []:
                s_text = (s.get("text") or "").strip()
                if not s_text:
                    continue
                lang = s.get("language")
                if lang:
                    lang_votes[lang] = lang_votes.get(lang, 0) + len(s_text)
                si.append({
                    "sentence": s_text,
                    "start": s.get("begin_time", 0) or 0,
                    "end": s.get("end_time", 0) or 0,
                    "spk": s.get("speaker_id"),
                })
        return "".join(text_parts).strip(), si, lang_votes

    @staticmethod
    def _to_transcript_result(text: str, si: List[dict], lang_votes: dict,
                              language: Optional[str]) -> TranscriptResult:
        segments = _FunASRBackend._paragraphs_from_sentence_info(si)
        detected = max(lang_votes, key=lang_votes.get) if lang_votes else language
        return TranscriptResult(text=text, detected_language=detected, segments=segments)

    # ---- 长音频切片（说话人分离官方建议 ≤2h，超长则带重叠切片并发转写后合并）----
    @staticmethod
    def _chunk_max_s() -> float:
        try:
            return max(60.0, float(os.getenv("DASHSCOPE_CHUNK_MAX_MIN", "110")) * 60)
        except (ValueError, TypeError):
            return 110 * 60.0

    @staticmethod
    def _chunk_overlap_s() -> float:
        try:
            return max(5.0, float(os.getenv("DASHSCOPE_CHUNK_OVERLAP_S", "30")))
        except (ValueError, TypeError):
            return 30.0

    @staticmethod
    def _probe_duration(audio_path: str) -> float:
        try:
            import soundfile as sf
            info = sf.info(audio_path)
            if info.samplerate:
                return info.frames / float(info.samplerate)
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _merge_chunks(chunk_si: List[List[dict]], starts_s: List[float],
                      max_s: float, overlap_s: float) -> List[dict]:
        """合并各片句级结果到全局时间轴（接缝去重 + 说话人跨片重映射）。

        接缝切点在重叠区中点（带宽 b=overlap/2）：片 k>0 丢弃片内 begin < b 的句
        （前片保留区已覆盖），非末片丢弃 begin >= max_s-b 的句（后片给出完整版）。
        说话人映射：本片头部带（将被丢弃的部分）按说话人聚合文本，与前片保留区
        末尾 b 带宽内各全局说话人的文本算相似度，>=0.5 即匹配，否则分配新编号。
        """
        import difflib
        b = overlap_s * 1000 / 2
        merged: List[dict] = []
        next_gid = 0
        for k, si in enumerate(chunk_si):
            offset = starts_s[k] * 1000
            is_last = k == len(chunk_si) - 1
            spk_map: dict = {}
            if k > 0:
                head_texts: dict = {}
                for s in si:
                    if s.get("spk") is not None and s["start"] < b:
                        head_texts.setdefault(s["spk"], []).append(s["sentence"])
                prev_tail: dict = {}
                for s in merged:
                    if s.get("spk") is not None and offset <= s["start"] < offset + b:
                        prev_tail.setdefault(s["spk"], []).append(s["sentence"])
                for spk, parts in head_texts.items():
                    cur = "".join(parts)
                    best_gid, best_ratio = None, 0.0
                    for gid, gparts in prev_tail.items():
                        ratio = difflib.SequenceMatcher(
                            None, cur, "".join(gparts)).ratio()
                        if ratio > best_ratio:
                            best_gid, best_ratio = gid, ratio
                    if best_gid is not None and best_ratio >= 0.5:
                        spk_map[spk] = best_gid
                        logger.info(f"切片 {k} 说话人 {spk} → 全局说话人 {best_gid}"
                                    f"（接缝文本相似度 {best_ratio:.2f}）")
            for s in si:
                if k > 0 and s["start"] < b:
                    continue
                if not is_last and s["start"] >= max_s * 1000 - b:
                    continue
                spk = s.get("spk")
                if spk is not None:
                    if spk not in spk_map:
                        spk_map[spk] = next_gid
                        next_gid += 1
                    spk = spk_map[spk]
                merged.append({
                    "sentence": s["sentence"],
                    "start": int(s["start"] + offset),
                    "end": int(s["end"] + offset),
                    "spk": spk,
                })
        return merged

    def _transcribe_file_via_cloud(self, audio_path: str, language: Optional[str],
                                   spk_num: Optional[int]) -> tuple:
        """单文件完整链路：OSS 上传 → 提交 → 轮询 → 下载解析，返回 (text, si, lang_votes)。"""
        bucket = key = None
        try:
            bucket, key, file_url = self._stage_to_oss(audio_path)
            task_id = self._submit(file_url, language, spk_num)
            logger.info(f"DashScope 任务已提交: task_id={task_id}"
                        f"（{os.path.basename(audio_path)}），开始轮询")
            output = self._wait_task(task_id)
            logger.info(f"DashScope 任务完成: {(output.get('task_metrics') or {})}")
            data = self._fetch_result(output)
            return self._collect_sentences(data)
        finally:
            if bucket is not None and key is not None:
                try:
                    bucket.delete_object(key)
                except Exception:
                    logger.warning(f"OSS 临时文件删除失败: {key}（生命周期规则会兜底）")

    def _transcribe_chunked(self, audio_path: str, language: Optional[str],
                            spk_num: Optional[int], duration: float) -> tuple:
        """长音频切片并发转写 + 合并，返回 (text, si, lang_votes)。"""
        import subprocess
        import tempfile
        import uuid as _uuid
        from concurrent.futures import ThreadPoolExecutor
        max_s, overlap = self._chunk_max_s(), self._chunk_overlap_s()
        step = max_s - overlap
        starts: List[float] = []
        s = 0.0
        while s < duration - 1:
            starts.append(s)
            s += step
        logger.info(f"音频 {duration / 60:.1f} 分钟超过分离单片上限 "
                    f"{max_s / 60:.0f} 分钟，切为 {len(starts)} 片（重叠 {overlap:.0f}s）并发转写")
        tag = _uuid.uuid4().hex[:8]
        chunk_files: List[str] = []
        try:
            for i, st in enumerate(starts):
                out = os.path.join(tempfile.gettempdir(), f"dashscope_chunk_{tag}_{i}.wav")
                subprocess.run(
                    ["ffmpeg", "-y", "-nostdin", "-ss", f"{st:.3f}",
                     "-t", f"{min(max_s, duration - st):.3f}",
                     "-i", audio_path, "-ac", "1", "-ar", "16000", out],
                    check=True, capture_output=True)
                chunk_files.append(out)
            with ThreadPoolExecutor(max_workers=min(4, len(chunk_files))) as ex:
                results = list(ex.map(
                    lambda p: self._transcribe_file_via_cloud(p, language, spk_num),
                    chunk_files))
        finally:
            for p in chunk_files:
                try:
                    os.remove(p)
                except OSError:
                    pass
        merged_si = self._merge_chunks([r[1] for r in results], starts, max_s, overlap)
        lang_votes: dict = {}
        for r in results:
            for lg, w in r[2].items():
                lang_votes[lg] = lang_votes.get(lg, 0) + w
        text = "".join(s["sentence"] for s in merged_si).strip()
        return text, merged_si, lang_votes

    # ---- 主流程 ----
    def transcribe_sync(
        self,
        audio_path: str,
        language: Optional[str],
        spk_num: Optional[int] = None,
    ) -> TranscriptResult:
        # 该模型未返回句级 language 字段时，回退到 language_hints 首个语言
        fallback = language or (self._language_hints(None) or [None])[0]
        duration = self._probe_duration(audio_path)
        if self._diarization_enabled(spk_num) and duration > self._chunk_max_s():
            text, si, lang_votes = self._transcribe_chunked(
                audio_path, language, spk_num, duration)
        else:
            text, si, lang_votes = self._transcribe_file_via_cloud(
                audio_path, language, spk_num)
        return self._to_transcript_result(text, si, lang_votes, fallback)


# ======================================================================
# FunASR 后端（默认）：SenseVoice-Small / Paraformer-large
# ======================================================================

# SenseVoice 前缀标签里的语言码 → ISO
_SENSEVOICE_LANG_TAG = {
    "zh": "zh", "en": "en", "ja": "ja", "ko": "ko",
    "yue": "zh-yue", "auto": None,
}


class _FunASRBackend:
    """FunASR 后端。模型由 FUNASR_MODEL 选择：paraformer（默认）/ sensevoice / 自定义 repo id。"""

    def __init__(self):
        self.model = None
        self._kind = None

    def _load(self):
        if self.model is not None:
            return
        from funasr import AutoModel

        kind = (os.getenv("FUNASR_MODEL", "paraformer").strip().lower() or "paraformer")
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
            # 说话人分离（diarization）：cam++ 声纹嵌入 + 自动聚类，逐句输出 spk 标签。
            # 依赖 punc 模型 + 时间戳，仅 paraformer 路径可用（SenseVoice 无时间戳）。
            spk = os.getenv("FUNASR_SPK", "").strip()
            if spk:
                kwargs["spk_model"] = spk
                logger.info(
                    f"加载 FunASR Paraformer-large（device={device}, vad={vad}, "
                    f"punc=ct-punc, spk={spk}）"
                )
            else:
                logger.info(f"加载 FunASR Paraformer-large（device={device}, vad={vad}, punc=ct-punc）")
        else:
            kwargs["model"] = kind   # 允许直接填 ModelScope repo id
            if os.getenv("FUNASR_SPK", "").strip():
                logger.warning("FUNASR_SPK 仅在 paraformer 路径生效，自定义模型忽略说话人分离")
            logger.info(f"加载 FunASR 模型: {kind}（device={device}, vad={vad}）")

        self.model = AutoModel(**kwargs)
        self._kind = kind

    def transcribe_sync(
        self,
        audio_path: str,
        language: Optional[str],
        spk_num: Optional[int] = None,
    ) -> TranscriptResult:
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
            # 固定说话人数（可选）：请求参数优先于 FUNASR_SPK_NUM 环境变量；
            # 已知双人播客/访谈时聚类更稳
            effective_spk_num = spk_num
            if effective_spk_num is None:
                spk_num_env = os.getenv("FUNASR_SPK_NUM", "").strip()
                if spk_num_env:
                    try:
                        effective_spk_num = int(spk_num_env)
                    except (ValueError, TypeError):
                        logger.warning(f"FUNASR_SPK_NUM 非法: {spk_num_env!r}，忽略")
            if effective_spk_num:
                gen_kwargs["preset_spk_num"] = effective_spk_num

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

        # Paraformer 返回 sentence_info（带毫秒时间戳，开 FUNASR_SPK 时含 spk 标签）；
        # SenseVoice 无（纯文本）。聚合规则：说话人变化 > 停顿阈值 > 字数上限。
        segments = self._paragraphs_from_sentence_info(r.get("sentence_info"))

        if not text and segments:
            text = " ".join(s.text for s in segments)

        return TranscriptResult(text=text, detected_language=lang, segments=segments)

    @staticmethod
    def _paragraphs_from_sentence_info(si) -> List[Segment]:
        """句级片段聚合成段落：说话人变化（优先）/ 句间停顿 / 累积字数 超限即分段；
        sentence_info 带 spk 时段落文本加「说话人A：」前缀。"""
        if not si:
            return []
        try:
            pause_threshold = float(os.getenv("FUNASR_PARAGRAPH_PAUSE", "1.5"))
        except (ValueError, TypeError):
            pause_threshold = 1.5
        try:
            length_limit = int(os.getenv("FUNASR_PARAGRAPH_CHARS", "300"))
        except (ValueError, TypeError):
            length_limit = 300

        segments: List[Segment] = []
        para_texts: List[str] = []
        para_start = 0
        para_end = 0
        para_len = 0
        para_spk: Optional[int] = None
        spk_seen: set = set()

        def _flush_para() -> None:
            nonlocal para_texts, para_start, para_end, para_len, para_spk
            if para_texts:
                body = "".join(para_texts)
                if para_spk is not None:
                    body = f"{_speaker_label(para_spk)}：{body}"
                segments.append(Segment(
                    start_fmt=_format_time_ms(para_start),
                    end_fmt=_format_time_ms(para_end),
                    text=body,
                ))
                para_texts = []
                para_start = 0
                para_end = 0
                para_len = 0
                para_spk = None

        for s in si:
            seg_text = (s.get("sentence") or s.get("text") or "").strip()
            if not seg_text:
                continue
            s_start = int(s.get("start", 0) or 0)
            s_end = int(s.get("end", 0) or 0)
            s_spk = s.get("spk")
            if isinstance(s_spk, int):
                spk_seen.add(s_spk)
            if para_texts and (
                (s_spk is not None and para_spk is not None and s_spk != para_spk)
                or (s_start - para_end) / 1000.0 > pause_threshold
                or para_len >= length_limit
            ):
                _flush_para()
            if not para_texts:
                para_start = s_start
            # 段首若干句无标签、后句才有时，用首个出现的说话人作段落标签
            if para_spk is None and s_spk is not None:
                para_spk = s_spk
            para_texts.append(seg_text)
            para_end = s_end
            para_len += len(seg_text)
        _flush_para()

        if spk_seen:
            logger.info(f"说话人分离: 识别到 {len(spk_seen)} 个说话人")
        return segments

    @staticmethod
    def _parse_language(raw_text: str, result) -> Optional[str]:
        # SenseVoice 可能混入情绪/事件/噪声标签；只接受已知语言并继续扫描后续标签。
        supported_iso = set(LANG_NAME_TO_ISO.values())
        for matched in re.finditer(r"<\|([a-zA-Z]{2,8})\|>", raw_text or ""):
            tag = matched.group(1).lower()
            if tag in _SENSEVOICE_LANG_TAG:
                mapped = _SENSEVOICE_LANG_TAG[tag]
                if mapped:
                    return mapped
                continue
            if tag in supported_iso:
                return tag
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

    def transcribe_sync(
        self,
        audio_path: str,
        language: Optional[str],
        spk_num: Optional[int] = None,  # Qwen 后端无说话人分离，忽略
    ) -> TranscriptResult:
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
