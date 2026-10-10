"""媒体缓存：同 URL 复用已下载音频与原始转录，避免重复下载/ASR 转录。

索引 temp/media_cache.json:
    {url_key: {url, audio_file, raw_file, video_title, detected_language, asr_key, updated_at}}

- raw 级命中：零网络，跳过字幕检测/下载/转录直接进优化管线（要求 asr_key 与当前 ASR 配置一致）
- audio 级命中：跳过下载，仅重跑转录（音频文件与 ASR 配置无关）
- asr_key：ASR 相关环境变量指纹，换模型/热词后 raw 自动失效重转录；字幕来源固定为 "subtitle"
- 文件缺失的条目在查找时惰性清理；DELETE 任务接口不删文件，缓存天然存活

手动作废：删 media_cache.json 或对应缓存文件；环境变量 MEDIA_CACHE=0 全局关闭。
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

TEMP_DIR = Path(__file__).parent.parent / "temp"
INDEX_FILE = TEMP_DIR / "media_cache.json"

# 影响 ASR 输出的环境变量（换任何一个，raw 缓存即失效）
_ASR_ENV_KEYS = (
    "ASR_BACKEND",
    "FUNASR_MODEL",
    "FUNASR_HOTWORD",
    "FUNASR_LANG",
    "FUNASR_PARAGRAPH_PAUSE",
    "FUNASR_PARAGRAPH_CHARS",
    "FUNASR_SPK",
    "FUNASR_SPK_NUM",
    "DASHSCOPE_ASR_MODEL",
    "DASHSCOPE_DIARIZATION",
    "DASHSCOPE_LANGUAGE_HINTS",
)


def enabled() -> bool:
    return os.getenv("MEDIA_CACHE", "1").strip() not in ("0", "false", "False")


def cache_key(url: str) -> Optional[str]:
    """仅 http/https URL 参与缓存；其余协议返回 None（不缓存也不命中）。"""
    key = (url or "").strip()
    if key.lower().startswith(("http://", "https://")):
        return key
    return None


def current_asr_key(spk_num: Optional[int] = None) -> str:
    """ASR 配置指纹。spk_num 为按请求的说话人数（优先于 FUNASR_SPK_NUM 环境变量）。"""
    envs = [f"{k}={os.getenv(k, '')}" for k in _ASR_ENV_KEYS if k != "FUNASR_SPK_NUM"]
    envs.append(f"FUNASR_SPK_NUM={spk_num if spk_num is not None else os.getenv('FUNASR_SPK_NUM', '')}")
    return "|".join(envs)


def _load_index() -> dict:
    try:
        if INDEX_FILE.exists():
            data = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        logger.exception("读取媒体缓存索引失败，按空索引处理")
    return {}


def _save_index(index: dict) -> None:
    try:
        TEMP_DIR.mkdir(exist_ok=True)
        INDEX_FILE.write_text(
            json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        logger.exception("写入媒体缓存索引失败，不影响任务流程")


def _raw_filename(key: str) -> str:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:10]
    return f"cache_raw_{digest}.md"


def lookup_raw(url: str, asr_key: Optional[str] = None) -> Optional[dict]:
    """命中返回 {raw_text, video_title, detected_language}；asr_key 不匹配或文件缺失返回 None。"""
    key = cache_key(url)
    if key is None or not enabled():
        return None
    index = _load_index()
    entry = index.get(key)
    if not isinstance(entry, dict):
        return None
    raw_file = entry.get("raw_file")
    if not raw_file or not (TEMP_DIR / raw_file).exists():
        if raw_file:
            entry.pop("raw_file", None)
            _save_index(index)
        return None
    if entry.get("asr_key") != (asr_key if asr_key is not None else current_asr_key()):
        return None
    try:
        raw_text = (TEMP_DIR / raw_file).read_text(encoding="utf-8")
    except Exception:
        logger.exception("读取缓存转录文件失败: %s", raw_file)
        return None
    if not raw_text.strip():
        return None
    return {
        "raw_text": raw_text,
        "video_title": entry.get("video_title") or "",
        "detected_language": entry.get("detected_language"),
    }


def lookup_audio(url: str) -> Optional[dict]:
    """命中返回 {path, video_title}；文件缺失返回 None（并惰性清理条目）。"""
    key = cache_key(url)
    if key is None or not enabled():
        return None
    index = _load_index()
    entry = index.get(key)
    if not isinstance(entry, dict):
        return None
    audio_file = entry.get("audio_file")
    if not audio_file:
        return None
    path = TEMP_DIR / audio_file
    if not path.exists():
        entry.pop("audio_file", None)
        _save_index(index)
        return None
    return {"path": path, "video_title": entry.get("video_title") or ""}


def record_raw(
    url: str,
    raw_text: str,
    video_title: str = "",
    detected_language: Optional[str] = None,
    asr_key: Optional[str] = None,
) -> None:
    """转录/字幕完成后登记 raw 缓存（保留已有 audio_file）。放在优化管线之前，
    即使优化失败，下次任务也可免下载+转录。"""
    key = cache_key(url)
    if key is None or not enabled() or not (raw_text or "").strip():
        return
    index = _load_index()
    entry = index.get(key) if isinstance(index.get(key), dict) else {}
    entry.update({
        "url": key,
        "raw_file": _raw_filename(key),
        "video_title": video_title or entry.get("video_title", ""),
        "detected_language": detected_language,
        "asr_key": asr_key if asr_key is not None else current_asr_key(),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    })
    try:
        (TEMP_DIR / entry["raw_file"]).write_text(raw_text, encoding="utf-8")
    except Exception:
        logger.exception("写入缓存转录文件失败: %s", entry["raw_file"])
        return
    index[key] = entry
    _save_index(index)
    logger.info(f"媒体缓存(raw)已登记: {key} -> {entry['raw_file']}")


def record_audio(url: str, audio_path, video_title: str = "") -> None:
    key = cache_key(url)
    if key is None or not enabled():
        return
    path = Path(audio_path)
    if not path.exists():
        return
    index = _load_index()
    entry = index.get(key) if isinstance(index.get(key), dict) else {}
    entry.update({
        "url": key,
        "audio_file": path.name,
        # 标题仅在尚未有值时补写（raw 登记的标题通常更准）
        "video_title": entry.get("video_title") or video_title,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    })
    index[key] = entry
    _save_index(index)
    logger.info(f"媒体缓存(audio)已登记: {key} -> {path.name}")
