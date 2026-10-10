"""优化结果缓存：同 (输入文本, 模型, 档位) 的无损优化/口语清理结果直接复用。

- 键 = sha256(profile | model | 正文)，文件即 temp/optimize_cache/{key}.md（哈希可反查，无需索引）
- 命中收益：整个优化/清理阶段零 LLM 调用（长视频省几分钟 + 几十次调用）
- 适用场景：同模型重试、服务重启续跑、只调摘要/命名等下游参数的重跑；
  换 LLM 模型对比时键变化，会真实重跑（这正是要测的东西）
- prompt 变更时需 bump 本文件的 profile 版本号（lossless-v1 / cleanup-v1），否则旧缓存会被复用
- OPTIMIZE_CACHE=0 关闭；写入用临时文件 + os.replace 保证原子性，避免读到半截文件
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

TEMP_DIR = Path(__file__).parent.parent / "temp"
CACHE_DIR = TEMP_DIR / "optimize_cache"

# 档位版本号：prompt / 校验规则变更时 bump，旧缓存自然失效
LOSSLESS_PROFILE = "lossless-v1"
CLEANUP_PROFILE = "cleanup-v1"


def enabled() -> bool:
    return os.getenv("OPTIMIZE_CACHE", "1").strip() not in ("0", "false", "False")


def make_key(text: str, model: str, profile: str) -> str:
    payload = f"{profile}|{model or ''}|".encode("utf-8") + (text or "").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _path(key: str) -> Path:
    return CACHE_DIR / f"{key}.md"


def lookup(key: str) -> Optional[str]:
    if not enabled():
        return None
    try:
        text = _path(key).read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except Exception:
        logger.exception("读取优化缓存失败: %s", key)
        return None
    if not text.strip():
        return None
    return text


def store(key: str, text: str) -> None:
    if not enabled() or not (text or "").strip():
        return
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = _path(key).with_suffix(".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, _path(key))
    except Exception:
        logger.exception("写入优化缓存失败，不影响任务流程")
