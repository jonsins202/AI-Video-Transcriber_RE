from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import os
import time
import asyncio
import logging
import logging.handlers
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, List
import aiofiles
import uuid
import json
import re
import openai
from pydantic import BaseModel, Field

# 加载项目根 .env（已有环境变量优先，不覆盖）；必须在 backend 各模块 import 前执行，
# 因为 video_processor 等模块在 import 时就读取环境变量
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from video_processor import VideoProcessor
from transcriber import Transcriber, rename_speaker_labels
from summarizer import Summarizer
from translator import Translator
import media_cache
import optimize_cache

# ── 日志配置：控制台 + 文件（按天轮转，保留 30 天）─────────────────────
def setup_logging():
    """配置根 logger：控制台 + 文件双输出，所有模块的 logger 自动落盘到 logs/app.log"""
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    fmt = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    fh = logging.handlers.TimedRotatingFileHandler(
        log_dir / "app.log", when='midnight', backupCount=30, encoding='utf-8',
    )
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(fh)
    root.addHandler(ch)


setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="AI视频转录器", version="1.0.0")

# CORS中间件配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 挂载静态文件
app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "static")), name="static")

# 创建临时目录
TEMP_DIR = PROJECT_ROOT / "temp"
TEMP_DIR.mkdir(exist_ok=True)

# 初始化处理器
video_processor = VideoProcessor()
transcriber = Transcriber()
summarizer = Summarizer()
translator = Translator()

# 启动横幅：验证 .env/环境变量是否真的生效（grep "服务启动"）
logger.info(
    "服务启动: "
    f"ASR={os.getenv('ASR_BACKEND', 'funasr')}/{os.getenv('FUNASR_MODEL', 'paraformer')}, "
    f"说话人分离={os.getenv('FUNASR_SPK') or '关'}, "
    f"固定人数={os.getenv('FUNASR_SPK_NUM') or '自动'}, "
    f"媒体缓存={'开' if media_cache.enabled() else '关'}"
)

# 存储任务状态 - 使用文件持久化
import threading

TASKS_FILE = TEMP_DIR / "tasks.json"
tasks_lock = threading.Lock()

def load_tasks():
    """加载任务状态"""
    try:
        if TASKS_FILE.exists():
            with open(TASKS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except:
        pass
    return {}

def save_tasks(tasks_data):
    """保存任务状态"""
    try:
        with tasks_lock:
            with open(TASKS_FILE, 'w', encoding='utf-8') as f:
                json.dump(tasks_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"保存任务状态失败: {e}")

async def broadcast_task_update(task_id: str, task_data: dict):
    """向所有连接的SSE客户端广播任务状态更新"""
    logger.info(f"广播任务更新: {task_id}, 状态: {task_data.get('status')}, 连接数: {len(sse_connections.get(task_id, []))}")
    if task_id in sse_connections:
        connections_to_remove = []
        for queue in sse_connections[task_id]:
            try:
                await queue.put(json.dumps(task_data, ensure_ascii=False))
                logger.debug(f"消息已发送到队列: {task_id}")
            except Exception as e:
                logger.warning(f"发送消息到队列失败: {e}")
                connections_to_remove.append(queue)
        
        # 移除断开的连接
        for queue in connections_to_remove:
            sse_connections[task_id].remove(queue)
        
        # 如果没有连接了，清理该任务的连接列表
        if not sse_connections[task_id]:
            del sse_connections[task_id]

# 启动时加载任务状态
tasks = load_tasks()
# 存储正在处理的URL，防止重复处理
processing_urls = set()
# 存储活跃的任务对象，用于控制和取消
active_tasks = {}
# 存储SSE连接，用于实时推送状态更新
sse_connections = {}

# ── 批量队列：串行 worker（GPU/LLM 并发限制，一次只处理一个任务）──────────
_task_queue: asyncio.Queue = asyncio.Queue()
_queue_worker_task: Optional[asyncio.Task] = None


async def _queue_worker():
    """串行队列消费者：逐个处理 queued 任务，任何异常都不让 worker 退出。"""
    while True:
        try:
            task_id = await _task_queue.get()
            t = tasks.get(task_id)
            if t is None or t.get("queue_status") != "queued":
                continue  # 任务已被删除或取消

            settings = t.get("settings") or {}
            url = t.get("url") or ""

            t.update({
                "queue_status": "running",
                "status": "processing",
                "message": "开始处理...",
            })
            save_tasks(tasks)
            await broadcast_task_update(task_id, t)
            processing_urls.add(url)

            try:
                runner = asyncio.create_task(process_video_task(
                    task_id,
                    url,
                    settings.get("summary_language", "zh"),
                    settings.get("api_key", ""),
                    settings.get("model_base_url", ""),
                    settings.get("model_id", ""),
                    settings.get("oral_cleanup", False),
                    _parse_spk_num(settings.get("spk_num", "")),
                ))
                active_tasks[task_id] = runner
                await runner
            except asyncio.CancelledError:
                # runner 被 DELETE 取消时任务记录已删除 → 静默收尾；
                # worker 自身被取消（应用关闭）→ 上抛让 worker 退出
                if tasks.get(task_id) is not None:
                    raise
            finally:
                if task_id in active_tasks:
                    del active_tasks[task_id]
                t = tasks.get(task_id)
                if t is not None:
                    # 正常完成/失败 → done；异常中断（如服务关闭）→ interrupted，重启后可重新入队
                    t["queue_status"] = "done" if t.get("status") in ("completed", "error") else "interrupted"
                    save_tasks(tasks)
                    await broadcast_task_update(task_id, t)
        except Exception:
            logger.exception("队列 worker 异常，继续处理下一个任务")


def _strip_first_heading(text: str) -> str:
    """去掉正文首个非空行的一级标题（合并文档的章节标题已携带标题）。"""
    lines = text.replace("\r\n", "\n").split("\n")
    for i, line in enumerate(lines):
        if line.strip():
            if line.strip().startswith("# "):
                del lines[i]
            break
    return "\n".join(lines).strip()


@contextmanager
def _stage(task_id: str, stage: str):
    """记录任务阶段耗时：写日志 + tasks[task_id].timeline。
    用法: with _stage(task_id, "transcribe"): await transcriber.transcribe(...)"""
    t0 = time.perf_counter()
    short = task_id.replace("-", "")[:8]
    logger.info(f"[task:{short}] --> {stage}")
    try:
        yield
    finally:
        duration = time.perf_counter() - t0
        logger.info(f"[task:{short}] <-- {stage} ({duration:.1f}s)")
        if task_id in tasks:
            tasks[task_id].setdefault("timeline", {})[stage] = round(duration, 1)

# 本地上传：允许的类型与大小上限（MB），可用环境变量 UPLOAD_MAX_MB 调整
UPLOAD_ALLOWED_EXT = frozenset({".txt", ".mp3", ".mp4", ".m4a", ".wav", ".webm", ".mkv", ".ogg", ".flac"})
UPLOAD_MAX_MB = int(os.getenv("UPLOAD_MAX_MB", "200"))


def _sanitize_title_for_filename(title: str) -> str:
    """将视频标题清洗为安全的文件名片段。"""
    if not title:
        return "untitled"
    # 仅保留字母数字、下划线、连字符与空格
    safe = re.sub(r"[^\w\-\s]", "", title)
    # 压缩空白并转为下划线
    safe = re.sub(r"\s+", "_", safe).strip("._-")
    # 最长限制，避免过长文件名问题
    return safe[:80] or "untitled"


def _txt_to_raw_transcript_markdown(body: str) -> str:
    """将纯文本包装为与 Whisper 输出结构一致的 Markdown。"""
    text = body.strip() if body.strip() else "(empty)"
    return "\n".join([
        "# Video Transcription",
        "",
        "**Detected Language:**",
        "**Language Probability:** —",
        "",
        "## Transcription Content",
        "",
        text,
    ])


async def _write_optimization_audit(path: Path, audit: dict) -> None:
    """将不含正文的优化审计元数据写入 JSON。"""
    async with aiofiles.open(path, "w", encoding="utf-8") as f:
        await f.write(json.dumps(audit, ensure_ascii=False, indent=2))


def _parse_spk_num(raw: str) -> Optional[int]:
    """前端说话人数输入 → 1-30 的整数；空/非法返回 None（自动检测）。"""
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        n = int(raw)
    except (ValueError, TypeError):
        return None
    return n if 1 <= n <= 30 else None


def _make_stage_progress_callback(task_id: str, start: int, end: int, label: str):
    """构造 on_progress(done, total)：把批次完成进度映射到 start→end 的进度条。

    单调保护：asyncio 单线程下 completed+=1 到写 progress 之间无 await，再加
    pct<=当前值则跳过，确保进度绝不倒退。调用方需保证 start 大于前一阶段终值。"""
    async def _on_progress(done: int, total: int) -> None:
        if total <= 0:
            return
        pct = start + round((done / total) * (end - start))
        task = tasks.get(task_id)
        if task is None:
            return
        if pct <= task.get("progress", 0):
            return
        task["progress"] = pct
        task["message"] = f"{label} ({done}/{total})"
        save_tasks(tasks)
        await broadcast_task_update(task_id, task)
    return _on_progress


async def _run_post_extract_pipeline(
    task_id: str,
    raw_script: str,
    video_title: str,
    source_ref: str,
    summary_language: str,
    request_summarizer: Summarizer,
    dedup_url: Optional[str] = None,
    api_key: str = "",
    model_base_url: str = "",
    model_id: str = "",
    oral_cleanup: bool = False,
) -> None:
    """取得 raw_script 后的共用管线：归档、优化、翻译、摘要、广播。"""
    short_id = task_id.replace("-", "")[:6]
    safe_title = _sanitize_title_for_filename(video_title)

    try:
        raw_md_filename = f"raw_{safe_title}_{short_id}.md"
        raw_md_path = TEMP_DIR / raw_md_filename
        with open(raw_md_path, "w", encoding="utf-8") as f:
            f.write((raw_script or "") + f"\n\nsource: {source_ref}\n")
        tasks[task_id].update({"raw_script_file": raw_md_filename})
        save_tasks(tasks)
        await broadcast_task_update(task_id, tasks[task_id])
    except Exception:
        logger.exception("保存原始转录 Markdown 失败，停止后续优化")
        raise

    tasks[task_id].update({
        "progress": 55,
        "message": "正在优化转录文本...",
    })
    save_tasks(tasks)
    await broadcast_task_update(task_id, tasks[task_id])

    transcript_body = request_summarizer.extract_transcript_body(raw_script)
    if os.getenv("SKIP_OPTIMIZE", "").strip() in ("1", "true", "True"):
        # 云端 ASR 自带标点、错字很少时可跳过无损优化，省一次 LLM 调用
        request_summarizer.reset_audits()
        script = transcript_body
        tasks[task_id].update({"progress": 63, "message": "已跳过无损优化（SKIP_OPTIMIZE）"})
        save_tasks(tasks)
        await broadcast_task_update(task_id, tasks[task_id])
        logger.info(f"SKIP_OPTIMIZE=1，跳过无损优化: {task_id}")
    else:
        optimize_key = optimize_cache.make_key(
            transcript_body, request_summarizer.fast_model, optimize_cache.LOSSLESS_PROFILE
        )
        cached_script = optimize_cache.lookup(optimize_key)
        if cached_script is not None:
            # 缓存命中跳过了优化调用，需清掉全局单例可能残留的上一任务 audit
            request_summarizer.reset_audits()
            script = cached_script
            tasks[task_id]["optimize_cache"] = "hit"
            tasks[task_id].update({"progress": 63, "message": "命中优化缓存，跳过无损优化"})
            save_tasks(tasks)
            await broadcast_task_update(task_id, tasks[task_id])
            logger.info(f"优化缓存命中，跳过无损优化: {task_id}")
        else:
            with _stage(task_id, "optimize"):
                script = await request_summarizer.optimize_transcript_body(
                    transcript_body,
                    on_progress=_make_stage_progress_callback(
                        task_id, 55, 63, "正在优化转录文本..."
                    ),
                )
            optimize_cache.store(optimize_key, script)

    audit = request_summarizer.get_last_optimization_audit()
    if audit:
        try:
            audit_filename = f"optimization_audit_{safe_title}_{short_id}.json"
            await _write_optimization_audit(TEMP_DIR / audit_filename, audit)
            tasks[task_id]["optimization_audit_file"] = audit_filename
            save_tasks(tasks)
        except Exception:
            logger.exception("保存无损优化审计失败，不影响转录结果")

    script_with_title = f"# {video_title}\n\n{script}\n\nsource: {source_ref}\n"

    if oral_cleanup:
        # 口语清理是非逐字操作：先归档无损优化后的逐字稿，清理失败也有底稿
        try:
            verbatim_filename = f"verbatim_{safe_title}_{short_id}.md"
            with open(TEMP_DIR / verbatim_filename, "w", encoding="utf-8") as f:
                f.write(script_with_title)
            tasks[task_id]["verbatim_script_file"] = verbatim_filename
            tasks[task_id]["oral_cleanup"] = True
            save_tasks(tasks)
        except Exception:
            logger.exception("保存逐字稿失败，不影响后续清理")

        tasks[task_id].update({
            "progress": 65,
            "message": "正在清理口语...",
        })
        save_tasks(tasks)
        await broadcast_task_update(task_id, tasks[task_id])

        cleanup_key = optimize_cache.make_key(
            script, request_summarizer.fast_model, optimize_cache.CLEANUP_PROFILE
        )
        cached_cleanup = optimize_cache.lookup(cleanup_key)
        if cached_cleanup is not None:
            request_summarizer.reset_audits()
            script = cached_cleanup
            tasks[task_id]["cleanup_cache"] = "hit"
            tasks[task_id].update({"progress": 72, "message": "命中清理缓存，跳过口语清理"})
            save_tasks(tasks)
            await broadcast_task_update(task_id, tasks[task_id])
            logger.info(f"清理缓存命中，跳过口语清理: {task_id}")
        else:
            with _stage(task_id, "oral_cleanup"):
                script = await request_summarizer.cleanup_transcript_body(
                    script,
                    on_progress=_make_stage_progress_callback(
                        task_id, 66, 72, "正在清理口语..."
                    ),
                )
            optimize_cache.store(cleanup_key, script)
        script_with_title = f"# {video_title}\n\n{script}\n\nsource: {source_ref}\n"

        cleanup_audit = request_summarizer.get_last_cleanup_audit()
        if cleanup_audit:
            try:
                cleanup_audit_filename = f"cleanup_audit_{safe_title}_{short_id}.json"
                await _write_optimization_audit(TEMP_DIR / cleanup_audit_filename, cleanup_audit)
                tasks[task_id]["cleanup_audit_file"] = cleanup_audit_filename
                save_tasks(tasks)
            except Exception:
                logger.exception("保存口语清理审计失败，不影响转录结果")

    detected_language = transcriber.get_detected_language(raw_script)
    detected_language = (detected_language or "").strip()
    if not detected_language:
        detected_language = translator.infer_language_code(raw_script)
    detected_language = translator.normalize_lang_code(detected_language) or detected_language

    logger.info(f"检测到的语言: {detected_language}, 摘要语言: {summary_language}")

    translation_content = None
    translation_filename = None
    translation_path = None

    eff_key = (api_key or "").strip()
    eff_base = (model_base_url or "").strip().rstrip("/")
    if eff_key:
        request_translator = Translator(
            api_key=eff_key,
            base_url=eff_base or None,
            model=model_id or None,
        )
    else:
        request_translator = translator

    need_translation = translator.languages_differ_for_translation(
        detected_language, summary_language
    )

    if need_translation:
        logger.info(f"需要翻译: {detected_language} -> {summary_language}")
        tasks[task_id].update({
            "progress": 75,
            "message": "正在生成翻译...",
        })
        save_tasks(tasks)
        await broadcast_task_update(task_id, tasks[task_id])

        with _stage(task_id, "translate"):
            translation_content = await request_translator.translate_text(
                script, summary_language, detected_language
            )
        translation_with_title = f"# {video_title}\n\n{translation_content}\n\nsource: {source_ref}\n"
        translation_filename = f"translation_{safe_title}_{short_id}.md"
        translation_path = TEMP_DIR / translation_filename
        async with aiofiles.open(translation_path, "w", encoding="utf-8") as f:
            await f.write(translation_with_title)
    else:
        logger.info(
            f"不需要翻译: detected_language={detected_language}, summary_language={summary_language}, "
            f"need_translation={need_translation}"
        )

    tasks[task_id].update({
        "progress": 85,
        "message": "正在生成摘要...",
    })
    save_tasks(tasks)
    await broadcast_task_update(task_id, tasks[task_id])

    with _stage(task_id, "summarize"):
        summary = await request_summarizer.summarize(script, summary_language, video_title)
    summary_with_source = summary + f"\n\nsource: {source_ref}\n"

    script_filename = f"transcript_{task_id}.md"
    script_path = TEMP_DIR / script_filename
    async with aiofiles.open(script_path, "w", encoding="utf-8") as f:
        await f.write(script_with_title)

    new_script_filename = f"transcript_{safe_title}_{short_id}.md"
    new_script_path = TEMP_DIR / new_script_filename
    try:
        if script_path.exists():
            script_path.rename(new_script_path)
            script_path = new_script_path
    except Exception:
        pass

    summary_filename = f"summary_{safe_title}_{short_id}.md"
    summary_path = TEMP_DIR / summary_filename
    async with aiofiles.open(summary_path, "w", encoding="utf-8") as f:
        await f.write(summary_with_source)

    task_result = {
        "status": "completed",
        "progress": 100,
        "message": "处理完成！",
        "video_title": video_title,
        "script": script_with_title,
        "summary": summary_with_source,
        "script_path": str(script_path),
        "summary_path": str(summary_path),
        "short_id": short_id,
        "safe_title": safe_title,
        "detected_language": detected_language,
        "summary_language": summary_language,
    }

    if translation_content and translation_path:
        task_result.update({
            "translation": translation_with_title,
            "translation_path": str(translation_path),
            "translation_filename": translation_filename,
        })

    tasks[task_id].update(task_result)
    save_tasks(tasks)
    # 记录完成时间 + 总耗时（前端展示用）
    from datetime import datetime
    tasks[task_id]["completed_at"] = datetime.now().isoformat(timespec="seconds")
    tasks[task_id]["total_duration"] = round(sum((tasks[task_id].get("timeline") or {}).values()), 1)
    logger.info(f"任务完成，准备广播最终状态: {task_id}")
    await broadcast_task_update(task_id, tasks[task_id])
    logger.info(f"最终状态已广播: {task_id}")

    if dedup_url:
        processing_urls.discard(dedup_url)
    if task_id in active_tasks:
        del active_tasks[task_id]


@app.get("/")
async def read_root():
    """返回前端页面"""
    return FileResponse(str(PROJECT_ROOT / "static" / "index.html"))

@app.post("/api/models")
async def list_models(
    base_url: str = Form(default=""),
    api_key:  str = Form(default=""),
):
    """Proxy: fetch model list from any OpenAI-compatible API."""
    effective_key = api_key or os.getenv("OPENAI_API_KEY", "")
    effective_url = base_url.rstrip("/") or os.getenv("OPENAI_BASE_URL") or None

    if not effective_key:
        raise HTTPException(status_code=400, detail="API key is required")

    try:
        client = openai.OpenAI(api_key=effective_key, base_url=effective_url)
        resp   = await asyncio.to_thread(client.models.list)
        models = [{"id": m.id, "name": getattr(m, "name", m.id)} for m in resp.data]
        # Sort by id for readability
        models.sort(key=lambda x: x["id"])
        return {"data": models}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


async def _enqueue_upload_job(
    file: UploadFile,
    summary_language: str,
    api_key: str,
    model_base_url: str,
    model_id: str,
    oral_cleanup: bool = False,
    spk_num: Optional[int] = None,
) -> dict:
    """保存上传文件并入队 process_upload_task，返回 {task_id, message}。"""
    raw_name = file.filename or "upload.bin"
    if ".." in raw_name or "/" in raw_name or "\\" in raw_name:
        raise HTTPException(status_code=400, detail="Invalid filename")
    safe_name = os.path.basename(raw_name)
    ext = Path(safe_name).suffix.lower()
    if ext not in UPLOAD_ALLOWED_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext or '(none)'}",
        )

    max_bytes = UPLOAD_MAX_MB * 1024 * 1024
    task_id = str(uuid.uuid4())
    unique_stem = task_id.replace("-", "")[:12]
    dest = TEMP_DIR / f"upload_{unique_stem}{ext}"

    total = 0
    with open(dest, "wb") as out_f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                try:
                    dest.unlink(missing_ok=True)
                except Exception:
                    pass
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds limit of {UPLOAD_MAX_MB} MB",
                )
            out_f.write(chunk)

    if total == 0:
        try:
            dest.unlink(missing_ok=True)
        except Exception:
            pass
        raise HTTPException(status_code=400, detail="Empty file")

    video_title = _sanitize_title_for_filename(Path(safe_name).stem) or "upload"
    source_label = f"upload:{safe_name}"

    tasks[task_id] = {
        "status": "processing",
        "progress": 0,
        "message": "开始处理上传文件...",
        "script": None,
        "summary": None,
        "error": None,
        "url": source_label,
    }
    save_tasks(tasks)

    bg = asyncio.create_task(
        process_upload_task(
            task_id,
            dest,
            safe_name,
            video_title,
            ext,
            summary_language,
            api_key,
            model_base_url,
            model_id,
            oral_cleanup,
            spk_num,
        )
    )
    active_tasks[task_id] = bg

    return {"task_id": task_id, "message": "任务已创建，正在处理中..."}


@app.post("/api/process-video")
async def process_video(
    url: str = Form(default=""),
    summary_language: str = Form(default="zh"),
    api_key: str = Form(default=""),
    model_base_url: str = Form(default=""),
    model_id: str = Form(default=""),
    oral_cleanup: bool = Form(default=False),
    spk_num: str = Form(default=""),
    file: Optional[UploadFile] = File(None),
):
    """
    处理视频链接或本地上传（multipart 中带 file 且无有效 URL 时走上传流程）。
    上传与 URL 共用此路径，便于反向代理只放行 /api/process-video 的环境。
    """
    parsed_spk_num = _parse_spk_num(spk_num)
    try:
        if file is not None and (file.filename or "").strip():
            return await _enqueue_upload_job(
                file, summary_language, api_key, model_base_url, model_id, oral_cleanup,
                parsed_spk_num,
            )

        stripped = (url or "").strip()
        if not stripped:
            raise HTTPException(
                status_code=400,
                detail="Provide a video URL or upload a file",
            )

        url = stripped

        # 检查是否已经在处理相同的URL
        if url in processing_urls:
            # 查找现有任务
            for tid, task in tasks.items():
                if task.get("url") == url:
                    return {"task_id": tid, "message": "该视频正在处理中，请等待..."}
            
        # 生成唯一任务ID
        task_id = str(uuid.uuid4())
        
        # 标记URL为正在处理
        processing_urls.add(url)
        
        # 初始化任务状态
        tasks[task_id] = {
            "status": "processing",
            "progress": 0,
            "message": "开始处理视频...",
            "script": None,
            "summary": None,
            "error": None,
            "url": url  # 保存URL用于去重
        }
        save_tasks(tasks)
        
        # 创建并跟踪异步任务
        task = asyncio.create_task(process_video_task(
            task_id, url, summary_language, api_key, model_base_url, model_id, oral_cleanup,
            parsed_spk_num,
        ))
        active_tasks[task_id] = task
        
        return {"task_id": task_id, "message": "任务已创建，正在处理中..."}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"处理视频时出错: {e}")
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

async def process_video_task(
    task_id: str,
    url: str,
    summary_language: str,
    api_key: str = "",
    model_base_url: str = "",
    model_id: str = "",
    oral_cleanup: bool = False,
    spk_num: Optional[int] = None,
):
    """
    异步处理视频任务
    """
    try:
        # ── 阶段一：优先尝试获取平台字幕（快速路径） ──────────────────────
        tasks[task_id].update({
            "status": "processing",
            "progress": 10,
            "message": "正在检测视频字幕..."
        })
        save_tasks(tasks)
        await broadcast_task_update(task_id, tasks[task_id])
        await asyncio.sleep(0.1)

        # 如果前端传入了 API 凭据，创建专用 Summarizer（线程安全，覆盖全局实例）
        if api_key:
            effective_url = model_base_url.rstrip("/") or None
            request_summarizer = Summarizer(
                api_key=api_key,
                base_url=effective_url,
                model=model_id or None,
            )
            logger.info(f"使用前端提供的 API Key，base_url={effective_url}, model={model_id or 'default'}")
        else:
            request_summarizer = summarizer  # 全局实例（使用环境变量）

        # ── 阶段〇：媒体缓存优先（最快路径：同 URL 直接复用转录，零网络） ──
        # 指纹含按请求的说话人数：换人数重转录，不会命中旧稿
        task_asr_key = media_cache.current_asr_key(spk_num=spk_num)
        cached_raw = media_cache.lookup_raw(url, asr_key=task_asr_key)
        if cached_raw is not None:
            video_title = cached_raw["video_title"] or url
            raw_script = cached_raw["raw_text"]
            # 全局 transcriber 单例残留上一个任务的语言，必须显式用缓存值覆盖
            transcriber.last_detected_language = cached_raw.get("detected_language")

            tasks[task_id].update({
                "progress": 40,
                "message": "命中本地转录缓存，跳过下载与转录",
                "cache_hit": "raw",
            })
            save_tasks(tasks)
            await broadcast_task_update(task_id, tasks[task_id])
            logger.info(f"媒体缓存(raw)命中，跳过下载/转录: {url}")
        else:
            with _stage(task_id, "subtitle_fetch"):
                subtitle_text, sub_title, sub_lang = await video_processor.fetch_subtitles(url, TEMP_DIR)

        if cached_raw is not None:
            pass  # 缓存命中：跳过字幕/下载/转录分支
        elif subtitle_text:
            # ── 快速路径：有字幕，跳过音频下载和 ASR ──────────────────
            video_title = sub_title
            raw_script = subtitle_text
            # 把语言写入 transcriber，保持下游逻辑一致
            transcriber.last_detected_language = sub_lang
            media_cache.record_raw(
                url, raw_script, video_title, sub_lang, asr_key="subtitle"
            )

            tasks[task_id].update({
                "progress": 40,
                "message": f"字幕获取成功（{sub_lang}），正在处理文本..."
            })
            save_tasks(tasks)
            await broadcast_task_update(task_id, tasks[task_id])
        else:
            # ── 慢速路径：无字幕，下载音频 → Qwen3-ASR 转录 ─────────────────
            tasks[task_id].update({
                "progress": 15,
                "message": "未找到字幕，正在下载视频音频..."
            })
            save_tasks(tasks)
            await broadcast_task_update(task_id, tasks[task_id])

            cached_audio = media_cache.lookup_audio(url)
            if cached_audio is not None:
                audio_path = str(cached_audio["path"])
                video_title = cached_audio["video_title"] or sub_title or url
                tasks[task_id].update({
                    "progress": 35,
                    "message": "命中本地音频缓存，跳过下载",
                    "cache_hit": "audio",
                })
                save_tasks(tasks)
                await broadcast_task_update(task_id, tasks[task_id])
                logger.info(f"媒体缓存(audio)命中，跳过下载: {url}")
            else:
                with _stage(task_id, "audio_download"):
                    audio_path, video_title = await video_processor.download_and_convert(
                        url, TEMP_DIR, prefetched_title=sub_title or None
                    )
                media_cache.record_audio(url, audio_path, video_title)

                tasks[task_id].update({
                    "progress": 35,
                    "message": "音频下载完成，准备转录..."
                })
                save_tasks(tasks)
                await broadcast_task_update(task_id, tasks[task_id])

            tasks[task_id].update({
                "progress": 40,
                "message": "正在转录音频（Qwen3-ASR）..."
            })
            save_tasks(tasks)
            await broadcast_task_update(task_id, tasks[task_id])

            with _stage(task_id, "transcribe"):
                raw_script = await transcriber.transcribe(audio_path, spk_num=spk_num)
            # 优化管线之前登记：即使优化失败，下次任务也免下载+转录
            media_cache.record_raw(
                url, raw_script, video_title, transcriber.last_detected_language,
                asr_key=task_asr_key,
            )

        await _run_post_extract_pipeline(
            task_id=task_id,
            raw_script=raw_script,
            video_title=video_title,
            source_ref=url,
            summary_language=summary_language,
            request_summarizer=request_summarizer,
            dedup_url=url,
            api_key=api_key,
            model_base_url=model_base_url,
            model_id=model_id,
            oral_cleanup=oral_cleanup,
        )

        # 不要立即删除临时文件！保留给用户下载
        # 文件会在一定时间后自动清理或用户手动清理

    except Exception as e:
        logger.exception(f"任务 {task_id} 处理失败: {e}")
        # 从处理列表中移除URL
        processing_urls.discard(url)
        
        # 从活跃任务列表中移除
        if task_id in active_tasks:
            del active_tasks[task_id]
            
        tasks[task_id].update({
            "status": "error",
            "error": str(e),
            "message": f"处理失败: {str(e)}"
        })
        save_tasks(tasks)
        await broadcast_task_update(task_id, tasks[task_id])

class BatchItem(BaseModel):
    """批量任务单项：字段与 /api/process-video 的表单参数一一对应。"""
    url: str
    summary_language: str = "zh"
    oral_cleanup: bool = False
    spk_num: str = ""          # 保持字符串语义，运行时用 _parse_spk_num 解析
    api_key: str = ""
    model_base_url: str = ""
    model_id: str = ""


class BatchRequest(BaseModel):
    items: List[BatchItem]


@app.post("/api/process-batch")
async def process_batch(req: BatchRequest):
    """批量入队：创建 queued 任务，由串行队列 worker 逐个处理（不立即占用 GPU）。"""
    valid = []
    skipped = []
    seen = set()
    for item in req.items:
        url = (item.url or "").strip()
        if not url:
            continue
        if url in seen:
            skipped.append({"url": url, "reason": "批内重复链接"})
            continue
        seen.add(url)
        if url in processing_urls:
            skipped.append({"url": url, "reason": "该链接正在处理中"})
            continue
        valid.append(item)

    if len(valid) > 10:
        raise HTTPException(status_code=400, detail="批量任务最多支持 10 个链接")
    if not valid:
        raise HTTPException(status_code=400, detail="没有有效的链接")

    batch_id = str(uuid.uuid4())
    created = []
    for item in valid:
        task_id = str(uuid.uuid4())
        tasks[task_id] = {
            "status": "processing",
            "progress": 0,
            "message": "已加入队列，等待处理...",
            "script": None,
            "summary": None,
            "error": None,
            "url": item.url,
            "batch_id": batch_id,
            "queue_status": "queued",
            "settings": {
                "summary_language": item.summary_language,
                "oral_cleanup": item.oral_cleanup,
                "spk_num": item.spk_num,
                "api_key": item.api_key,
                "model_base_url": item.model_base_url,
                "model_id": item.model_id,
            },
        }
        created.append({"task_id": task_id, "url": item.url})

    save_tasks(tasks)
    for c in created:
        await _task_queue.put(c["task_id"])
    return {"batch_id": batch_id, "tasks": created, "skipped": skipped}

@app.post("/api/process-upload")
async def process_upload(
    file: UploadFile = File(...),
    summary_language: str = Form(default="zh"),
    api_key: str = Form(default=""),
    model_base_url: str = Form(default=""),
    model_id: str = Form(default=""),
    oral_cleanup: bool = Form(default=False),
    spk_num: str = Form(default=""),
):
    """独立上传入口；逻辑与 multipart 带 file 的 /api/process-video 相同。"""
    return await _enqueue_upload_job(
        file, summary_language, api_key, model_base_url, model_id, oral_cleanup,
        _parse_spk_num(spk_num),
    )


async def process_upload_task(
    task_id: str,
    saved_path: Path,
    original_name: str,
    video_title: str,
    ext_lower: str,
    summary_language: str,
    api_key: str = "",
    model_base_url: str = "",
    model_id: str = "",
    oral_cleanup: bool = False,
    spk_num: Optional[int] = None,
):
    source_ref = f"upload:{original_name}"
    try:
        if api_key:
            effective_url = model_base_url.rstrip("/") or None
            request_summarizer = Summarizer(
                api_key=api_key,
                base_url=effective_url,
                model=model_id or None,
            )
            logger.info(
                f"上传任务使用前端 API Key，base_url={effective_url}, model={model_id or 'default'}"
            )
        else:
            request_summarizer = summarizer

        if ext_lower == ".txt":
            tasks[task_id].update({
                "progress": 20,
                "message": "正在读取文本文件...",
            })
            save_tasks(tasks)
            await broadcast_task_update(task_id, tasks[task_id])

            body = saved_path.read_text(encoding="utf-8", errors="replace")
            if not body.strip():
                raise Exception("文本文件为空")
            transcriber.last_detected_language = None
            raw_script = _txt_to_raw_transcript_markdown(body)
        else:
            tasks[task_id].update({
                "progress": 15,
                "message": "正在转换音频格式...",
            })
            save_tasks(tasks)
            await broadcast_task_update(task_id, tasks[task_id])

            with _stage(task_id, "audio_normalize"):
                audio_path = await video_processor.normalize_local_media_to_wav(saved_path, TEMP_DIR)

            tasks[task_id].update({
                "progress": 35,
                "message": "音频准备完成，准备转录...",
            })
            save_tasks(tasks)
            await broadcast_task_update(task_id, tasks[task_id])

            tasks[task_id].update({
                "progress": 40,
                "message": "正在转录音频（Qwen3-ASR）...",
            })
            save_tasks(tasks)
            await broadcast_task_update(task_id, tasks[task_id])

            with _stage(task_id, "transcribe"):
                raw_script = await transcriber.transcribe(audio_path, spk_num=spk_num)

        await _run_post_extract_pipeline(
            task_id=task_id,
            raw_script=raw_script,
            video_title=video_title,
            source_ref=source_ref,
            summary_language=summary_language,
            request_summarizer=request_summarizer,
            dedup_url=None,
            api_key=api_key,
            model_base_url=model_base_url,
            model_id=model_id,
            oral_cleanup=oral_cleanup,
        )

    except Exception as e:
        logger.exception(f"任务 {task_id} 处理失败: {e}")
        if task_id in active_tasks:
            del active_tasks[task_id]
        tasks[task_id].update({
            "status": "error",
            "error": str(e),
            "message": f"处理失败: {str(e)}",
        })
        save_tasks(tasks)
        await broadcast_task_update(task_id, tasks[task_id])


class SpeakerRenameRequest(BaseModel):
    """说话人标签 → 真名 映射；值为空表示保留/还原为「说话人X：」。"""
    speakers: dict = Field(default_factory=dict)


@app.post("/api/task/{task_id}/speakers")
async def rename_task_speakers(task_id: str, req: SpeakerRenameRequest):
    """把已完成任务文稿中的「说话人A：」等标签替换为用户填写的真名。"""
    task = tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    script = task.get("script") or ""
    if not script:
        raise HTTPException(status_code=400, detail="任务没有转录文稿")

    new_script, applied = rename_speaker_labels(
        script, req.speakers, previous=task.get("speaker_names")
    )
    task["script"] = new_script
    task["speaker_names"] = applied

    script_path = task.get("script_path")
    if script_path:
        try:
            async with aiofiles.open(script_path, "w", encoding="utf-8") as f:
                await f.write(new_script)
        except Exception:
            logger.exception("重写转录文件失败，不影响任务状态")

    save_tasks(tasks)
    await broadcast_task_update(task_id, task)
    logger.info(f"任务 {task_id} 说话人命名已应用: {applied}")
    return task


@app.get("/api/task-status/{task_id}")
async def get_task_status(task_id: str):
    """
    获取任务状态
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return tasks[task_id]

@app.get("/api/task-stream/{task_id}")
async def task_stream(task_id: str):
    """
    SSE实时任务状态流
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    async def event_generator():
        # 创建任务专用的队列
        queue = asyncio.Queue()
        
        # 将队列添加到连接列表
        if task_id not in sse_connections:
            sse_connections[task_id] = []
        sse_connections[task_id].append(queue)
        
        try:
            # 立即发送当前状态
            current_task = tasks.get(task_id, {})
            yield f"data: {json.dumps(current_task, ensure_ascii=False)}\n\n"
            
            # 持续监听状态更新
            while True:
                try:
                    # 等待状态更新，超时时间30秒发送心跳
                    data = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"data: {data}\n\n"
                    
                    # 如果任务完成、失败或取消，结束流
                    task_data = json.loads(data)
                    if task_data.get("status") in ["completed", "error", "cancelled"]:
                        break
                        
                except asyncio.TimeoutError:
                    # 发送心跳保持连接
                    yield f"data: {json.dumps({'type': 'heartbeat'}, ensure_ascii=False)}\n\n"
                    
        except asyncio.CancelledError:
            logger.info(f"SSE连接被取消: {task_id}")
        except Exception as e:
            logger.error(f"SSE流异常: {e}")
        finally:
            # 清理连接
            if task_id in sse_connections and queue in sse_connections[task_id]:
                sse_connections[task_id].remove(queue)
                if not sse_connections[task_id]:
                    del sse_connections[task_id]
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )

@app.get("/api/tasks")
async def list_tasks():
    """轻量任务列表：不含文稿/摘要正文与文件路径，供前端轮询分组。"""
    result = []
    for tid, t in tasks.items():
        status = t.get("status", "")
        qs = t.get("queue_status") or ("done" if status in ("completed", "error") else "")
        settings = t.get("settings") or {}
        result.append({
            "id": tid,
            "url": t.get("url", ""),
            "video_title": t.get("video_title", ""),
            "status": status,
            "queue_status": qs,
            "progress": t.get("progress", 0),
            "message": t.get("message", ""),
            "error": t.get("error"),
            "batch_id": t.get("batch_id", ""),
            "summary_language": t.get("summary_language") or settings.get("summary_language", ""),
            "oral_cleanup": bool(settings.get("oral_cleanup", False)),
            "spk_num": settings.get("spk_num", ""),
            "cache_hit": t.get("cache_hit"),
            "completed_at": t.get("completed_at"),
            "total_duration": t.get("total_duration"),
            "timeline": t.get("timeline") or {},
            "has_script": bool(t.get("script")),
            "has_summary": bool(t.get("summary")),
        })
    return {"tasks": result}


@app.post("/api/task/{task_id}/requeue")
async def requeue_task(task_id: str):
    """重新入队：仅允许失败（error）或服务重启中断（interrupted）的任务。"""
    t = tasks.get(task_id)
    if t is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    status = t.get("status", "")
    qs = t.get("queue_status", "")
    if status != "error" and qs != "interrupted":
        raise HTTPException(
            status_code=400,
            detail=f"当前状态不可重新入队（status={status}, queue_status={qs}）",
        )
    t.update({
        "queue_status": "queued",
        "status": "processing",
        "progress": 0,
        "message": "已重新入队，等待处理...",
        "error": None,
    })
    save_tasks(tasks)
    await _task_queue.put(task_id)
    await broadcast_task_update(task_id, t)
    return t


@app.post("/api/task/{task_id}/cancel")
async def cancel_task(task_id: str):
    """取消排队中的任务（仅限 queued 状态，运行中请用 DELETE）。"""
    t = tasks.get(task_id)
    if t is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    if t.get("queue_status") != "queued":
        raise HTTPException(
            status_code=400,
            detail=f"当前状态不可取消（queue_status={t.get('queue_status', '')}）",
        )
    t.update({
        "queue_status": "cancelled",
        "status": "cancelled",
        "message": "已取消",
        "error": None,
    })
    save_tasks(tasks)
    await broadcast_task_update(task_id, t)
    return t


class MergeRequest(BaseModel):
    """合并请求：要合并的任务 ID 列表（至少 1 个）。"""
    task_ids: List[str]


@app.post("/api/batch/merge")
async def merge_tasks(req: MergeRequest):
    """合并多个已完成任务的转录+摘要为单个 Markdown 文档。"""
    if not req.task_ids:
        raise HTTPException(status_code=400, detail="task_ids 不能为空")
    missing = [tid for tid in req.task_ids if tid not in tasks]
    if missing:
        raise HTTPException(status_code=404, detail=f"任务不存在: {missing[0]}")

    merged = []
    failed = []
    for tid in req.task_ids:
        t = tasks[tid]
        if t.get("status") != "completed" or not (t.get("script") or "").strip():
            reason = "任务未完成" if t.get("status") != "completed" else "无转录文稿"
            failed.append({"task_id": tid, "url": t.get("url", ""), "reason": reason})
            continue
        merged.append(t)

    if not merged:
        raise HTTPException(status_code=400, detail="没有可合并的已完成任务")

    from datetime import datetime
    now = datetime.now()
    lines = [
        f"# 批量转录合并文档 · {now.strftime('%Y-%m-%d %H:%M')}",
        "",
        f"共 {len(req.task_ids)} 个任务：成功 {len(merged)} · 未收录 {len(failed)}",
        "",
        "## 目录",
    ]
    for i, t in enumerate(merged, 1):
        title = t.get("video_title") or t.get("url") or f"任务{i}"
        lines.append(f"{i}. {title}")
    lines += ["", "---", ""]
    for i, t in enumerate(merged, 1):
        title = t.get("video_title") or t.get("url") or f"任务{i}"
        lines += [
            f"## {i}. {title}",
            f"> 来源：{t.get('url', '')} · 总耗时 {t.get('total_duration', 0)}s · 完成于 {t.get('completed_at', '')}",
            "",
            "### 转录文本",
            "",
            _strip_first_heading(t.get("script", "")),
            "",
            "### 智能摘要",
            "",
            (t.get("summary") or "").strip(),
            "",
            "---",
            "",
        ]
    if failed:
        for f in failed:
            lines.append(f"> 未收录：{f['task_id']}（{f['url']}）：{f['reason']}")
        lines.append("")

    filename = f"merged_transcripts_{now.strftime('%Y%m%d_%H%M%S')}.md"
    (TEMP_DIR / filename).write_text("\n".join(lines), encoding="utf-8")
    return {
        "filename": filename,
        "download_url": f"/api/download/{filename}",
        "merged": len(merged),
        "failed": failed,
    }

@app.get("/api/download/{filename}")
async def download_file(filename: str):
    """
    直接从temp目录下载文件（简化方案）
    """
    try:
        # 检查文件扩展名安全性
        if not filename.endswith('.md'):
            raise HTTPException(status_code=400, detail="仅支持下载.md文件")
        
        # 检查文件名格式（防止路径遍历攻击）
        if '..' in filename or '/' in filename or '\\' in filename:
            raise HTTPException(status_code=400, detail="文件名格式无效")
            
        file_path = TEMP_DIR / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="文件不存在")
            
        return FileResponse(
            file_path,
            filename=filename,
            media_type="text/markdown"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"下载文件失败: {e}")
        raise HTTPException(status_code=500, detail=f"下载失败: {str(e)}")


@app.delete("/api/task/{task_id}")
async def delete_task(task_id: str):
    """
    取消并删除任务
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    # 如果任务还在运行，先取消它
    if task_id in active_tasks:
        task = active_tasks[task_id]
        if not task.done():
            task.cancel()
            logger.info(f"任务 {task_id} 已被取消")
        del active_tasks[task_id]
    
    # 从处理URL列表中移除
    task_url = tasks[task_id].get("url")
    if task_url:
        processing_urls.discard(task_url)
    
    # 删除任务记录
    del tasks[task_id]
    return {"message": "任务已取消并删除"}

@app.get("/api/tasks/active")
async def get_active_tasks():
    """
    获取当前活跃任务列表（用于调试）
    """
    active_count = len(active_tasks)
    processing_count = len(processing_urls)
    return {
        "active_tasks": active_count,
        "processing_urls": processing_count,
        "task_ids": list(active_tasks.keys())
    }

@app.on_event("startup")
async def _startup_batch_queue():
    """启动时：中断遗留任务（排队中 + 重启前残留的 processing 僵尸）+ 启动串行队列 worker。"""
    global _queue_worker_task
    changed = False
    for t in tasks.values():
        # 启动时刻不存在任何"还在跑"的任务：status==processing 的一律是遗留
        # （queued 队列项 / running 崩溃残留 / 旧版本直跑僵尸）——统一标记中断，可重新入队
        if t.get("status") == "processing":
            t["queue_status"] = "interrupted"
            t["message"] = "服务重启，任务中断，请在任务列表中重新入队"
            changed = True
    if changed:
        save_tasks(tasks)
    _queue_worker_task = asyncio.create_task(_queue_worker())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 9000)))
