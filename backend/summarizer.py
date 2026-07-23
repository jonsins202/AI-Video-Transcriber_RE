import os
import openai
import logging
import asyncio
from collections import Counter
import json
import re
import random
from typing import Optional, Callable, Awaitable

from llm_sanitize import strip_llm_artifacts
from lossless_optimizer import (
    SegmentBatch,
    assemble_segments,
    build_batches,
    evaluate_batch_response,
    fallback_batch,
    split_source_segments,
)

logger = logging.getLogger(__name__)

class Summarizer:
    """文本总结器，使用OpenAI API生成多语言摘要"""
    
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        """
        初始化总结器。

        优先级：参数 > 环境变量。
        model 指定时会同时作为 fast_model 和 advanced_model 使用。
        """
        effective_key = api_key or os.getenv("OPENAI_API_KEY")
        effective_url = base_url or os.getenv("OPENAI_BASE_URL")

        if not effective_key:
            logger.warning("未设置OPENAI_API_KEY环境变量，将无法使用摘要功能")

        if effective_key:
            kwargs = {"api_key": effective_key}
            if effective_url:
                kwargs["base_url"] = effective_url
                logger.info(f"OpenAI客户端已初始化，base_url={effective_url}")
            else:
                logger.info("OpenAI客户端已初始化，使用默认端点")
            self.client = openai.OpenAI(**kwargs)
        else:
            self.client = None

        # 允许前端指定模型，覆盖硬编码的 gpt-3.5-turbo / gpt-4o
        self.fast_model     = model or "gpt-3.5-turbo"
        self.advanced_model = model or "gpt-4o"
        self.last_optimization_audit = None
        
        # 支持的语言映射
        self.language_map = {
            "en": "English",
            "zh": "中文（简体）",
            "es": "Español",
            "fr": "Français", 
            "de": "Deutsch",
            "it": "Italiano",
            "pt": "Português",
            "ru": "Русский",
            "ja": "日本語",
            "ko": "한국어",
            "ar": "العربية"
        }
    
    async def optimize_transcript(self, raw_transcript: str) -> str:
        """
        优化转录文本：修正错别字，按含义分段
        支持长文本自动分块处理
        
        Args:
            raw_transcript: 原始转录文本
            
        Returns:
            优化后的转录文本（Markdown格式）
        """
        body = self.extract_transcript_body(raw_transcript)
        return await self.optimize_transcript_body(body)

    async def optimize_transcript_body(
        self,
        body: str,
        transcript_language: Optional[str] = None,
        on_progress: Optional[Callable[[int, int], Awaitable[None]]] = None,
    ) -> str:
        """严格无损优化纯正文；任何未验证段均回退原文。"""
        if not body:
            self.last_optimization_audit = {
                "model": self.fast_model,
                "source_length": 0,
                "segment_count": 0,
                "batch_count": 0,
                "status_counts": {},
                "segments": [],
            }
            return body
        language = transcript_language or self._detect_transcript_language(body)
        try:
            return await self._optimize_lossless_body(body, language, on_progress=on_progress)
        except Exception:
            logger.exception("无损优化管线异常，返回完整原文")
            self.last_optimization_audit = {
                "model": self.fast_model,
                "source_length": len(body),
                "segment_count": 0,
                "batch_count": 0,
                "status_counts": {"fallback_pipeline": 1},
                "segments": [],
            }
            return body

    def get_last_optimization_audit(self) -> Optional[dict]:
        return self.last_optimization_audit

    @staticmethod
    def _extract_json_object(content: str) -> dict:
        text = (content or "").strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else ""
            if text.rstrip().endswith("```"):
                text = text.rstrip()[:-3]
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start:
            raise ValueError("response does not contain a JSON object")
        payload = json.loads(text[start:end + 1])
        if not isinstance(payload, dict):
            raise ValueError("response JSON must be an object")
        return payload

    async def _call_lossless_batch(self, batch: SegmentBatch, language: str) -> dict:
        request_payload = {
            "left_context": batch.left_context,
            "segments": [
                {"id": segment.id, "text": segment.source_text}
                for segment in batch.segments
            ],
            "right_context": batch.right_context,
        }
        system_prompt = (
            "你是严格无损的转录校对器。left_context/right_context 仅供理解，禁止输出。"
            "segments 中每个 id 必须恰好返回一次且不可改名、遗漏、重复或重排。"
            "只允许添加标点和分段、修正少量错别字/同音字/专有名词；"
            "禁止删除、概括、合并、扩写或改变任何口语、重复、寒暄、引用和举例。"
            "只输出一个 JSON 对象：{\"segments\":[{\"id\":\"...\",\"text\":\"...\"}]}。"
            f"正文语言：{language or 'unknown'}。"
        )
        max_retries = 3
        base_delay = 2.0
        cap_delay = 20.0
        for attempt in range(max_retries + 1):
            try:
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=self.fast_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": json.dumps(request_payload, ensure_ascii=False),
                        },
                    ],
                    max_tokens=5000,
                    temperature=0.0,
                )
                if not getattr(response, "choices", None):
                    return {"ok": False, "reason": "missing_choice"}
                choice = response.choices[0]
                finish_reason = getattr(choice, "finish_reason", None)
                if finish_reason != "stop":
                    return {
                        "ok": False,
                        "reason": "finish_reason",
                        "finish_reason": finish_reason,
                    }
                content = getattr(getattr(choice, "message", None), "content", "") or ""
                if not content.strip():
                    return {"ok": False, "reason": "empty_response"}
                return {
                    "ok": True,
                    "payload": self._extract_json_object(content),
                    "finish_reason": finish_reason,
                }
            except openai.RateLimitError as exc:
                if attempt >= max_retries:
                    return {"ok": False, "reason": "rate_limited", "error": str(exc)}
                delay = self._compute_backoff(exc, attempt, base_delay, cap_delay)
                logger.warning(
                    f"无损优化批次被限流(429)，{delay:.1f}s 后重试 "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                await asyncio.sleep(delay)
                continue
            except Exception as exc:
                return {"ok": False, "reason": "api_or_json", "error": str(exc)}
        return {"ok": False, "reason": "rate_limited", "error": "exhausted"}

    @staticmethod
    def _compute_backoff(
        exc: Exception, attempt: int, base_delay: float, cap_delay: float
    ) -> float:
        """429 退避时长：优先 honor Retry-After，否则指数退避 + full jitter。"""
        retry_after = None
        resp = getattr(exc, "response", None)
        headers = getattr(resp, "headers", None)
        if headers is not None:
            raw = headers.get("retry-after")
            if raw is not None:
                try:
                    retry_after = float(raw)
                except (TypeError, ValueError):
                    retry_after = None  # 忽略 HTTP-date 形式
        computed = min(cap_delay, base_delay * (2 ** attempt))
        # retry_after 也用 cap 兜底，防病态巨大值（如 3600）让单批槽位阻塞过久
        delay = max(computed, min(retry_after or 0.0, cap_delay))
        return delay + random.uniform(0.0, 1.0)  # full jitter，避免惊群

    async def _process_lossless_batch(
        self, batch: SegmentBatch, language: str, allow_split: bool = True
    ) -> dict:
        result = await self._call_lossless_batch(batch, language)
        if (
            not result.get("ok")
            and result.get("finish_reason") == "length"
            and allow_split
            and len(batch.segments) > 1
        ):
            midpoint = len(batch.segments) // 2
            left = SegmentBatch(tuple(batch.segments[:midpoint]), batch.left_context, "")
            right = SegmentBatch(tuple(batch.segments[midpoint:]), "", batch.right_context)
            decisions = await self._process_lossless_batch(left, language, allow_split=False)
            decisions.update(
                await self._process_lossless_batch(right, language, allow_split=False)
            )
            return decisions
        if not result.get("ok"):
            fail_reason = result.get("reason")
            # rate_limited 已在 _call_lossless_batch 内退避重试过，这里跳过立即重试避免双重退避
            if fail_reason != "rate_limited":
                retry = await self._call_lossless_batch(batch, language)
                if retry.get("ok"):
                    result = retry
                else:
                    fail_reason = retry.get("reason") or fail_reason
            if not result.get("ok"):
                reason = fail_reason or result.get("reason") or "unknown"
                status = (
                    "fallback_finish_reason"
                    if reason == "finish_reason"
                    else "fallback_api"
                )
                return fallback_batch(batch.segments, status, reason)
        return evaluate_batch_response(batch.segments, result["payload"])

    async def _optimize_lossless_body(
        self,
        body: str,
        language: str,
        on_progress: Optional[Callable[[int, int], Awaitable[None]]] = None,
    ) -> str:
        segments = split_source_segments(body)
        batches = build_batches(body, segments)
        decisions = {}
        if not self.client:
            logger.warning("OpenAI API不可用，无损优化全部回退原文")
            for batch in batches:
                decisions.update(
                    fallback_batch(batch.segments, "fallback_api", "client_unavailable")
                )
        else:
            try:
                concurrency = max(1, min(int(os.getenv("LOSSLESS_CONCURRENCY", "32")), 32))
            except (ValueError, TypeError):
                concurrency = 32
            sem = asyncio.Semaphore(concurrency)
            total = len(batches)
            completed = 0
            logger.info(
                f"无损优化: {len(segments)} 段, {total} 批, 并发 {concurrency}"
            )

            async def _run_one(idx: int, batch: SegmentBatch) -> dict:
                nonlocal completed
                # 关键：信号量包顶层 _process_lossless_batch，不包 _call_lossless_batch，
                # 否则递归二分会重复占槽位、压垮有效并发。
                async with sem:
                    try:
                        result = await self._process_lossless_batch(batch, language)
                    except Exception:
                        logger.exception(f"无损优化批次 {idx}/{total} 异常，回退原文")
                        result = fallback_batch(batch.segments, "fallback_api", "exception")
                completed += 1
                logger.info(f"无损优化批次 {idx}/{total} 完成")
                if on_progress is not None:
                    try:
                        await on_progress(completed, total)
                    except Exception:
                        logger.debug("无损优化进度回调异常，忽略")
                return result

            batch_results = list(await asyncio.gather(
                *[_run_one(i, b) for i, b in enumerate(batches, 1)]
            ))
            for r in batch_results:
                decisions.update(r)

        status_counts = Counter(decision.status for decision in decisions.values())
        self.last_optimization_audit = {
            "model": self.fast_model,
            "source_length": len(body),
            "segment_count": len(segments),
            "batch_count": len(batches),
            "status_counts": dict(status_counts),
            "segments": [
                {
                    "id": segment.id,
                    "start": segment.start,
                    "end": segment.end,
                    "source_hash": segment.source_hash,
                    "result_hash": decisions[segment.id].result_hash,
                    "status": decisions[segment.id].status,
                    "reason": decisions[segment.id].reason,
                    "replaced": decisions[segment.id].replaced,
                    "inserted": decisions[segment.id].inserted,
                    "deleted": decisions[segment.id].deleted,
                }
                for segment in segments
            ],
        }
        return assemble_segments(segments, decisions)

    def _estimate_tokens(self, text: str) -> int:
        """
        改进的token数量估算算法
        更保守的估算，考虑系统prompt和格式化开销
        """
        # 更保守的估算：考虑实际使用中的token膨胀
        chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        english_words = len([word for word in text.split() if word.isascii() and word.isalpha()])
        
        # 计算基础tokens
        base_tokens = chinese_chars * 1.5 + english_words * 1.3
        
        # 考虑markdown格式、时间戳等开销（约30%额外开销）
        format_overhead = len(text) * 0.15
        
        # 考虑系统prompt开销（约2000-3000 tokens）
        system_prompt_overhead = 2500
        
        total_estimated = int(base_tokens + format_overhead + system_prompt_overhead)
        
        return total_estimated

    # ===== JS openaiService.js 移植：分块/上下文/去重/格式化 =====

    def _ensure_markdown_paragraphs(self, text: str) -> str:
        """确保Markdown段落空行、标题后空行、压缩多余空行。"""
        if not text:
            return text
        formatted = text.replace("\r\n", "\n")
        import re
        # 标题后加空行
        formatted = re.sub(r"(^#{1,6}\s+.*)\n([^\n#])", r"\1\n\n\2", formatted, flags=re.M)
        # 压缩≥3个换行为2个
        formatted = re.sub(r"\n{3,}", "\n\n", formatted)
        # 去首尾空行
        formatted = re.sub(r"^\n+", "", formatted)
        formatted = re.sub(r"\n+$", "", formatted)
        return formatted

    async def _format_single_chunk(self, chunk_text: str, transcript_language: str = 'zh',
                                   prev_context: Optional[str] = None) -> str:
        """单块优化（修正+格式化），遵循4000 tokens 限制。

        prev_context: 长文本分块时传入的"上一块结尾"，仅作为衔接上下文喂给 LLM，
            放在独立于正文的【上一段结尾】参考区并明确禁止复述；取代旧实现把
            [上文续：…] 拼进"原始转录文本"导致的标记泄漏与上下文重复。
        """
        # 构建与JS版一致的系统/用户提示
        if transcript_language == 'zh':
            context_block = (
                "<previous_context>\n"
                f"{prev_context}\n"
                "</previous_context>\n\n"
            ) if prev_context else ""
            prompt = (
                "请对以下音频转录文本进行智能优化和格式化，要求：\n\n"
                "**内容优化（正确性优先）：**\n"
                "1. 错误修正（转录错误/错别字/同音字/专有名词）\n"
                "2. 适度改善语法，补全不完整句子，保持原意和语言不变\n"
                "3. 口语处理：保留自然口语与重复表达，不要删减内容，仅添加必要标点\n"
                "4. **绝对不要改变人称代词（I/我、you/你等）和说话者视角**\n"
                "5. **严禁删减任何句子**：开场白、主持人介绍、寒暄、引用、举例都属正文，必须逐句完整保留，禁止跳过、概括或合并删除。只修错别字+加标点，绝不删内容\n\n"
                "**分段规则：**\n"
                "- 按主题和逻辑含义分段，每段包含1-8个相关句子\n"
                "- 单段长度不超过400字符\n"
                "- 避免过多的短段落，合并相关内容\n\n"
                "**格式要求：**Markdown 段落，段落间空行\n\n"
                f"{context_block}原始转录文本：\n{chunk_text}"
            )
            system_prompt = (
                "你是专业的音频转录文本优化助手，修正错误、改善通顺度和排版格式，"
                "必须保持原意，不得删减口语/重复/细节；仅移除时间戳或元信息。"
                "绝对不要改变人称代词或说话者视角。这可能是访谈对话，访谈者用'you'，被访者用'I/we'。"
                "严禁删减任何句子（尤其开场白、主持人介绍、寒暄），必须逐句保留全部内容。"
                "输入中的 <previous_context> 标签是上一段结尾，仅供理解上下文衔接，严禁将其任何内容复述或改写进你的输出。"
            )
        else:
            context_block = (
                "<previous_context>\n"
                f"{prev_context}\n"
                "</previous_context>\n\n"
            ) if prev_context else ""
            prompt = (
                "Please intelligently optimize and format the following audio transcript text:\n\n"
                "Content Optimization (Accuracy First):\n"
                "1. Error Correction (typos, homophones, proper nouns)\n"
                "2. Moderate grammar improvement, complete incomplete sentences, keep original language/meaning\n"
                "3. Speech processing: keep natural fillers and repetitions, do NOT remove content; only add punctuation if needed\n"
                "4. **NEVER change pronouns (I, you, he, she, etc.) or speaker perspective**\n\n"
                "Segmentation Rules: Group 1-8 related sentences per paragraph by topic/logic; paragraph length NOT exceed 400 characters; avoid too many short paragraphs\n\n"
                "Format: Markdown paragraphs with blank lines between paragraphs\n\n"
                f"{context_block}Original transcript text:\n{chunk_text}"
            )
            system_prompt = (
                "You are a professional transcript formatting assistant. Fix errors and improve fluency "
                "without changing meaning or removing any content; only timestamps/meta may be removed; keep Markdown paragraphs with blank lines. "
                "NEVER change pronouns or speaker perspective. This may be an interview: interviewer uses 'you', interviewee uses 'I/we'."
                " The <previous_context> tag in the input is the end of the previous segment, for continuity context ONLY; NEVER reproduce or paraphrase any of it in your output."
            )

        try:
            import asyncio
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.fast_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4000,  # 对齐JS：优化/格式化阶段最大tokens≈4000
                temperature=0.1
            )
            optimized_text = strip_llm_artifacts(response.choices[0].message.content or "")
            # 移除诸如 "# Transcript" / "## Transcript" 等标题
            optimized_text = self._remove_transcript_heading(optimized_text)
            enforced = self._enforce_paragraph_max_chars(optimized_text.strip(), max_chars=400)
            return self._ensure_markdown_paragraphs(enforced)
        except Exception as e:
            logger.error(f"单块文本优化失败: {e}")
            return self._apply_basic_formatting(chunk_text)

    def _smart_split_long_chunk(self, text: str, max_chars_per_chunk: int) -> list:
        """在句子/空格边界处安全切分超长文本。"""
        chunks = []
        pos = 0
        while pos < len(text):
            end = min(pos + max_chars_per_chunk, len(text))
            if end < len(text):
                # 优先句子边界
                sentence_endings = ['。', '！', '？', '.', '!', '?']
                best = -1
                for ch in sentence_endings:
                    idx = text.rfind(ch, pos, end)
                    if idx > best:
                        best = idx
                if best > pos + int(max_chars_per_chunk * 0.7):
                    end = best + 1
                else:
                    # 次选：空格边界
                    space_idx = text.rfind(' ', pos, end)
                    if space_idx > pos + int(max_chars_per_chunk * 0.8):
                        end = space_idx
            chunks.append(text[pos:end].strip())
            pos = end
        return [c for c in chunks if c]

    def _find_safe_cut_point(self, text: str) -> int:
        """找到安全的切割点（段落>句子>短语）。"""
        import re
        # 段落
        p = text.rfind("\n\n")
        if p > 0:
            return p + 2
        # 句子
        last_sentence_end = -1
        for m in re.finditer(r"[。！？\.!?]\s*", text):
            last_sentence_end = m.end()
        if last_sentence_end > 20:
            return last_sentence_end
        # 短语
        last_phrase_end = -1
        for m in re.finditer(r"[，；,;]\s*", text):
            last_phrase_end = m.end()
        if last_phrase_end > 20:
            return last_phrase_end
        return len(text)

    def _find_overlap_between_texts(self, text1: str, text2: str) -> str:
        """检测相邻两段的重叠内容，用于去重。"""
        max_len = min(len(text1), len(text2))
        # 逐步从长到短尝试
        for length in range(max_len, 19, -1):
            suffix = text1[-length:]
            prefix = text2[:length]
            if suffix == prefix:
                cut = self._find_safe_cut_point(prefix)
                if cut > 20:
                    return prefix[:cut]
                return suffix
        return ""

    def _strip_context_echo(self, text: str, prev_context: str,
                            ratio_threshold: float = 0.65) -> str:
        """后置兜底：若 text 开头复述了 prev_context（LLM 不听话把上一段结尾写进输出），截掉重复段。

        邻接去重 _find_overlap_between_texts 只兜"逐字滚动重叠"，对"LLM 把整段上下文从头
        复述"无效——复述会带改写、且从头讲而非尾首相接（task:37ee27 真实案例 ratio=0.752）。
        本方法用相似度兜底：逐字前缀复述直接截；否则 text 开头窗口与 prev_context 相似度
        超阈值且命中过半 → 截到最远命中位置。阈值从严（0.65 + 命中过半 + 匹配块≥4字），
        宁可漏过也不误删正文；触发时 log warning 便于察觉误删。
        已知限制：对高度同质/重复结构的文本（罕见；真实转录内容多样、风险低），
        可能误判为复述并截掉开头几句；若误删频繁可调高 ratio_threshold。
        """
        if not prev_context or not text:
            return text
        import difflib
        text_l = text.lstrip()
        # 1) 逐字前缀复述（最稳）
        if text_l.startswith(prev_context):
            logger.info(f"检测到逐字复述上文，已截掉 {len(prev_context)} 字")
            return text_l[len(prev_context):].lstrip()
        # 2) 相似度复述（带改写）
        window = text_l[:len(prev_context) + 60]
        sm = difflib.SequenceMatcher(None, prev_context, window)
        if sm.ratio() >= ratio_threshold:
            farthest = 0
            for _a, b, size in sm.get_matching_blocks():
                if size >= 4:
                    farthest = max(farthest, b + size)
            if farthest > len(prev_context) * 0.5:
                logger.warning(
                    f"检测到疑似复述上文（相似度 {sm.ratio():.2f}），截掉开头 {farthest} 字；"
                    f"若误删正文请检查分块或降低 ratio_threshold"
                )
                return text_l[farthest:].lstrip()
        return text

    def _maybe_overall_fallback(self, merged: str, raw_transcript: str,
                                threshold: float = 0.60) -> str:
        """整体长度兜底：优化+去重后若整体明显短于原始（<threshold），整体用基础格式化兜底。

        单块 70% 兜底能拦住"某块输出过短"（如跳段、过度精简），但拦不住单块兜底自身失效、
        去重误删等边缘情况。本方法作为最后防线：不管什么原因，整体丢 >(1-threshold) 就整体回退，
        宁可不优化也不丢段（task:1128f3 即严重丢失案例：块3跳段丢 ~4000 字核心论述）。
        正常优化长度基本持平或略增（修错别字+分段），threshold=0.60 不会误触发。
        """
        if not merged.strip():
            logger.warning("优化后整体为空，整体使用原文兜底")
            return self._apply_basic_formatting(raw_transcript)
        if len(merged.strip()) < len(raw_transcript) * threshold:
            ratio_pct = len(merged.strip()) * 100 // max(1, len(raw_transcript))
            logger.warning(
                f"优化后整体过短（{len(merged.strip())} vs 原始 {len(raw_transcript)}，"
                f"{ratio_pct}%），疑丢大段内容，整体兜底"
            )
            return self._apply_basic_formatting(raw_transcript)
        return merged

    def _apply_basic_formatting(self, text: str) -> str:
        """当AI失败时的回退：按句子拼段，段落≤250字符，双换行分隔。"""
        if not text or not text.strip():
            return text
        import re
        parts = re.split(r"([。！？\.!?]+\s*)", text)
        sentences = []
        current = ""
        for i, part in enumerate(parts):
            if i % 2 == 0:
                current += part
            else:
                current += part
                if current.strip():
                    sentences.append(current.strip())
                    current = ""
        if current.strip():
            sentences.append(current.strip())
        paras = []
        cur = ""
        sentence_count = 0
        for s in sentences:
            candidate = (cur + " " + s).strip() if cur else s
            sentence_count += 1
            # 改进的分段逻辑：考虑句子数量和长度
            should_break = False
            if len(candidate) > 400 and cur:  # 段落过长
                should_break = True
            elif len(candidate) > 200 and sentence_count >= 3:  # 中等长度且句子数足够
                should_break = True
            elif sentence_count >= 6:  # 句子数过多
                should_break = True
            
            if should_break:
                paras.append(cur.strip())
                cur = s
                sentence_count = 1
            else:
                cur = candidate
        if cur.strip():
            paras.append(cur.strip())
        return self._ensure_markdown_paragraphs("\n\n".join(paras))

    async def _format_long_transcript_in_chunks(self, raw_transcript: str, transcript_language: str, max_chars_per_chunk: int) -> str:
        """智能分块+上下文+去重 合成优化文本（JS策略移植）。"""
        import re
        # 先按句子切分，组装不超过max_chars_per_chunk的块
        parts = re.split(r"([。！？\.!?]+\s*)", raw_transcript)
        sentences = []
        buf = ""
        for i, part in enumerate(parts):
            if i % 2 == 0:
                buf += part
            else:
                buf += part
                if buf.strip():
                    sentences.append(buf.strip())
                    buf = ""
        if buf.strip():
            sentences.append(buf.strip())

        chunks = []
        cur = ""
        for s in sentences:
            candidate = (cur + " " + s).strip() if cur else s
            if len(candidate) > max_chars_per_chunk and cur:
                chunks.append(cur.strip())
                cur = s
            else:
                cur = candidate
        if cur.strip():
            chunks.append(cur.strip())

        # 对仍然过长的块二次安全切分
        final_chunks = []
        for c in chunks:
            if len(c) <= max_chars_per_chunk:
                final_chunks.append(c)
            else:
                final_chunks.extend(self._smart_split_long_chunk(c, max_chars_per_chunk))

        logger.info(f"文本分为 {len(final_chunks)} 块处理")

        # 串行处理。注：曾改为并发（gather+Semaphore）提速，但 DeepSeek 在并发压力下
        # 偶发返回不完整/过度精简的输出，导致开场等大段内容丢失（实测丢 ~25k 字符/28%）。
        # 质量优先，回归串行（一次一个请求，LLM 输出稳定）。
        optimized = []
        for i, c in enumerate(final_chunks):
            # 上一块【优化后输出】的结尾作为衔接上下文（取优化后而非原文，让"喂 LLM 的上下文"
            # 与下方 _strip_context_echo / 邻接去重比较的 prev 是同一份文本，且质量更高）。
            # 用 <previous_context> 标签结构化喂给 LLM，弱模型对标签边界更敏感、不易当正文复述。
            prev_context = optimized[-1][-100:] if optimized else None
            try:
                oc = await self._format_single_chunk(c, transcript_language, prev_context=prev_context)
                # 后置兜底：LLM 偶发把 <previous_context> 内容复述进输出开头。邻接去重只兜逐字
                # 滚动重叠、兜不住"整段从头复述"（task:37ee27 真实案例 ratio=0.752），这里相似度截断。
                if prev_context:
                    oc = self._strip_context_echo(oc, prev_context)
                # 质量校验：LLM 偶发过度精简（把正文当冗余删，如开场介绍）。若优化后明显短于原始（<70%）
                # 疑丢内容 → 用原始块基础格式化兜底（保留全部内容，宁可不优化也不丢段）
                if not oc.strip():
                    logger.warning(f"第 {i+1} 块优化结果为空，使用原始块兜底")
                    oc = self._apply_basic_formatting(c)
                elif len(oc.strip()) < len(c.strip()) * 0.7:
                    logger.warning(f"第 {i+1} 块优化后异常短（{len(oc.strip())} vs 原始 {len(c.strip())}），疑丢内容，用原始块兜底")
                    oc = self._apply_basic_formatting(c)
                optimized.append(oc)
            except Exception as e:
                logger.warning(f"第 {i+1} 块优化失败，使用基础格式化: {e}")
                optimized.append(self._apply_basic_formatting(c))

        # 邻接块去重
        deduped = []
        for i, c in enumerate(optimized):
            cur_txt = c
            if i > 0 and deduped:
                prev = deduped[-1]
                overlap = self._find_overlap_between_texts(prev[-200:], cur_txt[:200])
                if overlap:
                    cur_txt = cur_txt[len(overlap):].lstrip()
                    if not cur_txt:
                        continue
            if cur_txt.strip():
                deduped.append(cur_txt)

        merged = "\n\n".join(deduped)
        merged = self._remove_transcript_heading(merged)
        merged = self._maybe_overall_fallback(merged, raw_transcript)
        enforced = self._enforce_paragraph_max_chars(merged, max_chars=400)
        return self._ensure_markdown_paragraphs(enforced)

    @staticmethod
    def extract_transcript_body(text: str) -> str:
        """从兼容 Markdown 中只移除本项目生成的元信息，保留用户正文。"""
        if not text:
            return text
        kept = []
        generated_titles = {"# Video Transcription", "# 视频转录"}
        metadata_pattern = re.compile(
            r"^\*\*(?:Detected Language|Language Probability|检测语言|语言概率):\*\*"
        )
        content_heading_pattern = re.compile(
            r"^#{1,6}\s*(?:Transcription Content|转录内容)\s*$",
            flags=re.I,
        )
        timestamp_pattern = re.compile(r"^\*\*\[[^\]]+\]\*\*$")
        for line in text.replace("\r\n", "\n").split("\n"):
            stripped = line.strip()
            if stripped in generated_titles:
                continue
            if metadata_pattern.match(stripped):
                continue
            if content_heading_pattern.match(stripped):
                continue
            if timestamp_pattern.match(stripped):
                continue
            kept.append(line)
        return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()

    def _remove_timestamps_and_meta(self, text: str) -> str:
        """兼容旧调用；统一委托给正文提取器。"""
        return self.extract_transcript_body(text)

    def _enforce_paragraph_max_chars(self, text: str, max_chars: int = 400) -> str:
        """按段落拆分并确保每段不超过max_chars，必要时按句子边界拆为多段。"""
        if not text:
            return text
        import re
        paragraphs = [p for p in re.split(r"\n\s*\n", text) if p is not None]
        new_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if len(para) <= max_chars:
                new_paragraphs.append(para)
                continue
            # 句子切分
            parts = re.split(r"([。！？\.!?]+\s*)", para)
            sentences = []
            buf = ""
            for i, part in enumerate(parts):
                if i % 2 == 0:
                    buf += part
                else:
                    buf += part
                    if buf.strip():
                        sentences.append(buf.strip())
                        buf = ""
            if buf.strip():
                sentences.append(buf.strip())
            cur = ""
            for s in sentences:
                candidate = (cur + (" " if cur else "") + s).strip()
                if len(candidate) > max_chars and cur:
                    new_paragraphs.append(cur)
                    cur = s
                else:
                    cur = candidate
            if cur:
                new_paragraphs.append(cur)
        return "\n\n".join([p.strip() for p in new_paragraphs if p is not None])

    def _remove_transcript_heading(self, text: str) -> str:
        """移除开头或段落中的以 Transcript 为标题的行（任意级别#），不改变正文。"""
        if not text:
            return text
        import re
        # 移除形如 '## Transcript'、'# Transcript Text'、'### transcript' 的标题行
        lines = text.split('\n')
        filtered = []
        for line in lines:
            stripped = line.strip()
            if re.match(r"^#{1,6}\s*transcript(\s+text)?\s*$", stripped, flags=re.I):
                continue
            filtered.append(line)
        return '\n'.join(filtered)

    async def summarize(self, transcript: str, target_language: str = "zh", video_title: str = None) -> str:
        """
        生成视频转录的摘要
        
        Args:
            transcript: 转录文本
            target_language: 目标语言代码
            
        Returns:
            摘要文本（Markdown格式）
        """
        try:
            if not self.client:
                logger.warning("OpenAI API不可用，生成备用摘要")
                return self._generate_fallback_summary(transcript, target_language, video_title)
            
            # 估算转录文本长度，决定是否需要分块摘要
            estimated_tokens = self._estimate_tokens(transcript)
            max_summarize_tokens = 4000  # 提高限制，优先使用单文本处理以获得更好的总结质量
            
            if estimated_tokens <= max_summarize_tokens:
                # 短文本直接摘要
                return await self._summarize_single_text(transcript, target_language, video_title)
            else:
                # 长文本分块摘要
                logger.info(f"文本较长({estimated_tokens} tokens)，启用分块摘要")
                return await self._summarize_with_chunks(transcript, target_language, video_title, max_summarize_tokens)
            
        except Exception as e:
            logger.error(f"生成摘要失败: {str(e)}")
            return self._generate_fallback_summary(transcript, target_language, video_title)

    async def _summarize_single_text(self, transcript: str, target_language: str, video_title: str = None) -> str:
        """
        对单个文本进行摘要
        """
        # 获取目标语言名称
        language_name = self.language_map.get(target_language, "中文（简体）")
        
        # 构建英文提示词，适用于所有目标语言
        system_prompt = f"""You are an expert editor. Write a concise EXECUTIVE SUMMARY in {language_name} of the following material.

Hard rules:
- Length: about 180–450 words in {language_name} (use the lower end if the source is short). Never reproduce long verbatim quotes or extended sentence-by-sentence rewrites of the transcript.
- Content: main thesis, 3–7 key takeaways, important conclusions, and critical facts or numbers only. Tight prose; short bullet lists are OK for takeaways.
- Do NOT restate the full transcript, do NOT add preamble ("Here is…"), and do NOT add closings such as offers to revise or "let me know if…" / 客套尾注.
- Markdown: optional `## Key takeaways` then paragraphs; avoid decorative filler headings.

Output ONLY the summary body in {language_name}."""

        user_prompt = f"""Summarize the following content in {language_name}. Follow the system rules strictly (brief executive summary, no meta-commentary):

{transcript}"""

        logger.info(f"正在生成{language_name}摘要...")
        
        # 调用OpenAI API
        response = self.client.chat.completions.create(
            model=self.advanced_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=2200,
            temperature=0.25
        )
        
        summary = strip_llm_artifacts(response.choices[0].message.content or "")

        return self._format_summary_with_meta(summary, target_language, video_title)

    async def _summarize_with_chunks(self, transcript: str, target_language: str, video_title: str, max_tokens: int) -> str:
        """
        分块摘要长文本
        """
        language_name = self.language_map.get(target_language, "中文（简体）")

        # 使用JS策略：按字符进行智能分块（段落>句子）
        chunks = self._smart_chunk_text(transcript, max_chars_per_chunk=4000)
        logger.info(f"分割为 {len(chunks)} 个块进行摘要")
        
        chunk_summaries = []
        
        # 每块生成局部摘要
        # 预构造每块 prompt（块间无依赖：每块只看自己的原文+编号，可全部预先组装）
        prompts = []
        for i, chunk in enumerate(chunks):
            system_prompt = f"""You are a summarization expert. Write a brief section summary in {language_name}.

This is part {i+1} of {len(chunks)} of the full transcript.

Rules:
- About 80–160 words in {language_name}; bullets OK for key points.
- Do not echo the transcript verbatim; capture only new information in this segment.
- No preamble or meta-closings."""
            user_prompt = f"""[Part {i+1}/{len(chunks)}] Summarize in {language_name} (80–160 words, tight prose):

{chunk}

Output content only, no headings like "Summary:"."""
            prompts.append((system_prompt, user_prompt, chunk))

        # 并发调用（semaphore 控制并发数，避免 DeepSeek 速率限制）；每块输入/输出与串行完全一致 → 质量不变
        import asyncio, os
        try:
            concurrency = max(1, int(os.getenv("SUMMARY_CONCURRENCY", "6")))
        except (ValueError, TypeError):
            concurrency = 6
        sem = asyncio.Semaphore(concurrency)
        logger.info(f"并发摘要: {len(chunks)} 块，并发数 {concurrency}")

        async def _summarize_one(idx: int, sp: str, up: str, chunk: str) -> str:
            async with sem:
                logger.info(f"正在摘要第 {idx+1}/{len(chunks)} 块...")
                try:
                    response = await asyncio.to_thread(
                        self.client.chat.completions.create,
                        model=self.advanced_model,
                        messages=[
                            {"role": "system", "content": sp},
                            {"role": "user", "content": up}
                        ],
                        max_tokens=600,
                        temperature=0.25
                    )
                    return strip_llm_artifacts(response.choices[0].message.content or "")
                except Exception as e:
                    logger.error(f"摘要第 {idx+1} 块失败: {e}")
                    return f"第{idx+1}部分内容概述：" + chunk[:200] + "..."

        chunk_summaries = list(await asyncio.gather(
            *[_summarize_one(i, sp, up, c) for i, (sp, up, c) in enumerate(prompts)]
        ))
        
        # 合并所有局部摘要（带编号），如分块较多则分层整合（不引入小标题）
        combined_summaries = "\n\n".join([f"[Part {idx+1}]\n" + s for idx, s in enumerate(chunk_summaries)])

        logger.info("正在整合最终摘要...")
        if len(chunk_summaries) > 10:
            final_summary = await self._integrate_hierarchical_summaries(chunk_summaries, target_language)
        else:
            final_summary = await self._integrate_chunk_summaries(combined_summaries, target_language)

        return self._format_summary_with_meta(final_summary, target_language, video_title)

    def _smart_chunk_text(self, text: str, max_chars_per_chunk: int = 3500) -> list:
        """智能分块（先段落后句子），按字符上限切分。"""
        chunks = []
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        cur = ""
        for p in paragraphs:
            candidate = (cur + "\n\n" + p).strip() if cur else p
            if len(candidate) > max_chars_per_chunk and cur:
                chunks.append(cur.strip())
                cur = p
            else:
                cur = candidate
        if cur.strip():
            chunks.append(cur.strip())

        # 二次按句子切分过长块
        import re
        final_chunks = []
        for c in chunks:
            if len(c) <= max_chars_per_chunk:
                final_chunks.append(c)
            else:
                sentences = [s.strip() for s in re.split(r"[。！？\.!?]+", c) if s.strip()]
                scur = ""
                for s in sentences:
                    candidate = (scur + '。' + s).strip() if scur else s
                    if len(candidate) > max_chars_per_chunk and scur:
                        final_chunks.append(scur.strip())
                        scur = s
                    else:
                        scur = candidate
                if scur.strip():
                    final_chunks.append(scur.strip())
        return final_chunks

    async def _integrate_hierarchical_summaries(
        self, chunk_summaries: list, target_language: str
    ) -> str:
        """Many partial summaries: fold through the same integrator as the <=10 case."""
        combined = "\n\n".join(
            f"[Part {idx + 1}]\n{s}" for idx, s in enumerate(chunk_summaries)
        )
        return await self._integrate_chunk_summaries(combined, target_language)

    async def _integrate_chunk_summaries(self, combined_summaries: str, target_language: str) -> str:
        """
        整合分块摘要为最终连贯摘要
        """
        language_name = self.language_map.get(target_language, "中文（简体）")
        
        try:
            system_prompt = f"""You integrate partial summaries into ONE concise executive summary in {language_name}.

Rules:
- Total length about 280–650 words in {language_name}; remove duplication, do not expand into a transcript-length rewrite.
- Markdown: paragraphs separated by blank lines; optional `## Key takeaways` only if it adds clarity.
- No preamble, no meta-closings (e.g. offers to revise or "let me know")."""

            user_prompt = f"""Merge the following partial summaries into one executive summary in {language_name}:

{combined_summaries}"""

            response = self.client.chat.completions.create(
                model=self.advanced_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2200,
                temperature=0.25
            )

            return strip_llm_artifacts(response.choices[0].message.content or "")
        except Exception as e:
            logger.error(f"整合摘要失败: {e}")
            # 失败时直接合并
            return combined_summaries

    def _format_summary_with_meta(self, summary: str, target_language: str, video_title: str = None) -> str:
        """
        为摘要添加标题和元信息
        """
        language_name = self.language_map.get(target_language, "中文（简体）")
        meta_labels = self._get_summary_labels(target_language)
        
        # 不加任何小标题/免责声明，可保留视频标题作为一级标题
        if video_title:
            prefix = f"# {video_title}\n\n"
        else:
            prefix = ""
        return prefix + summary

    def _generate_fallback_summary(self, transcript: str, target_language: str, video_title: str = None) -> str:
        """
        生成备用摘要（当OpenAI API不可用时）
        
        Args:
            transcript: 转录文本
            video_title: 视频标题
            target_language: 目标语言代码
            
        Returns:
            备用摘要文本
        """
        language_name = self.language_map.get(target_language, "中文（简体）")
        
        # 简单的文本处理，提取关键信息
        lines = transcript.split('\n')
        content_lines = [line for line in lines if line.strip() and not line.startswith('#') and not line.startswith('**')]
        
        # 计算大概的长度
        total_chars = sum(len(line) for line in content_lines)
        
        # 使用目标语言的标签
        meta_labels = self._get_summary_labels(target_language)
        fallback_labels = self._get_fallback_labels(target_language)
        
        # 直接使用视频标题作为主标题  
        title = video_title if video_title else "Summary"
        
        summary = f"""# {title}

**{meta_labels['language_label']}:** {language_name}
**{fallback_labels['notice']}:** {fallback_labels['api_unavailable']}



## {fallback_labels['overview_title']}

**{fallback_labels['content_length']}:** {fallback_labels['about']} {total_chars} {fallback_labels['characters']}
**{fallback_labels['paragraph_count']}:** {len(content_lines)} {fallback_labels['paragraphs']}

## {fallback_labels['main_content']}

{fallback_labels['content_description']}

{fallback_labels['suggestions_intro']}

1. {fallback_labels['suggestion_1']}
2. {fallback_labels['suggestion_2']}
3. {fallback_labels['suggestion_3']}

## {fallback_labels['recommendations']}

- {fallback_labels['recommendation_1']}
- {fallback_labels['recommendation_2']}


<br/>

<p style="color: #888; font-style: italic; text-align: center; margin-top: 16px;"><em>{fallback_labels['fallback_disclaimer']}</em></p>"""
        
        return summary
    
    def get_supported_languages(self) -> dict:
        """
        获取支持的语言列表
        
        Returns:
            语言代码到语言名称的映射
        """
        return self.language_map.copy()
    
    def _detect_transcript_language(self, transcript: str) -> str:
        """
        检测转录文本的主要语言
        
        Args:
            transcript: 转录文本
            
        Returns:
            检测到的语言代码
        """
        # 简单的语言检测逻辑：查找转录文本中的语言标记
        if "**检测语言:**" in transcript:
            # 从Whisper转录中提取检测到的语言
            lines = transcript.split('\n')
            for line in lines:
                if "**检测语言:**" in line:
                    # 提取语言代码，例如: "**检测语言:** en"
                    lang = line.split(":")[-1].strip()
                    return lang
        
        # 如果没有找到语言标记，使用简单的字符检测
        # 计算英文字符、中文字符等的比例
        total_chars = len(transcript)
        if total_chars == 0:
            return "en"  # 默认英文
            
        # 统计中文字符
        chinese_chars = sum(1 for char in transcript if '\u4e00' <= char <= '\u9fff')
        chinese_ratio = chinese_chars / total_chars
        
        # 统计英文字母
        english_chars = sum(1 for char in transcript if char.isascii() and char.isalpha())
        english_ratio = english_chars / total_chars
        
        # 根据比例判断
        if chinese_ratio > 0.3:
            return "zh"
        elif english_ratio > 0.3:
            return "en"
        else:
            return "en"  # 默认英文
    

    def _get_summary_labels(self, lang_code: str) -> dict:
        """
        获取摘要页面的多语言标签
        
        Args:
            lang_code: 语言代码
            
        Returns:
            标签字典
        """
        labels = {
            "en": {
                "language_label": "Summary Language",
                "disclaimer": "This summary is automatically generated by AI for reference only"
            },
            "zh": {
                "language_label": "摘要语言",
                "disclaimer": "本摘要由AI自动生成，仅供参考"
            },
            "ja": {
                "language_label": "要約言語",
                "disclaimer": "この要約はAIによって自動生成されており、参考用です"
            },
            "ko": {
                "language_label": "요약 언어",
                "disclaimer": "이 요약은 AI에 의해 자동 생성되었으며 참고용입니다"
            },
            "es": {
                "language_label": "Idioma del Resumen",
                "disclaimer": "Este resumen es generado automáticamente por IA, solo para referencia"
            },
            "fr": {
                "language_label": "Langue du Résumé",
                "disclaimer": "Ce résumé est généré automatiquement par IA, à titre de référence uniquement"
            },
            "de": {
                "language_label": "Zusammenfassungssprache",
                "disclaimer": "Diese Zusammenfassung wird automatisch von KI generiert, nur zur Referenz"
            },
            "it": {
                "language_label": "Lingua del Riassunto",
                "disclaimer": "Questo riassunto è generato automaticamente dall'IA, solo per riferimento"
            },
            "pt": {
                "language_label": "Idioma do Resumo",
                "disclaimer": "Este resumo é gerado automaticamente por IA, apenas para referência"
            },
            "ru": {
                "language_label": "Язык резюме",
                "disclaimer": "Это резюме автоматически генерируется ИИ, только для справки"
            },
            "ar": {
                "language_label": "لغة الملخص",
                "disclaimer": "هذا الملخص تم إنشاؤه تلقائياً بواسطة الذكاء الاصطناعي، للمرجع فقط"
            }
        }
        return labels.get(lang_code, labels["en"])
    
    def _get_fallback_labels(self, lang_code: str) -> dict:
        """
        获取备用摘要的多语言标签
        
        Args:
            lang_code: 语言代码
            
        Returns:
            标签字典
        """
        labels = {
            "en": {
                "notice": "Notice",
                "api_unavailable": "OpenAI API is unavailable, this is a simplified summary",
                "overview_title": "Transcript Overview",
                "content_length": "Content Length",
                "about": "About",
                "characters": "characters",
                "paragraph_count": "Paragraph Count",
                "paragraphs": "paragraphs",
                "main_content": "Main Content",
                "content_description": "The transcript contains complete video speech content. Since AI summary cannot be generated currently, we recommend:",
                "suggestions_intro": "For detailed information, we suggest you:",
                "suggestion_1": "Review the complete transcript text for detailed information",
                "suggestion_2": "Focus on important paragraphs marked with timestamps",
                "suggestion_3": "Manually extract key points and takeaways",
                "recommendations": "Recommendations",
                "recommendation_1": "Configure OpenAI API key for better summary functionality",
                "recommendation_2": "Or use other AI services for text summarization",
                "fallback_disclaimer": "This is an automatically generated fallback summary"
            },
            "zh": {
                "notice": "注意",
                "api_unavailable": "由于OpenAI API不可用，这是一个简化的摘要",
                "overview_title": "转录概览",
                "content_length": "内容长度",
                "about": "约",
                "characters": "字符",
                "paragraph_count": "段落数量",
                "paragraphs": "段",
                "main_content": "主要内容",
                "content_description": "转录文本包含了完整的视频语音内容。由于当前无法生成智能摘要，建议您：",
                "suggestions_intro": "为获取详细信息，建议您：",
                "suggestion_1": "查看完整的转录文本以获取详细信息",
                "suggestion_2": "关注时间戳标记的重要段落",
                "suggestion_3": "手动提取关键观点和要点",
                "recommendations": "建议",
                "recommendation_1": "配置OpenAI API密钥以获得更好的摘要功能",
                "recommendation_2": "或者使用其他AI服务进行文本总结",
                "fallback_disclaimer": "本摘要为自动生成的备用版本"
            }
        }
        return labels.get(lang_code, labels["en"])
    
    def is_available(self) -> bool:
        """
        检查摘要服务是否可用
        
        Returns:
            True if OpenAI API is configured, False otherwise
        """
        return self.client is not None
