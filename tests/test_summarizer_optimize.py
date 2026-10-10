"""回归测试：长文本分块优化的"上文续"标记泄漏（task:37ee27 案例）+ 对抗补强。

修复演进：
1. 初版：把 [上文续：…] 拼进正文 → DeepSeek 偶发去括号复述 → 清理正则失效 → 泄漏。
2. 治本：上一块结尾放进独立 <previous_context> 参考区（XML 结构化），prompt 禁止复述。
3. 补强（对抗审查后）：prev_context 取"优化后输出"结尾；新增 _strip_context_echo 后置
   相似度兜底——因为邻接去重只兜逐字滚动重叠，兜不住"LLM 整段从头复述"（实测 ratio=0.752）。

mock 掉 OpenAI client（不依赖真实 API），覆盖：
- prompt 结构：prev_context 进 <previous_context> 区、不污染正文、不注入旧标记（中英）；
- _strip_context_echo：逐字复述 / 真实案例改写复述 都能截断；不相似时不误删；
- 端到端：听话 LLM 无泄漏；不听话 LLM（复述上下文）下 strip 被调用且无标签泄漏；
- 首块无上下文；英文分块路径对等。

运行：
    venv\\Scripts\\python.exe tests\\test_summarizer_optimize.py
"""
import os
import sys
import json
import time
import threading
import unittest
import httpx
import openai
from unittest.mock import patch, AsyncMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from summarizer import Summarizer  # noqa: E402


# ---- 多样化语料（避免相邻块开头/结尾天然相似触发 false positive）----
POOL = [
    "人工智能的发展经历了多个不同的阶段",
    "深度学习技术在二零一二年迎来了重大突破",
    "transformer 架构彻底改变了自然语言处理领域",
    "大规模语言模型展现出令人惊讶的涌现能力",
    "多模态技术成功融合了计算机视觉与自然语言",
    "强化学习从人类反馈中提升了模型的可用性",
    "开源模型生态推动了整个行业的快速进步",
    "推理能力的提升让模型能够处理更复杂的任务",
    "智能体架构开始具备工具调用与自主规划能力",
    "模型训练的算力需求呈现指数级别的增长趋势",
]


def _long_text(repeats=40):
    return "。".join(POOL * repeats) + "。"


# ---- mock OpenAI client ----

class _Message:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content, finish_reason="stop"):
        self.message = _Message(content)
        self.finish_reason = finish_reason


class _Response:
    def __init__(self, content, finish_reason="stop"):
        self.choices = [_Choice(content, finish_reason)]


class _FakeCompletions:
    def __init__(self, recorder, responder):
        self._recorder = recorder
        self._responder = responder

    def create(self, **kwargs):
        self._recorder.append(kwargs)
        return self._responder(kwargs)


class _FakeChat:
    def __init__(self, recorder, responder):
        self.completions = _FakeCompletions(recorder, responder)


class _FakeClient:
    def __init__(self, recorder, responder):
        self.chat = _FakeChat(recorder, responder)


def _extract_body(user_prompt):
    if "原始转录文本：" in user_prompt:
        return user_prompt.split("原始转录文本：", 1)[1].strip()
    if "Original transcript text:" in user_prompt:
        return user_prompt.split("Original transcript text:", 1)[1].strip()
    return user_prompt


def _extract_prev_context(user_prompt):
    """从 user prompt 取 <previous_context>...</previous_context> 内容。"""
    if "<previous_context>" in user_prompt:
        return user_prompt.split("<previous_context>", 1)[1].split("</previous_context>", 1)[0].strip()
    return ""


class OptimizeChunkContextTests(unittest.IsolatedAsyncioTestCase):

    def _make_summarizer(self, responder):
        s = Summarizer(api_key="sk-test-dummy")
        self.calls = []
        s.client = _FakeClient(self.calls, responder)
        return s

    @staticmethod
    def _echo_body_responder(kwargs):
        # 听话 LLM：只回吐正文
        return _Response(_extract_body(kwargs["messages"][-1]["content"]))

    @staticmethod
    def _disobedient_responder(kwargs):
        # 不听话 LLM：把 <previous_context> 复述进输出开头（task:37ee27 的真实诱因）
        user = kwargs["messages"][-1]["content"]
        body = _extract_body(user)
        echo = _extract_prev_context(user)
        return _Response((echo + "\n" + body) if echo else body)

    @staticmethod
    def _jump_skip_responder(kwargs):
        # 跳段 LLM：只返回块正文头尾，跳过中间（task:1128f3 块3 的真实诱因）
        body = _extract_body(kwargs["messages"][-1]["content"])
        if len(body) > 200:
            return _Response(body[:80] + "\n" + body[-80:])
        return _Response(body)

    @staticmethod
    def _aggressive_trim_responder(kwargs):
        # 过度精简 LLM：只返回块正文的前一半（把正文当冗余删）
        body = _extract_body(kwargs["messages"][-1]["content"])
        return _Response(body[:len(body) // 2])

    @staticmethod
    def _empty_first_responder_factory():
        calls = {"count": 0}

        def responder(kwargs):
            calls["count"] += 1
            body = _extract_body(kwargs["messages"][-1]["content"])
            return _Response("" if calls["count"] == 1 else body)

        return responder

    # ===== _strip_context_echo 单元 =====

    def test_strip_context_echo_word_for_word(self):
        s = Summarizer(api_key="sk-test-dummy")
        ctx = "这是上一段结尾的复述内容用于测试截断。"
        body = "这里是真正的正文应该被完整保留下来。"
        out = s._strip_context_echo(ctx + body, ctx)
        self.assertNotIn(ctx, out)
        self.assertIn(body, out)

    def test_strip_context_echo_real_case(self):
        """用 task:37ee27 真实数据：邻接去重对改写复述返回空（H1），_strip_context_echo 能截断。"""
        s = Summarizer(api_key="sk-test-dummy")
        f = r"E:\ai_video_transcript\temp\transcript_AI_发展了_4_年把应用发展没了AI_年中复盘_37ee27.md"
        if not os.path.exists(f):
            self.skipTest("真实案例文件不存在，跳过")
        with open(f, encoding="utf-8") as fh:
            lines = fh.read().split("\n")
        prev_context = lines[60][-100:]                  # 上一块优化后输出结尾
        echoed = lines[62].replace("上文续：", "", 1)     # 模拟 LLM 复述（去掉标记词）
        # H1：邻接去重要求逐字符后缀==前缀，对改写复述失效
        self.assertEqual(s._find_overlap_between_texts(prev_context, echoed[:200]), "")
        # 兜底：相似度识别并截断
        stripped = s._strip_context_echo(echoed, prev_context)
        self.assertLess(len(stripped), len(echoed),
                        "改写复述未被截断，_strip_context_echo 对真实案例失效")

    def test_strip_context_echo_no_false_positive(self):
        """正文与 prev_context 不相似时不应误删。"""
        s = Summarizer(api_key="sk-test-dummy")
        ctx = "昨天我去超市买了苹果香蕉和橙子回家。"
        body = "量子力学主要研究微观粒子的运动规律与相互作用。"
        out = s._strip_context_echo(body, ctx)
        self.assertEqual(out, body, "未复述却被截断 → 误删正文")

    # ===== prompt 结构 =====

    async def test_prev_context_in_xml_block(self):
        s = self._make_summarizer(self._echo_body_responder)
        ctx = "这是上一段的结尾内容提到一个转折点。"
        body = "这是当前段需要被优化的正文句子。"
        await s._format_single_chunk(body, "zh", prev_context=ctx)
        self.assertEqual(len(self.calls), 1)
        user = self.calls[0]["messages"][-1]["content"]
        sysp = self.calls[0]["messages"][0]["content"]
        self.assertIn("<previous_context>", user)
        self.assertIn("</previous_context>", user)
        self.assertIn(ctx, user)
        after = user.split("原始转录文本：", 1)[1]
        self.assertIn(body, after)
        self.assertNotIn(ctx, after)
        self.assertNotIn("[上文续", user)
        self.assertNotIn("上文续：", user)
        self.assertNotIn("[Context continued", user)
        self.assertIn("复述", sysp)

    async def test_no_context_omits_block(self):
        s = self._make_summarizer(self._echo_body_responder)
        await s._format_single_chunk("没有上下文的单块正文内容。", "zh")
        user = self.calls[0]["messages"][-1]["content"]
        self.assertNotIn("<previous_context>", user)
        self.assertNotIn("[上文续", user)

    async def test_english_prev_context_xml_block(self):
        s = self._make_summarizer(self._echo_body_responder)
        ctx = "End of previous section about AI cycles."
        body = "This is the current segment to optimize."
        await s._format_single_chunk(body, "en", prev_context=ctx)
        user = self.calls[0]["messages"][-1]["content"]
        sysp = self.calls[0]["messages"][0]["content"]
        self.assertIn("<previous_context>", user)
        self.assertIn(ctx, user)
        after = user.split("Original transcript text:", 1)[1]
        self.assertNotIn(ctx, after)
        self.assertNotIn("[Context continued", user)
        self.assertIn("NEVER reproduce", sysp)

    # ===== 端到端 =====

    async def test_long_transcript_no_marker_leak(self):
        s = self._make_summarizer(self._echo_body_responder)
        text = _long_text()
        self.assertGreater(len(text), 4000)
        result = await s._format_long_transcript_in_chunks(text, "zh", 4000)
        self.assertNotIn("上文续", result)
        self.assertNotIn("[上文续", result)
        self.assertNotIn("<previous_context>", result)
        with_ctx = [c for c in self.calls if "<previous_context>" in c["messages"][-1]["content"]]
        self.assertGreater(len(with_ctx), 0)
        for c in self.calls:
            self.assertNotIn("[上文续", c["messages"][-1]["content"])

    async def test_first_chunk_has_no_context(self):
        s = self._make_summarizer(self._echo_body_responder)
        await s._format_long_transcript_in_chunks(_long_text(), "zh", 4000)
        first_user = self.calls[0]["messages"][-1]["content"]
        self.assertNotIn("<previous_context>", first_user)

    async def test_disobedient_path_strips_and_no_leak(self):
        """对抗：LLM 把上下文复述进输出，_strip_context_echo 必被调用且无标签泄漏。"""
        s = self._make_summarizer(self._disobedient_responder)
        with patch.object(s, "_strip_context_echo", wraps=s._strip_context_echo) as spy:
            result = await s._format_long_transcript_in_chunks(_long_text(), "zh", 4000)
        self.assertNotIn("<previous_context>", result)
        self.assertNotIn("上文续", result)
        # 非首块都有 prev_context → disobedient 复述 → strip 必然被调用
        self.assertGreater(spy.call_count, 0, "_strip_context_echo 未被调用，对抗防线未接入")

    async def test_english_long_transcript_no_leak(self):
        s = self._make_summarizer(self._echo_body_responder)
        # 用多样英文句（非模板句），反映真实转录、避免相邻块高度同质误触发 strip
        en_pool = [
            "Artificial intelligence has evolved through several distinct phases",
            "Deep learning achieved a major breakthrough in two thousand twelve",
            "The transformer architecture reshaped natural language processing",
            "Large language models began to demonstrate surprising emergent abilities",
            "Multimodal systems successfully fused vision and language together",
            "Reinforcement learning from human feedback improved model usability",
            "Open source ecosystems accelerated progress across the whole industry",
            "Improved reasoning enabled models to tackle far more complex tasks",
            "Agentic architectures gained tool use and autonomous planning skills",
            "Compute requirements for training grew at an exponential rate",
        ]
        text = ". ".join(en_pool * 60) + "."
        self.assertGreater(len(text), 4000)
        result = await s._format_long_transcript_in_chunks(text, "en", 4000)
        self.assertNotIn("<previous_context>", result)
        self.assertNotIn("[Context continued", result)
        with_ctx = [c for c in self.calls if "<previous_context>" in c["messages"][-1]["content"]]
        self.assertGreater(len(with_ctx), 0)

    # ===== 丢失类对抗：单块 70% 兜底应拦截跳段/精简，整体不被掏空 =====

    async def test_jump_skip_chunk_fallback(self):
        """对抗：LLM 跳段（输出头尾、丢中间，task:1128f3 诱因），单块兜底应拦截并保留全部。"""
        s = self._make_summarizer(self._jump_skip_responder)
        text = _long_text()
        result = await s._format_long_transcript_in_chunks(text, "zh", 4000)
        # 跳段输出 ~160字 << 块的 70% → 单块兜底用原始块 → 整体应保留大部分内容
        self.assertGreater(len(result), len(text) * 0.8,
                           "跳段丢失未被单块兜底拦截，整体内容被掏空")

    async def test_aggressive_trim_chunk_fallback(self):
        """对抗：LLM 每块过度精简到 50%，单块兜底应每块拦截，整体保留全部。"""
        s = self._make_summarizer(self._aggressive_trim_responder)
        text = _long_text()
        result = await s._format_long_transcript_in_chunks(text, "zh", 4000)
        self.assertGreater(len(result), len(text) * 0.8,
                           "过度精简未被单块兜底拦截，内容大量丢失")

    async def test_empty_chunk_falls_back_to_original(self):
        s = self._make_summarizer(self._empty_first_responder_factory())
        text = "甲" * 4000 + "乙" * 4000 + "丙" * 4000
        result = await s._format_long_transcript_in_chunks(text, "zh", 4000)
        self.assertIn("甲" * 100, result)
        self.assertIn("乙" * 100, result)
        self.assertIn("丙" * 100, result)

    # ===== 整体长度兜底（单块兜底失效时的最后防线）=====

    def test_maybe_overall_fallback_triggers_when_short(self):
        s = Summarizer(api_key="sk-test-dummy")
        raw = "这是用于测试整体兜底的原始正文句子内容。" * 40
        merged_short = "极短的优化结果。"  # 远 < 60% → 应触发整体兜底
        out = s._maybe_overall_fallback(merged_short, raw)
        self.assertNotEqual(out, merged_short)
        self.assertGreaterEqual(len(out), len(raw) * 0.9)  # 兜底后接近原文长度

    def test_maybe_overall_fallback_no_trigger_when_ok(self):
        s = Summarizer(api_key="sk-test-dummy")
        raw = "这是用于测试整体兜底的原始正文句子内容。" * 40
        merged_ok = raw + "额外保留的内容。"  # > 100% → 不应触发
        out = s._maybe_overall_fallback(merged_ok, raw)
        self.assertEqual(out, merged_ok)

    def test_empty_overall_result_falls_back(self):
        s = Summarizer(api_key="sk-test-dummy")
        raw = "这是不可丢失的原始正文" * 50
        out = s._maybe_overall_fallback("", raw)
        self.assertIn("这是不可丢失的原始正文", out)


class LosslessOptimizeTests(unittest.IsolatedAsyncioTestCase):
    def _make_summarizer(self, responder):
        summarizer = Summarizer(api_key="sk-test-dummy")
        self.calls = []
        summarizer.client = _FakeClient(self.calls, responder)
        return summarizer

    @staticmethod
    def _request_payload(kwargs):
        return json.loads(kwargs["messages"][-1]["content"])

    @classmethod
    def _echo_response(cls, kwargs):
        request = cls._request_payload(kwargs)
        return _Response(json.dumps({
            "segments": [
                {"id": segment["id"], "text": segment["text"]}
                for segment in request["segments"]
            ]
        }, ensure_ascii=False))

    async def test_lossless_optimizer_restores_empty_segment(self):
        def responder(kwargs):
            request = self._request_payload(kwargs)
            response_segments = [
                {"id": segment["id"], "text": segment["text"]}
                for segment in request["segments"]
            ]
            response_segments[0]["text"] = ""
            return _Response(json.dumps({"segments": response_segments}, ensure_ascii=False))

        summarizer = self._make_summarizer(responder)
        source = "开场白绝对不能丢失" * 100
        result = await summarizer.optimize_transcript_body(source, "zh")
        self.assertIn("开场白绝对不能丢失", result)
        audit = summarizer.get_last_optimization_audit()
        self.assertGreaterEqual(audit["status_counts"].get("fallback_empty", 0), 1)

    async def test_lossless_optimizer_splits_truncated_batch(self):
        # pin 到串行：mock create 在真实线程里跑，共享 call_count 非原子、"首调"非确定，
        # 并发下断言不可靠。该测试聚焦二分拆分逻辑本身，与并发无关。
        with patch.dict(os.environ, {"LOSSLESS_CONCURRENCY": "1"}):
            call_count = {"value": 0}

            def responder(kwargs):
                call_count["value"] += 1
                if call_count["value"] == 1:
                    return _Response("", finish_reason="length")
                return self._echo_response(kwargs)

            summarizer = self._make_summarizer(responder)
            source = "这一段内容用于测试截断后拆分批次" * 180
            result = await summarizer.optimize_transcript_body(source, "zh")
            self.assertIn("这一段内容用于测试截断后拆分批次", result)
            self.assertGreaterEqual(call_count["value"], 3)
            self.assertNotIn("fallback_finish_reason", summarizer.get_last_optimization_audit()["status_counts"])

    async def test_lossless_optimizer_rejects_summary_rewrite(self):
        def responder(kwargs):
            request = self._request_payload(kwargs)
            return _Response(json.dumps({
                "segments": [
                    {"id": segment["id"], "text": "本段主要讨论访谈主题。" * 50}
                    for segment in request["segments"]
                ]
            }, ensure_ascii=False))

        summarizer = self._make_summarizer(responder)
        source = "主持人的开场寒暄引用举例和详细论述都必须保留" * 100
        result = await summarizer.optimize_transcript_body(source, "zh")
        self.assertIn("主持人的开场寒暄引用举例和详细论述都必须保留", result)
        audit = summarizer.get_last_optimization_audit()
        self.assertGreater(audit["status_counts"].get("fallback_fidelity", 0), 0)

    async def test_audit_contains_metadata_not_transcript(self):
        summarizer = self._make_summarizer(self._echo_response)
        source = "审计中不能保存完整正文" * 80
        await summarizer.optimize_transcript_body(source, "zh")
        audit = summarizer.get_last_optimization_audit()
        json.dumps(audit, ensure_ascii=False)
        serialized = json.dumps(audit, ensure_ascii=False)
        self.assertNotIn(source, serialized)
        self.assertNotIn("source_text", serialized)
        self.assertNotIn("candidate_text", serialized)
        self.assertIn("source_hash", serialized)

    async def test_historical_21_segment_empty_pattern_preserves_every_block(self):
        empty_indexes = {1, 3, 4, 12, 13, 14, 16, 20}

        def responder(kwargs):
            request = self._request_payload(kwargs)
            response_segments = []
            for segment in request["segments"]:
                index = int(segment["id"].split("-")[1])
                response_segments.append({
                    "id": segment["id"],
                    "text": "" if index in empty_indexes else segment["text"],
                })
            return _Response(
                json.dumps({"segments": response_segments}, ensure_ascii=False)
            )

        summarizer = self._make_summarizer(responder)
        blocks = []
        for index in range(21):
            marker = f"历史片段{index:02d}不可丢失"
            blocks.append((marker + "正文内容必须完整保留" * 40)[:300])
        source = "".join(blocks)

        result = await summarizer.optimize_transcript_body(source, "zh")

        for index in range(21):
            self.assertIn(f"历史片段{index:02d}不可丢失", result)
        audit = summarizer.get_last_optimization_audit()
        self.assertEqual(audit["segment_count"], 21)
        self.assertEqual(audit["status_counts"].get("fallback_empty"), 8)

    async def test_lossless_concurrent_equivalence_and_bounded(self):
        """并发：实测并发度受 Semaphore 约束，且结果与串行完全一致。"""
        lock = threading.Lock()
        state = {"in_flight": 0, "max_in_flight": 0}

        def responder(kwargs):
            with lock:
                state["in_flight"] += 1
                state["max_in_flight"] = max(state["max_in_flight"], state["in_flight"])
            time.sleep(0.05)  # 强制重叠让并发可观测
            with lock:
                state["in_flight"] -= 1
            return self._echo_response(kwargs)

        source = "并发等价校验每一批必须独立且结果稳定不丢" * 300

        with patch.dict(os.environ, {"LOSSLESS_CONCURRENCY": "3"}):
            summarizer_cc = self._make_summarizer(responder)
            result_cc = await summarizer_cc.optimize_transcript_body(source, "zh")

        self.assertGreaterEqual(state["max_in_flight"], 2)  # 确实并发了
        self.assertLessEqual(state["max_in_flight"], 3)     # 受 Semaphore 约束

        with patch.dict(os.environ, {"LOSSLESS_CONCURRENCY": "1"}):
            summarizer_ser = self._make_summarizer(self._echo_response)
            result_ser = await summarizer_ser.optimize_transcript_body(source, "zh")

        self.assertEqual(result_cc, result_ser)  # 并发结果 == 串行结果

    async def test_lossless_429_backoff_recovers(self):
        """429：前两次限流，退避重试后恢复，正文保留。"""
        call_count = {"value": 0}

        def responder(kwargs):
            call_count["value"] += 1
            if call_count["value"] <= 2:
                resp = httpx.Response(429, request=httpx.Request("POST", "https://x"))
                raise openai.RateLimitError("rate limited", response=resp, body=None)
            return self._echo_response(kwargs)

        summarizer = self._make_summarizer(responder)
        source = "限流后应能通过退避重试恢复并完整保留正文" * 150

        with patch.dict(os.environ, {"LOSSLESS_CONCURRENCY": "1"}), \
             patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await summarizer.optimize_transcript_body(source, "zh")

        self.assertIn("限流后应能通过退避重试恢复并完整保留正文", result)
        self.assertGreaterEqual(mock_sleep.await_count, 2)

    async def test_lossless_429_exhausted_falls_back_to_original(self):
        """429：持续限流耗尽重试后回退原文，不丢内容。"""
        def responder(kwargs):
            resp = httpx.Response(429, request=httpx.Request("POST", "https://x"))
            raise openai.RateLimitError("rate limited", response=resp, body=None)

        summarizer = self._make_summarizer(responder)
        source = "持续限流时必须回退原文且不得丢失任何内容" * 150

        with patch.dict(os.environ, {"LOSSLESS_CONCURRENCY": "1"}), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            result = await summarizer.optimize_transcript_body(source, "zh")

        self.assertIn("持续限流时必须回退原文且不得丢失任何内容", result)
        audit = summarizer.get_last_optimization_audit()
        self.assertGreater(audit["status_counts"].get("fallback_api", 0), 0)


class TranscriptBodyTests(unittest.TestCase):
    def test_extract_body_removes_english_metadata(self):
        raw = """# Video Transcription

**Detected Language:** woitn
**Language Probability:** —

## Transcription Content

hello 大家好这是正文
"""
        self.assertEqual(
            Summarizer.extract_transcript_body(raw),
            "hello 大家好这是正文",
        )

    def test_extract_body_removes_chinese_metadata(self):
        raw = """# 视频转录

**检测语言:** zh
**语言概率:** 0.99

## 转录内容

这是正文
"""
        self.assertEqual(Summarizer.extract_transcript_body(raw), "这是正文")

    def test_extract_body_removes_timestamps_only(self):
        raw = (
            "**[00:00 - 00:05]**\n\n第一段\n\n"
            "**[00:05 - 00:10]**\n\n第二段"
        )
        self.assertEqual(
            Summarizer.extract_transcript_body(raw),
            "第一段\n\n第二段",
        )

    def test_extract_body_preserves_unrecognized_markdown_content(self):
        raw = "## 用户自己的标题\n\n正文 **强调内容** 不得删除"
        self.assertEqual(Summarizer.extract_transcript_body(raw), raw)


class CleanupPipelineTests(unittest.IsolatedAsyncioTestCase):
    """口语清理管线：复用 lossless 引擎，CLEANUP 档校验；失败回退输入原文。"""

    # 占位凭据仅供构造客户端（随后即被 _FakeClient 替换），非真实密钥
    _PLACEHOLDER_KEY = os.environ.get("TEST_API_KEY", "placeholder-not-a-real-key")

    def _make_summarizer(self, responder):
        summarizer = Summarizer(api_key=self._PLACEHOLDER_KEY)
        self.calls = []
        summarizer.client = _FakeClient(self.calls, responder)
        return summarizer

    @staticmethod
    def _request_payload(kwargs):
        return json.loads(kwargs["messages"][-1]["content"])

    @staticmethod
    def _echo_kwargs_responder(kwargs):
        request = json.loads(kwargs["messages"][-1]["content"])
        return _Response(json.dumps({"segments": [
            {"id": seg["id"], "text": seg["text"]}
            for seg in request["segments"]
        ]}, ensure_ascii=False))

    async def test_cleanup_removes_fillers(self):
        def responder(kwargs):
            request = self._request_payload(kwargs)
            return _Response(json.dumps({"segments": [
                {"id": seg["id"], "text": seg["text"].replace("呃", "").replace("啊", "").replace("嗯", "")}
                for seg in request["segments"]
            ]}, ensure_ascii=False))

        summarizer = self._make_summarizer(responder)
        source = "呃我们今天聊一聊啊人工智能的行业发展嗯顺便说说大模型的机会" * 60
        result = await summarizer.cleanup_transcript_body(source, "zh")
        self.assertNotIn("呃", result)
        self.assertNotIn("嗯", result)
        self.assertIn("我们今天聊一聊人工智能的行业发展", result)
        audit = summarizer.get_last_cleanup_audit()
        self.assertGreater(audit["status_counts"].get("optimized", 0), 0)

    async def test_cleanup_bad_ids_fall_back_to_source(self):
        def responder(kwargs):
            return _Response(json.dumps(
                {"segments": [{"id": "wrong-id", "text": "被篡改的正文"}]},
                ensure_ascii=False,
            ))

        summarizer = self._make_summarizer(responder)
        source = "口误重复的内容也需要完整保留" * 40
        result = await summarizer.cleanup_transcript_body(source, "zh")
        self.assertEqual(result.replace("\n", ""), source)
        audit = summarizer.get_last_cleanup_audit()
        self.assertGreater(audit["status_counts"].get("fallback_structure", 0), 0)

    async def test_cleanup_api_failure_returns_source(self):
        def responder(kwargs):
            raise RuntimeError("boom")

        summarizer = self._make_summarizer(responder)
        source = "接口异常情况下口语清理必须回退原文" * 40
        result = await summarizer.cleanup_transcript_body(source, "zh")
        self.assertEqual(result.replace("\n", ""), source)
        audit = summarizer.get_last_cleanup_audit()
        self.assertGreater(audit["status_counts"].get("fallback_api", 0), 0)

    async def test_cleanup_prompt_forbids_content_changes(self):
        summarizer = self._make_summarizer(self._echo_kwargs_responder)
        source = "检查清理提示词的正文" * 40
        await summarizer.cleanup_transcript_body(source, "zh")
        system_prompt = self.calls[0]["messages"][0]["content"]
        self.assertIn("口语清理", system_prompt)
        self.assertIn("不删实义内容", system_prompt)
        self.assertIn("人名/数字/术语/引用必须原样保留", system_prompt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
