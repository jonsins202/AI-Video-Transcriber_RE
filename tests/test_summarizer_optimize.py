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
import unittest
from unittest.mock import patch

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
    def __init__(self, content):
        self.message = _Message(content)


class _Response:
    def __init__(self, content):
        self.choices = [_Choice(content)]


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
