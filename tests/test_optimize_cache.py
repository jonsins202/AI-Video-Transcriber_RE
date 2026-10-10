"""优化结果缓存（optimize_cache）+ 批量参数化（LOSSLESS_BATCH_CHARS/MAX_TOKENS）测试。"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import optimize_cache  # noqa: E402
from summarizer import Summarizer  # noqa: E402

# ---- 复用 test_summarizer_optimize 的 FakeClient 形状 ----
from test_summarizer_optimize import _FakeClient, _Response  # noqa: E402

# 占位凭据仅供构造客户端（随即被 _FakeClient 替换），非真实密钥
_PLACEHOLDER_KEY = os.environ.get("TEST_API_KEY", "placeholder-not-a-real-key")


class OptimizeCacheModuleTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)
        self._p1 = patch.object(optimize_cache, "CACHE_DIR", self.tmp_dir)
        self._p1.start()
        self.addCleanup(self._p1.stop)
        self.addCleanup(self._tmp.cleanup)

    def test_roundtrip(self):
        key = optimize_cache.make_key("正文内容", "deepseek-chat", optimize_cache.LOSSLESS_PROFILE)
        self.assertIsNone(optimize_cache.lookup(key))
        optimize_cache.store(key, "# 优化后的正文")
        self.assertEqual(optimize_cache.lookup(key), "# 优化后的正文")

    def test_key_sensitivity(self):
        base = optimize_cache.make_key("文本", "model-a", optimize_cache.LOSSLESS_PROFILE)
        self.assertEqual(base, optimize_cache.make_key("文本", "model-a", optimize_cache.LOSSLESS_PROFILE))
        self.assertNotEqual(base, optimize_cache.make_key("文本2", "model-a", optimize_cache.LOSSLESS_PROFILE))
        self.assertNotEqual(base, optimize_cache.make_key("文本", "model-b", optimize_cache.LOSSLESS_PROFILE))
        self.assertNotEqual(base, optimize_cache.make_key("文本", "model-a", optimize_cache.CLEANUP_PROFILE))

    def test_empty_text_not_stored(self):
        key = optimize_cache.make_key("x", "m", "p")
        optimize_cache.store(key, "  ")
        self.assertIsNone(optimize_cache.lookup(key))

    def test_disabled_skips_lookup_and_store(self):
        key = optimize_cache.make_key("正文", "m", "p")
        optimize_cache.store(key, "结果")
        with patch.dict(os.environ, {"OPTIMIZE_CACHE": "0"}):
            self.assertIsNone(optimize_cache.lookup(key))
            optimize_cache.store(key, "新结果")
        self.assertEqual(optimize_cache.lookup(key), "结果")

    def test_missing_file_returns_none(self):
        self.assertIsNone(optimize_cache.lookup("0" * 64))


class _FakeSummarizer:
    """供 _run_post_extract_pipeline 使用的最小假 Summarizer。"""

    fast_model = "fake-model"

    def __init__(self):
        self.optimize_calls = 0
        self.cleanup_calls = 0
        self.summarize_calls = 0

    def extract_transcript_body(self, raw):
        return raw.split("## 转录内容", 1)[-1].strip()

    async def optimize_transcript_body(self, body, transcript_language=None, on_progress=None):
        self.optimize_calls += 1
        return f"优化({body})"

    async def cleanup_transcript_body(self, body, transcript_language=None, on_progress=None):
        self.cleanup_calls += 1
        return f"清理({body})"

    async def summarize(self, text, lang, title=None):
        self.summarize_calls += 1
        return "摘要"

    def get_last_optimization_audit(self):
        return None

    def get_last_cleanup_audit(self):
        return None

    def reset_audits(self):
        pass


class PipelineOptimizeCacheTests(unittest.IsolatedAsyncioTestCase):
    """同输入同模型二次进管线：优化/清理阶段零调用，产物一致。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        patchers = [
            patch.object(optimize_cache, "CACHE_DIR", Path(self._tmp.name) / "cache"),
            patch.dict(os.environ, {"OPTIMIZE_CACHE": "1", "SKIP_OPTIMIZE": "0"}),
        ]
        for p in patchers:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self._tmp.cleanup)

        import main
        self.main = main
        self.fake = _FakeSummarizer()
        pipeline_patchers = [
            patch.object(main, "tasks", {}),
            patch.object(main, "save_tasks", lambda t: None),
            patch.object(main, "broadcast_task_update", self._noop),
            patch.object(main, "TEMP_DIR", Path(self._tmp.name)),
            patch.object(main.transcriber, "get_detected_language", lambda raw: "zh"),
            patch.object(main.translator, "normalize_lang_code", lambda c: c),
            patch.object(main.translator, "languages_differ_for_translation", lambda a, b: False),
        ]
        for p in pipeline_patchers:
            p.start()
            self.addCleanup(p.stop)

    async def _noop(self, *args, **kwargs):
        pass

    async def _run(self, run_tag: str, oral_cleanup: bool = False) -> dict:
        task_id = f"task-{run_tag}"
        self.main.tasks[task_id] = {"status": "processing", "progress": 0}
        raw = "# 视频转录\n\n## 转录内容\n\n正文甲乙丙。" * 5
        await self.main._run_post_extract_pipeline(
            task_id=task_id,
            raw_script=raw,
            video_title="测试",
            source_ref="test://x",
            summary_language="zh",
            request_summarizer=self.fake,
            oral_cleanup=oral_cleanup,
        )
        return self.main.tasks[task_id]

    async def test_second_run_skips_optimize(self):
        first = await self._run("a")
        self.assertEqual(self.fake.optimize_calls, 1)
        self.assertIn("优化(", first["script"])

        second = await self._run("b")
        self.assertEqual(self.fake.optimize_calls, 1)   # 未再调用
        self.assertEqual(self.fake.summarize_calls, 2)  # 摘要正常执行
        self.assertEqual(second.get("optimize_cache"), "hit")
        self.assertEqual(second["script"], first["script"])

    async def test_cleanup_stage_cached_too(self):
        await self._run("c1", oral_cleanup=True)
        self.assertEqual(self.fake.optimize_calls, 1)
        self.assertEqual(self.fake.cleanup_calls, 1)

        second = await self._run("c2", oral_cleanup=True)
        self.assertEqual(self.fake.optimize_calls, 1)  # 两级缓存都命中
        self.assertEqual(self.fake.cleanup_calls, 1)
        self.assertEqual(second.get("optimize_cache"), "hit")
        self.assertEqual(second.get("cleanup_cache"), "hit")

    async def test_model_change_revalidates(self):
        await self._run("a")
        self.fake.fast_model = "another-model"
        await self._run("b")
        self.assertEqual(self.fake.optimize_calls, 2)  # 换模型 → 真实重跑

    async def test_skip_optimize_env(self):
        # SKIP_OPTIMIZE=1：跳过无损优化与优化缓存，文稿即 ASR 原文，摘要仍执行
        with patch.dict(os.environ, {"SKIP_OPTIMIZE": "1"}):
            result = await self._run("skip1")
        self.assertEqual(self.fake.optimize_calls, 0)
        self.assertEqual(self.fake.summarize_calls, 1)
        self.assertNotIn("optimize_cache", result)
        self.assertNotIn("优化(", result["script"])


class BatchConfigTests(unittest.IsolatedAsyncioTestCase):
    """LOSSLESS_BATCH_CHARS / LOSSLESS_MAX_TOKENS 环境变量生效。"""

    def _make_summarizer(self):
        summarizer = Summarizer(api_key=_PLACEHOLDER_KEY)
        self.calls = []
        summarizer.client = _FakeClient(self.calls, self._echo)
        return summarizer

    @staticmethod
    def _echo(kwargs):
        request = json.loads(kwargs["messages"][-1]["content"])
        return _Response(json.dumps({"segments": [
            {"id": seg["id"], "text": seg["text"]} for seg in request["segments"]
        ]}, ensure_ascii=False))

    async def test_batch_chars_env_controls_call_count(self):
        source = "甲" * 6000  # 无标点：每 300 字切段 → 20 段

        with patch.dict(os.environ, {"LOSSLESS_CONCURRENCY": "1", "LOSSLESS_BATCH_CHARS": "5000"}):
            s1 = self._make_summarizer()
            await s1.optimize_transcript_body(source, "zh")
            calls_big = len(self.calls)

        self.calls.clear()
        with patch.dict(os.environ, {"LOSSLESS_CONCURRENCY": "1", "LOSSLESS_BATCH_CHARS": "2700"}):
            s2 = self._make_summarizer()
            await s2.optimize_transcript_body(source, "zh")
            calls_small = len(self.calls)

        self.assertEqual(calls_big, 2)    # 5000 字/批 → 2 批
        self.assertEqual(calls_small, 3)  # 2700 字/批 → 3 批

    async def test_max_tokens_env_used(self):
        with patch.dict(os.environ, {"LOSSLESS_CONCURRENCY": "1", "LOSSLESS_MAX_TOKENS": "7777"}):
            s = self._make_summarizer()
            await s.optimize_transcript_body("甲" * 400, "zh")
        self.assertEqual(self.calls[0]["max_tokens"], 7777)

    async def test_default_max_tokens_is_8000(self):
        with patch.dict(os.environ, {"LOSSLESS_CONCURRENCY": "1"}):
            os.environ.pop("LOSSLESS_MAX_TOKENS", None)
            s = self._make_summarizer()
            await s.optimize_transcript_body("甲" * 400, "zh")
        self.assertEqual(self.calls[0]["max_tokens"], 8000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
