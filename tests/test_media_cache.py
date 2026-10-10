"""媒体缓存（media_cache）单测：索引往返、惰性清理、ASR 指纹失效、开关。"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import media_cache  # noqa: E402

URL = "https://www.youtube.com/watch?v=abc123"


class MediaCacheTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)
        self._patch_temp = patch.object(media_cache, "TEMP_DIR", self.tmp_dir)
        self._patch_index = patch.object(
            media_cache, "INDEX_FILE", self.tmp_dir / "media_cache.json"
        )
        self._patch_temp.start()
        self._patch_index.start()
        self.addCleanup(self._patch_temp.stop)
        self.addCleanup(self._patch_index.stop)
        self.addCleanup(self._tmp.cleanup)

    # ---- 键与开关 ----

    def test_cache_key_accepts_http_only(self):
        self.assertEqual(media_cache.cache_key("  https://a.com/x  "), "https://a.com/x")
        self.assertEqual(media_cache.cache_key("HTTP://a.com/x"), "HTTP://a.com/x")
        self.assertIsNone(media_cache.cache_key("ftp://a.com/x"))
        self.assertIsNone(media_cache.cache_key(""))
        self.assertIsNone(media_cache.cache_key("file:///etc/passwd"))

    def test_disabled_skips_everything(self):
        media_cache.record_raw(URL, "正文", "标题", "zh")
        with patch.dict(os.environ, {"MEDIA_CACHE": "0"}):
            self.assertIsNone(media_cache.lookup_raw(URL))
            self.assertIsNone(media_cache.lookup_audio(URL))
            media_cache.record_raw(URL, "新正文", "新标题", "zh")
            media_cache.record_audio(URL, self.tmp_dir / "nonexistent.wav")
        # 关闭期间不写入
        self.assertEqual(media_cache.lookup_raw(URL)["raw_text"], "正文")

    # ---- raw 缓存 ----

    def test_record_and_lookup_raw_roundtrip(self):
        media_cache.record_raw(URL, "# 视频转录\n\n正文内容", "测试标题", "zh")
        hit = media_cache.lookup_raw(URL)
        self.assertIsNotNone(hit)
        self.assertEqual(hit["raw_text"], "# 视频转录\n\n正文内容")
        self.assertEqual(hit["video_title"], "测试标题")
        self.assertEqual(hit["detected_language"], "zh")

    def test_raw_file_deleted_prunes_entry_keeps_audio(self):
        audio = self.tmp_dir / "audio_deadbeef.wav"
        audio.write_bytes(b"wav")
        media_cache.record_audio(URL, audio, "标题A")
        media_cache.record_raw(URL, "正文", "标题B", "zh")

        raw_file = self.tmp_dir / media_cache._raw_filename(media_cache.cache_key(URL))
        raw_file.unlink()
        self.assertIsNone(media_cache.lookup_raw(URL))

        # raw 失效不影响 audio 命中
        audio_hit = media_cache.lookup_audio(URL)
        self.assertIsNotNone(audio_hit)
        self.assertEqual(audio_hit["path"], audio)

    def test_asr_config_change_invalidates_raw_not_audio(self):
        media_cache.record_raw(URL, "正文", "标题", "zh")
        audio = self.tmp_dir / "audio_cafe.wav"
        audio.write_bytes(b"wav")
        media_cache.record_audio(URL, audio)

        with patch.dict(os.environ, {"FUNASR_MODEL": "sensevoice"}):
            self.assertIsNone(media_cache.lookup_raw(URL))
            self.assertIsNotNone(media_cache.lookup_audio(URL))

    def test_spk_toggle_invalidates_raw(self):
        # 显式控制 FUNASR_SPK 指纹，避免宿主环境/.env 里已有值影响测试
        with patch.dict(os.environ, {"FUNASR_SPK": ""}):
            media_cache.record_raw(URL, "正文", "标题", "zh")
            self.assertIsNotNone(media_cache.lookup_raw(URL))
        with patch.dict(os.environ, {"FUNASR_SPK": "cam++"}):
            self.assertIsNone(media_cache.lookup_raw(URL))
        # 关掉后指纹恢复 → 重新命中
        with patch.dict(os.environ, {"FUNASR_SPK": ""}):
            self.assertIsNotNone(media_cache.lookup_raw(URL))

    def test_per_request_spk_num_changes_asr_key(self):
        auto_key = media_cache.current_asr_key()
        fixed_key = media_cache.current_asr_key(spk_num=2)
        self.assertNotEqual(auto_key, fixed_key)
        self.assertIn("FUNASR_SPK_NUM=2", fixed_key)

        # 按请求人数登记后：同人数命中，不同人数/自动 未命中
        media_cache.record_raw(URL, "正文", "标题", "zh", asr_key=fixed_key)
        self.assertIsNotNone(media_cache.lookup_raw(URL, asr_key=fixed_key))
        self.assertIsNone(media_cache.lookup_raw(URL, asr_key=auto_key))
        self.assertIsNone(media_cache.lookup_raw(URL, asr_key=media_cache.current_asr_key(spk_num=3)))

    def test_record_raw_overwrites_and_keeps_audio(self):
        audio = self.tmp_dir / "audio_beef.wav"
        audio.write_bytes(b"wav")
        media_cache.record_audio(URL, audio, "旧标题")
        media_cache.record_raw(URL, "第一版", "标题1", "zh")
        media_cache.record_raw(URL, "第二版", "标题2", "en")

        hit = media_cache.lookup_raw(URL)
        self.assertEqual(hit["raw_text"], "第二版")
        self.assertEqual(hit["video_title"], "标题2")
        self.assertEqual(hit["detected_language"], "en")
        self.assertEqual(media_cache.lookup_audio(URL)["path"], audio)

    # ---- audio 缓存 ----

    def test_record_and_lookup_audio_roundtrip(self):
        audio = self.tmp_dir / "audio_1234.wav"
        audio.write_bytes(b"wav")
        media_cache.record_audio(URL, audio, "视频标题")
        hit = media_cache.lookup_audio(URL)
        self.assertEqual(hit["path"], audio)
        self.assertEqual(hit["video_title"], "视频标题")

    def test_audio_file_deleted_prunes_and_misses(self):
        audio = self.tmp_dir / "audio_gone.wav"
        audio.write_bytes(b"wav")
        media_cache.record_audio(URL, audio)
        audio.unlink()
        self.assertIsNone(media_cache.lookup_audio(URL))
        # 索引里 audio_file 已清掉
        index = json.loads(
            (self.tmp_dir / "media_cache.json").read_text(encoding="utf-8")
        )
        entry = index[media_cache.cache_key(URL)]
        self.assertNotIn("audio_file", entry)

    def test_empty_raw_text_not_recorded(self):
        media_cache.record_raw(URL, "   ", "标题", "zh")
        self.assertIsNone(media_cache.lookup_raw(URL))


class _FakeVideoProcessor:
    def __init__(self):
        self.subtitle_calls = 0
        self.download_calls = 0

    async def fetch_subtitles(self, url, output_dir):
        self.subtitle_calls += 1
        return (None, "缓存测试视频", None)

    async def download_and_convert(self, url, output_dir, prefetched_title=None):
        self.download_calls += 1
        audio = media_cache.TEMP_DIR / "audio_fake0001.wav"
        audio.write_bytes(b"wav")
        return str(audio), "缓存测试视频"


class _FakeTranscriber:
    last_detected_language = None

    def __init__(self):
        self.transcribe_calls = 0

    async def transcribe(self, audio_path, spk_num=None):
        self.transcribe_calls += 1
        self.last_detected_language = "zh"
        return "# 视频转录\n\n**Detected Language:** zh\n\n## 转录内容\n\n正文内容"


class TaskFlowCacheTests(unittest.IsolatedAsyncioTestCase):
    """process_video_task 接线：同 URL 第二次任务命中 raw 缓存，零下载零转录。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)
        patchers = [
            patch.object(media_cache, "TEMP_DIR", self.tmp_dir),
            patch.object(media_cache, "INDEX_FILE", self.tmp_dir / "media_cache.json"),
            patch.dict(os.environ, {"MEDIA_CACHE": "1"}),
        ]
        for p in patchers:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self._tmp.cleanup)

        import main
        self.main = main
        self.fake_vp = _FakeVideoProcessor()
        self.fake_tr = _FakeTranscriber()
        self.pipeline_calls = []
        patchers = [
            patch.object(main, "video_processor", self.fake_vp),
            patch.object(main, "transcriber", self.fake_tr),
            patch.object(main, "TASKS_FILE", self.tmp_dir / "tasks.json"),
            patch.object(main, "tasks", {}),
            patch.object(main, "save_tasks", lambda t: None),
            patch.object(main, "broadcast_task_update", self._noop_broadcast),
            patch.object(main, "_run_post_extract_pipeline", self._fake_pipeline),
        ]
        for p in patchers:
            p.start()
            self.addCleanup(p.stop)

    async def _noop_broadcast(self, task_id, task):
        pass

    async def _fake_pipeline(self, **kwargs):
        self.pipeline_calls.append(kwargs)

    async def _run_task(self) -> dict:
        task_id = f"t{len(self.pipeline_calls)}-{os.urandom(4).hex()}"
        self.main.tasks[task_id] = {"status": "processing", "progress": 0}
        await self.main.process_video_task(task_id, URL, "zh")
        return self.main.tasks[task_id]

    async def test_second_run_hits_raw_cache(self):
        first = await self._run_task()
        self.assertEqual(self.fake_vp.download_calls, 1)
        self.assertEqual(self.fake_tr.transcribe_calls, 1)
        self.assertIsNone(first.get("cache_hit"))

        second = await self._run_task()
        self.assertEqual(self.fake_vp.download_calls, 1)      # 未再下载
        self.assertEqual(self.fake_tr.transcribe_calls, 1)   # 未再转录
        self.assertEqual(self.fake_vp.subtitle_calls, 1)     # 字幕检测也跳过
        self.assertEqual(second.get("cache_hit"), "raw")
        self.assertEqual(
            self.pipeline_calls[0]["raw_script"],
            self.pipeline_calls[1]["raw_script"],  # 两次进管线的转录完全一致
        )

    async def test_audio_cache_hit_when_asr_config_changed(self):
        await self._run_task()
        # 换 ASR 模型 → raw 失效，但音频复用（只重转录，不再下载）
        with patch.dict(os.environ, {"FUNASR_MODEL": "sensevoice"}):
            task = await self._run_task()
        self.assertEqual(self.fake_vp.download_calls, 1)
        self.assertEqual(self.fake_tr.transcribe_calls, 2)
        self.assertEqual(task.get("cache_hit"), "audio")

    def test_parse_spk_num_boundaries(self):
        parse = self.main._parse_spk_num
        self.assertIsNone(parse(""))
        self.assertIsNone(parse("  "))
        self.assertIsNone(parse("abc"))
        self.assertIsNone(parse("0"))
        self.assertIsNone(parse("31"))
        self.assertEqual(parse("2"), 2)
        self.assertEqual(parse(" 12 "), 12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
