# -*- coding: utf-8 -*-
"""DashScope 云端 ASR 后端单测：结果解析 / 提交参数 / 切片合并（不触网）。"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from transcriber import _DashScopeBackend  # noqa: E402


def _make_backend() -> _DashScopeBackend:
    b = _DashScopeBackend.__new__(_DashScopeBackend)  # 跳过环境变量校验
    b.model = "qwen-audio-3.0-asr-flash-filetrans"
    b.base_url = "https://ws.cn-beijing.maas.aliyuncs.com/api/v1"
    b.api_key = "sk-test"
    return b


def _result(data, language=None):
    text, si, votes = _DashScopeBackend._collect_sentences(data)
    return _DashScopeBackend._to_transcript_result(text, si, votes, language)


class TestToTranscriptResult(unittest.TestCase):
    def test_sentences_to_paragraphs_and_language(self):
        data = {
            "transcripts": [{
                "text": "你好世界。今天我们聊AI。",
                "sentences": [
                    {"begin_time": 0, "end_time": 2000, "text": "你好世界。",
                     "speaker_id": 0, "language": "zh"},
                    # 停顿 4s > 默认 1.5s 阈值 → 应切成第二段
                    {"begin_time": 6000, "end_time": 9000, "text": "今天我们聊AI。",
                     "speaker_id": 0, "language": "zh"},
                ],
            }],
        }
        result = _result(data)
        self.assertEqual(result.text, "你好世界。今天我们聊AI。")
        self.assertEqual(result.detected_language, "zh")
        self.assertEqual(len(result.segments), 2)
        self.assertEqual(result.segments[0].start_fmt, "00:00")
        self.assertEqual(result.segments[1].start_fmt, "00:06")
        # 有 speaker_id 时段落带「说话人A：」前缀
        self.assertTrue(result.segments[0].text.startswith("说话人A："))

    def test_speaker_change_splits_paragraph(self):
        data = {
            "transcripts": [{
                "text": "",
                "sentences": [
                    {"begin_time": 0, "end_time": 1000, "text": "甲说。", "speaker_id": 0},
                    {"begin_time": 1000, "end_time": 2000, "text": "乙说。", "speaker_id": 1},
                ],
            }],
        }
        result = _result(data)
        self.assertEqual(len(result.segments), 2)
        self.assertTrue(result.segments[0].text.startswith("说话人A："))
        self.assertTrue(result.segments[1].text.startswith("说话人B："))

    def test_empty_transcripts(self):
        result = _result({"transcripts": []}, "zh")
        self.assertEqual(result.text, "")
        self.assertEqual(result.segments, [])
        self.assertEqual(result.detected_language, "zh")


class TestDiarizationDefault(unittest.TestCase):
    def setUp(self):
        self._old = os.environ.pop("DASHSCOPE_DIARIZATION", None)

    def tearDown(self):
        if self._old is not None:
            os.environ["DASHSCOPE_DIARIZATION"] = self._old
        else:
            os.environ.pop("DASHSCOPE_DIARIZATION", None)

    def test_default_on(self):
        self.assertTrue(_DashScopeBackend._diarization_enabled(None))

    def test_env_off(self):
        os.environ["DASHSCOPE_DIARIZATION"] = "0"
        self.assertFalse(_DashScopeBackend._diarization_enabled(None))

    def test_spk_num_forces_on(self):
        os.environ["DASHSCOPE_DIARIZATION"] = "0"
        self.assertTrue(_DashScopeBackend._diarization_enabled(2))


class TestSubmitParams(unittest.TestCase):
    def _capture_submit(self, backend, env, language=None, spk_num=None):
        captured = {}

        def fake_http(method, url, payload=None, extra_headers=None):
            captured["payload"] = payload
            captured["headers"] = extra_headers
            return {"output": {"task_id": "t-1"}}

        backend._http_json = fake_http
        keys = ("DASHSCOPE_LANGUAGE_HINTS", "DASHSCOPE_DIARIZATION")
        old = {k: os.environ.get(k) for k in keys}
        os.environ.update(env)
        try:
            backend._submit("https://oss.example/a.wav", language, spk_num)
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        return captured

    def test_sensitive_filter_disabled_and_async_header(self):
        backend = _make_backend()
        captured = self._capture_submit(
            backend, {"DASHSCOPE_LANGUAGE_HINTS": "zh,en", "DASHSCOPE_DIARIZATION": "0"})
        self.assertEqual(captured["headers"], {"X-DashScope-Async": "enable"})
        params = captured["payload"]["parameters"]
        self.assertEqual(params["special_word_filter"],
                         {"system_reserved_filter": False})
        self.assertEqual(params["language_hints"], ["zh", "en"])
        self.assertNotIn("diarization_enabled", params)

    def test_diarization_default_on(self):
        backend = _make_backend()
        captured = self._capture_submit(
            backend, {"DASHSCOPE_LANGUAGE_HINTS": ""})
        params = captured["payload"]["parameters"]
        self.assertTrue(params["diarization_enabled"])
        self.assertNotIn("speaker_count", params)  # 不填人数 → 自动聚类

    def test_spk_num_enables_speaker_count(self):
        backend = _make_backend()
        captured = self._capture_submit(
            backend, {"DASHSCOPE_LANGUAGE_HINTS": "", "DASHSCOPE_DIARIZATION": "0"},
            spk_num=2)
        params = captured["payload"]["parameters"]
        self.assertTrue(params["diarization_enabled"])
        self.assertEqual(params["speaker_count"], 2)


class TestMergeChunks(unittest.TestCase):
    """max_s=120s、overlap=30s（b=15s）的两片合并。片1 全局起点 step=90s。"""

    MAX_S, OVERLAP, B = 120.0, 30.0, 15000  # ms

    def _s(self, text, start_ms, spk=None):
        return {"sentence": text, "start": start_ms, "end": start_ms + 1000,
                "spk": spk}

    def test_seam_dedup_and_time_offset(self):
        chunk0 = [
            self._s("开头。", 0, 0),
            self._s("接缝前半。", 100_000, 0),      # < max_s-b=105s → 保留
            self._s("接缝截断版。", 110_000, 0),     # >= 105s → 丢弃（等片1的完整版）
        ]
        chunk1 = [
            self._s("重叠区头部。", 5_000, 0),       # < b=15s → 丢弃（片0已覆盖）
            self._s("接缝完整版。", 20_000, 0),      # 保留，全局 90s+20s=110s
        ]
        merged = _DashScopeBackend._merge_chunks(
            [chunk0, chunk1], [0.0, 90.0], self.MAX_S, self.OVERLAP)
        texts = [s["sentence"] for s in merged]
        self.assertEqual(texts, ["开头。", "接缝前半。", "接缝完整版。"])
        self.assertEqual(merged[2]["start"], 110_000)

    def test_speaker_remap_by_overlap_text(self):
        # 片0 结尾带（全局 90~105s，即片内 90~105s < 105s 边界）有说话人0的话
        chunk0 = [self._s("今天我们继续聊人工智能的发展趋势。", 95_000, 0)]
        # 片1 头部带（<15s）说话人0 说了同样的话 → 应映射到全局 0
        # 片1 主体里说话人1 是新人 → 应分配全局 1
        chunk1 = [
            self._s("今天我们继续聊人工智能的趋势。", 2_000, 0),
            self._s("片一正文甲。", 20_000, 0),
            self._s("片一正文乙。", 30_000, 1),
        ]
        merged = _DashScopeBackend._merge_chunks(
            [chunk0, chunk1], [0.0, 90.0], self.MAX_S, self.OVERLAP)
        spks = [(s["sentence"], s["spk"]) for s in merged]
        self.assertEqual(spks, [
            ("今天我们继续聊人工智能的发展趋势。", 0),
            ("片一正文甲。", 0),   # 跨片匹配成功
            ("片一正文乙。", 1),   # 新说话人拿到新编号
        ])

    def test_unmatched_head_speaker_gets_new_id(self):
        chunk0 = [self._s("片零正文。", 95_000, 0)]
        # 片1 头部说话人0 文本和前片完全不同 → 不匹配，分配新编号 1
        chunk1 = [
            self._s("完全不同的内容哈哈哈。", 2_000, 0),
            self._s("片一正文。", 20_000, 0),
        ]
        merged = _DashScopeBackend._merge_chunks(
            [chunk0, chunk1], [0.0, 90.0], self.MAX_S, self.OVERLAP)
        spks = [(s["sentence"], s["spk"]) for s in merged]
        self.assertEqual(spks, [("片零正文。", 0), ("片一正文。", 1)])


if __name__ == "__main__":
    unittest.main()
