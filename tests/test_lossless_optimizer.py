import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from lossless_optimizer import (  # noqa: E402
    assemble_segments,
    build_batches,
    evaluate_batch_response,
    split_source_segments,
    validate_segment_text,
)


class SourceSegmentationTests(unittest.TestCase):
    def test_segments_cover_source_exactly_once(self):
        text = "无标点中文正文" * 200
        segments = split_source_segments(text)
        self.assertEqual("".join(s.source_text for s in segments), text)
        self.assertEqual(segments[0].start, 0)
        self.assertEqual(segments[-1].end, len(text))
        for left, right in zip(segments, segments[1:]):
            self.assertEqual(left.end, right.start)
            self.assertNotEqual(left.id, right.id)

    def test_split_does_not_cut_ascii_token(self):
        token = "https://example.com/path?q=123"
        text = "中" * 290 + " " + token + " " + "文" * 320
        segments = split_source_segments(text)
        self.assertEqual("".join(s.source_text for s in segments), text)
        self.assertTrue(any(token in s.source_text for s in segments))

    def test_batches_do_not_repeat_core_segments(self):
        text = "甲乙丙丁" * 1000
        segments = split_source_segments(text)
        batches = build_batches(text, segments)
        self.assertEqual(
            [s.id for batch in batches for s in batch.segments],
            [s.id for s in segments],
        )
        self.assertTrue(all(len(batch.left_context) <= 200 for batch in batches))
        self.assertTrue(all(len(batch.right_context) <= 200 for batch in batches))


class FidelityValidationTests(unittest.TestCase):
    def test_punctuation_passes(self):
        result = validate_segment_text(
            "大家好我是小俊今天开始",
            "大家好，我是小俊。今天开始。",
        )
        self.assertTrue(result.accepted, result.reason)

    def test_small_same_length_typo_correction_passes(self):
        result = validate_segment_text("今天嘉宾是科粒移民", "今天嘉宾是柯丽一鸣")
        self.assertTrue(result.accepted, result.reason)

    def test_content_deletion_fails(self):
        result = validate_segment_text("主持人介绍完整内容", "主持人介绍内容")
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "content_deleted")

    def test_summary_rewrite_fails_even_when_long(self):
        source = "这是完整访谈原文包含开场寒暄引用举例以及详细论述" * 20
        candidate = "本段主要讨论访谈主题。" * 50
        self.assertFalse(validate_segment_text(source, candidate).accepted)

    def test_empty_fails(self):
        self.assertEqual(validate_segment_text("不可丢失", "").reason, "empty")


class ResponseAssemblyTests(unittest.TestCase):
    def test_bad_ids_fall_back_without_loss(self):
        text = "甲" * 300 + "乙" * 300 + "丙" * 300
        segments = split_source_segments(text)
        payload = {
            "segments": [
                {"id": segments[1].id, "text": segments[1].source_text},
                {"id": segments[0].id, "text": ""},
                {"id": segments[1].id, "text": segments[1].source_text},
                {"id": "unknown", "text": "invented"},
            ]
        }
        decisions = evaluate_batch_response(segments, payload)
        self.assertEqual(assemble_segments(segments, decisions).replace("\n", ""), text)
        self.assertEqual(decisions[segments[0].id].status, "fallback_empty")
        self.assertEqual(decisions[segments[1].id].status, "fallback_structure")
        self.assertEqual(decisions[segments[2].id].status, "fallback_structure")

    def test_response_order_cannot_change_output_order(self):
        text = "第一段。" * 60 + "第二段。" * 60
        segments = split_source_segments(text)
        payload = {
            "segments": [
                {"id": segment.id, "text": segment.source_text}
                for segment in reversed(segments)
            ]
        }
        decisions = evaluate_batch_response(segments, payload)
        self.assertEqual(assemble_segments(segments, decisions).replace("\n", ""), text)

    def test_fidelity_failure_falls_back_only_failed_segment(self):
        text = "保留第一段。" * 40 + "保留第二段。" * 40
        segments = split_source_segments(text)
        payload = {
            "segments": [
                {
                    "id": segment.id,
                    "text": "摘要内容" * 80 if index == 0 else segment.source_text,
                }
                for index, segment in enumerate(segments)
            ]
        }
        decisions = evaluate_batch_response(segments, payload)
        self.assertEqual(decisions[segments[0].id].status, "fallback_fidelity")
        self.assertEqual(assemble_segments(segments, decisions).replace("\n", ""), text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
