"""说话人感知的句级段落聚合测试（_paragraphs_from_sentence_info）。"""
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from transcriber import _FunASRBackend, _speaker_label, rename_speaker_labels  # noqa: E402


def _si(*items):
    """items: (text, start_ms, end_ms, spk_or_None) → funasr sentence_info 形状。"""
    return [
        {
            "sentence": t,
            "start": s,
            "end": e,
            **({"spk": spk} if spk is not None else {}),
        }
        for t, s, e, spk in items
    ]


class SpeakerParagraphTests(unittest.TestCase):
    def _run(self, si, **env):
        with patch.dict(os.environ, env):
            return _FunASRBackend._paragraphs_from_sentence_info(si)

    def test_speaker_change_breaks_paragraph(self):
        segs = self._run(_si(
            ("你好欢迎收听。", 0, 1000, 0),
            ("今天我们聊AI。", 1100, 2200, 0),
            ("谢谢，我很高兴来。", 2300, 3400, 1),  # 换人 → 新段
            ("先从大模型说起。", 3500, 4500, 1),
        ))
        self.assertEqual(len(segs), 2)
        self.assertTrue(segs[0].text.startswith("说话人A："))
        self.assertTrue(segs[1].text.startswith("说话人B："))
        self.assertIn("今天我们聊AI。", segs[0].text)
        self.assertIn("先从大模型说起。", segs[1].text)

    def test_no_spk_keeps_pause_rule(self):
        # 无 spk 标签：行为与旧逻辑一致（停顿阈值分段，无前缀）
        segs = self._run(_si(
            ("连续的两句话。", 0, 1000, None),
            ("间隔很近。", 1200, 2000, None),
            ("停顿很久之后。", 5000, 6000, None),
        ))
        self.assertEqual(len(segs), 2)
        self.assertFalse(segs[0].text.startswith("说话人"))

    def test_unlabeled_then_labeled_uses_first_spk(self):
        segs = self._run(_si(
            ("开场无标签。", 0, 1000, None),
            ("带标签的句子。", 1100, 2000, 2),
        ))
        self.assertEqual(len(segs), 1)
        self.assertTrue(segs[0].text.startswith("说话人C："))

    def test_char_limit_breaks_within_same_speaker(self):
        long_a = "字" * 200
        segs = self._run(_si(
            (long_a, 0, 1000, 0),
            (long_a, 1100, 2000, 0),
            (long_a, 2100, 3000, 0),
        ), FUNASR_PARAGRAPH_CHARS="300")
        self.assertEqual(len(segs), 2)
        for s in segs:
            self.assertTrue(s.text.startswith("说话人A："))

    def test_alternating_dialogue(self):
        segs = self._run(_si(*[
            (f"甲说第{i}句。", i * 1100, i * 1100 + 1000, i % 2)
            for i in range(6)
        ]))
        self.assertEqual(len(segs), 6)
        labels = [s.text.split("：", 1)[0] for s in segs]
        self.assertEqual(labels, ["说话人A", "说话人B"] * 3)

    def test_speaker_label_mapping(self):
        self.assertEqual(_speaker_label(0), "说话人A")
        self.assertEqual(_speaker_label(25), "说话人Z")
        self.assertEqual(_speaker_label(30), "说话人30")

    def test_empty_and_blank_sentences(self):
        segs = self._run(_si(
            ("", 0, 500, 0),
            ("正常句子。", 600, 1500, 0),
        ))
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0].text, "说话人A：正常句子。")


class PresetSpkNumTests(unittest.TestCase):
    """spk_num 传递：请求参数 > FUNASR_SPK_NUM 环境变量 > 不传（自动聚类）。"""

    class _FakeModel:
        def __init__(self):
            self.last_kwargs = None

        def generate(self, **kwargs):
            self.last_kwargs = kwargs
            return [{"text": "你好世界", "sentence_info": []}]

    def _backend(self):
        backend = _FunASRBackend()
        backend.model = self._FakeModel()
        backend._kind = "paraformer"
        return backend

    def test_param_overrides_env(self):
        with patch.dict(os.environ, {"FUNASR_SPK_NUM": "5"}):
            backend = self._backend()
            backend.transcribe_sync("x.wav", None, spk_num=2)
        self.assertEqual(backend.model.last_kwargs["preset_spk_num"], 2)

    def test_env_used_when_param_absent(self):
        with patch.dict(os.environ, {"FUNASR_SPK_NUM": "5"}):
            backend = self._backend()
            backend.transcribe_sync("x.wav", None)
        self.assertEqual(backend.model.last_kwargs["preset_spk_num"], 5)

    def test_absent_when_neither_set(self):
        with patch.dict(os.environ, {"FUNASR_SPK_NUM": ""}):
            backend = self._backend()
            backend.transcribe_sync("x.wav", None)
        self.assertNotIn("preset_spk_num", backend.model.last_kwargs)

    def test_invalid_env_ignored(self):
        with patch.dict(os.environ, {"FUNASR_SPK_NUM": "abc"}):
            backend = self._backend()
            backend.transcribe_sync("x.wav", None, spk_num=3)
        self.assertEqual(backend.model.last_kwargs["preset_spk_num"], 3)


class RenameSpeakerLabelsTests(unittest.TestCase):
    """说话人标签 → 真名 替换：基础替换 / 空保留 / 重复修改回退 / 清洗。"""

    def test_basic_rename(self):
        text = "说话人A：大家好。\n\n说话人B：谢谢邀请。"
        new, applied = rename_speaker_labels(text, {"A": "小辉博士", "B": "蓝小欢"})
        self.assertEqual(new, "小辉博士：大家好。\n\n蓝小欢：谢谢邀请。")
        self.assertEqual(applied, {"A": "小辉博士", "B": "蓝小欢"})

    def test_empty_name_keeps_label(self):
        new, applied = rename_speaker_labels("说话人A：正文", {"A": "  "})
        self.assertEqual(new, "说话人A：正文")
        self.assertEqual(applied, {})

    def test_colon_stripped_from_name(self):
        new, applied = rename_speaker_labels("说话人A：正文", {"A": "小辉：博士"})
        self.assertEqual(new, "小辉博士：正文")

    def test_label_without_colon_untouched(self):
        new, _ = rename_speaker_labels("说话人A说了很多", {"A": "小辉"})
        self.assertEqual(new, "说话人A说了很多")

    def test_reapply_with_previous(self):
        # 第一次应用
        text = "说话人A：第一句。"
        once, applied = rename_speaker_labels(text, {"A": "小辉博士"})
        self.assertEqual(once, "小辉博士：第一句。")
        # 改名（previous 还原旧名 → 再替换）
        twice, applied2 = rename_speaker_labels(once, {"A": "小辉"}, previous=applied)
        self.assertEqual(twice, "小辉：第一句。")
        self.assertEqual(applied2, {"A": "小辉"})

    def test_clear_reverts_to_label(self):
        once, applied = rename_speaker_labels("说话人A：正文", {"A": "小辉"})
        reverted, applied2 = rename_speaker_labels(once, {}, previous=applied)
        self.assertEqual(reverted, "说话人A：正文")
        self.assertEqual(applied2, {})

    def test_detect_speaker_labels(self):
        from transcriber import Transcriber
        t = Transcriber.__new__(Transcriber)  # 不触发后端构造
        labels = t.detect_speaker_labels("说话人B：x\n\n说话人A：y\n\n说话人B：z")
        self.assertEqual(labels, ["A", "B"])
        self.assertEqual(t.detect_speaker_labels("无标签正文"), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
