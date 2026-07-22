import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from transcriber import _FunASRBackend  # noqa: E402


class SenseVoiceLanguageTests(unittest.TestCase):
    def test_noise_tag_is_not_language(self):
        self.assertIsNone(_FunASRBackend._parse_language("<|woitn|>正文", {}))

    def test_known_tag_after_noise_is_found(self):
        self.assertEqual(
            _FunASRBackend._parse_language("<|HAPPY|><|woitn|><|zh|>正文", {}),
            "zh",
        )

    def test_known_iso_tag_is_kept(self):
        self.assertEqual(_FunASRBackend._parse_language("<|fr|>bonjour", {}), "fr")

    def test_result_language_is_used_when_tags_are_noise(self):
        self.assertEqual(
            _FunASRBackend._parse_language("<|woitn|>bonjour", {"language": "fr"}),
            "fr",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
