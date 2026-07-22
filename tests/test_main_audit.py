import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from main import _write_optimization_audit  # noqa: E402


class OptimizationAuditPersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_writes_json_without_changing_payload(self):
        audit = {
            "model": "test-model",
            "status_counts": {"optimized": 2, "fallback_empty": 1},
            "segments": [{"id": "seg-1", "source_hash": "abc"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.json"
            await _write_optimization_audit(path, audit)
            saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(saved, audit)


if __name__ == "__main__":
    unittest.main(verbosity=2)
