from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from export_api.storage import ExportPathViolation, resolve_export_path


class HotfixRegressionTests(unittest.TestCase):
    """Regression tests for the export-path-traversal hotfix (APPSEC-EXP-17)."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "tenant-root"
        self.root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_percent_encoded_dotdot_blocked(self) -> None:
        with self.assertRaises(ExportPathViolation):
            resolve_export_path(self.root, "..%2fsecrets.csv")

    def test_double_encoded_traversal_blocked(self) -> None:
        with self.assertRaises(ExportPathViolation):
            resolve_export_path(self.root, "reports/../../etc/passwd")

    def test_nested_export_allowed(self) -> None:
        actual = resolve_export_path(self.root, "a/b/c.csv")
        self.assertEqual(actual, (self.root / "a" / "b" / "c.csv").resolve(strict=False))


class RequestSampleTests(unittest.TestCase):
    """Validate against artifacts/request_samples.json."""

    @classmethod
    def _load_samples(cls) -> list[dict]:
        root = Path(__file__).resolve().parents[2] / "artifacts" / "request_samples.json"
        return json.loads(root.read_text(encoding="utf-8"))

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "tenant-root"
        self.root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_all_request_samples(self) -> None:
        for sample in self._load_samples():
            allowed = sample["allowed"]
            path = sample["requested_path"]
            try:
                resolve_export_path(self.root, path)
                raised = False
            except ExportPathViolation:
                raised = True
            if allowed and raised:
                self.fail(f"{sample['sample_id']}: expected allowed but was blocked")
            if not allowed and not raised:
                self.fail(f"{sample['sample_id']}: expected blocked but was allowed")


if __name__ == "__main__":
    unittest.main()
