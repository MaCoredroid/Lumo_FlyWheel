from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from export_api.storage import ExportPathViolation, resolve_export_path


class HotfixRegressionTests(unittest.TestCase):
    """Regressions covering every request-sample from artifacts/request_samples.json."""

    @classmethod
    def setUpClass(cls) -> None:
        samples_path = Path(__file__).resolve().parents[2] / "artifacts" / "request_samples.json"
        cls.samples = json.loads(samples_path.read_text(encoding="utf-8"))

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "tenant"
        self.root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_legit_nested_allowed(self) -> None:
        """REQ-legit-nested: nested tenant export must resolve."""
        result = resolve_export_path(self.root, "reports/partner-billing/2026-04/export.csv")
        self.assertTrue(str(result).startswith(str(self.root.resolve())))

    def test_encoded_dotdot_blocked(self) -> None:
        """REQ-encoded-dotdot: percent-encoded .. must be blocked."""
        with self.assertRaises(ExportPathViolation):
            resolve_export_path(self.root, "..%2fsecrets.csv")

    def test_double_encoded_blocked(self) -> None:
        """REQ-double-encoded: double-percent-encoded .. must be blocked."""
        with self.assertRaises(ExportPathViolation):
            resolve_export_path(self.root, "%252e%252e%252fsecrets.csv")

    def test_mixed_slash_blocked(self) -> None:
        """REQ-mixed-slash: backslash .. traversal must be blocked."""
        with self.assertRaises(ExportPathViolation):
            resolve_export_path(self.root, "reports\\..\\..\\secrets.csv")

    def test_absolute_blocked(self) -> None:
        """REQ-absolute: absolute paths must be blocked."""
        with self.assertRaises(ExportPathViolation):
            resolve_export_path(self.root, "/etc/passwd")

    def test_drive_qualified_blocked(self) -> None:
        """REQ-drive-qualified: drive-qualified paths must be blocked."""
        with self.assertRaises(ExportPathViolation):
            resolve_export_path(self.root, "C:\\temp\\secrets.csv")


if __name__ == "__main__":
    unittest.main()
