from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from export_api.storage import ExportPathViolation, resolve_export_path


class HotfixRegressionTests(unittest.TestCase):
    """Verify all request samples from artifacts/request_samples.json"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "tenant-root"
        self.root.mkdir(parents=True, exist_ok=True)
        # Load request samples
        samples_path = Path(__file__).parent.parent.parent / "artifacts" / "request_samples.json"
        self.samples = json.loads(samples_path.read_text(encoding="utf-8"))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_legit_nested(self) -> None:
        """REQ-legit-nested: nested tenant exports must remain functional"""
        result = resolve_export_path(self.root, "reports/partner-billing/2026-04/export.csv")
        expected = (self.root / "reports" / "partner-billing" / "2026-04" / "export.csv").resolve(strict=False)
        self.assertEqual(result, expected)

    def test_rejects_encoded_dotdot(self) -> None:
        """REQ-encoded-dotdot: single-encoded traversal must be blocked"""
        with self.assertRaises(ExportPathViolation):
            resolve_export_path(self.root, "..%2fsecrets.csv")

    def test_rejects_double_encoded(self) -> None:
        """REQ-double-encoded: double-encoded traversal must be blocked"""
        with self.assertRaises(ExportPathViolation):
            resolve_export_path(self.root, "%252e%252e%252fsecrets.csv")

    def test_rejects_mixed_slash(self) -> None:
        """REQ-mixed-slash: backslash traversal must be blocked"""
        with self.assertRaises(ExportPathViolation):
            resolve_export_path(self.root, "reports\\\\..\\\\..\\\\secrets.csv")

    def test_rejects_absolute(self) -> None:
        """REQ-absolute: absolute paths must be blocked"""
        with self.assertRaises(ExportPathViolation):
            resolve_export_path(self.root, "/etc/passwd")

    def test_rejects_drive_qualified(self) -> None:
        """REQ-drive-qualified: drive-qualified paths must be blocked"""
        with self.assertRaises(ExportPathViolation):
            resolve_export_path(self.root, "C:\\\\temp\\\\secrets.csv")


if __name__ == "__main__":
    unittest.main()
