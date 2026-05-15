from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from export_api.storage import ExportPathViolation, resolve_export_path


class HotfixRegressionTests(unittest.TestCase):
    """Regression tests for cnb55 export-path-traversal hotfix."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "tenant-root"
        self.root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_legit_nested_export(self) -> None:
        """REQ-legit-nested: nested tenant exports must still work."""
        actual = resolve_export_path(self.root, "reports/partner-billing/2026-04/export.csv")
        expected = (self.root / "reports" / "partner-billing" / "2026-04" / "export.csv").resolve(strict=False)
        self.assertEqual(actual, expected)

    def test_rejects_encoded_dotdot_traversal(self) -> None:
        """REQ-encoded-dotdot: percent-encoded parent traversal must be blocked."""
        with self.assertRaises(ExportPathViolation):
            resolve_export_path(self.root, "..%2fsecrets.csv")

    def test_rejects_double_encoded_traversal(self) -> None:
        """REQ-double-encoded: double-encoded traversal must be blocked."""
        with self.assertRaises(ExportPathViolation):
            resolve_export_path(self.root, "%252e%252e%252fsecrets.csv")

    def test_rejects_mixed_slash_traversal(self) -> None:
        """REQ-mixed-slash: mixed backslash/forward slash traversal must be blocked."""
        with self.assertRaises(ExportPathViolation):
            resolve_export_path(self.root, "reports\\\\..\\\\..\\\\secrets.csv")

    def test_rejects_absolute_path(self) -> None:
        """REQ-absolute: absolute path escapes must be blocked."""
        with self.assertRaises(ExportPathViolation):
            resolve_export_path(self.root, "/etc/passwd")

    def test_rejects_drive_qualified_path(self) -> None:
        """REQ-drive-qualified: Windows drive-qualified paths must be blocked."""
        with self.assertRaises(ExportPathViolation):
            resolve_export_path(self.root, "C:\\\\temp\\\\secrets.csv")


if __name__ == "__main__":
    unittest.main()
