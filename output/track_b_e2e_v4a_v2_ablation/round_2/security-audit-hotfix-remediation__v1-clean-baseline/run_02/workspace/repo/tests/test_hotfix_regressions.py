from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from export_api.storage import ExportPathViolation, resolve_export_path


class HotfixRegressionTests(unittest.TestCase):
    """Regression tests for export-path-traversal hotfix (APPSEC-EXP-17)."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "tenant-root"
        self.root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_legit_nested_export_allowed(self) -> None:
        """REQ-legit-nested: nested exports under tenant root must work."""
        actual = resolve_export_path(self.root, "reports/partner-billing/2026-04/export.csv")
        expected = (self.root / "reports" / "partner-billing" / "2026-04" / "export.csv").resolve(strict=False)
        self.assertEqual(actual, expected)

    def test_encoded_dotdot_blocked(self) -> None:
        """REQ-encoded-dotdot: percent-encoded parent traversal must be blocked."""
        with self.assertRaises(ExportPathViolation):
            resolve_export_path(self.root, "..%2fsecrets.csv")

    def test_double_encoded_traversal_blocked(self) -> None:
        """REQ-double-encoded: double-encoded traversal must be blocked."""
        with self.assertRaises(ExportPathViolation):
            resolve_export_path(self.root, "%252e%252e%252fsecrets.csv")

    def test_mixed_slash_traversal_blocked(self) -> None:
        """REQ-mixed-slash: mixed slash traversal must be blocked."""
        with self.assertRaises(ExportPathViolation):
            resolve_export_path(self.root, "reports\\..\\..\\secrets.csv")

    def test_absolute_path_blocked(self) -> None:
        """REQ-absolute: absolute Unix paths must be blocked."""
        with self.assertRaises(ExportPathViolation):
            resolve_export_path(self.root, "/etc/passwd")

    def test_drive_qualified_path_blocked(self) -> None:
        """REQ-drive-qualified: Windows drive-qualified paths must be blocked."""
        with self.assertRaises(ExportPathViolation):
            resolve_export_path(self.root, "C:\\temp\\secrets.csv")


if __name__ == "__main__":
    unittest.main()
