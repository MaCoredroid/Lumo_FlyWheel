from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from export_api.storage import ExportPathViolation, resolve_export_path


class HotfixRegressionTests(unittest.TestCase):
    """Regressions for the double-encoded path-traversal hotfix (APPSEC-EXP-17)."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "tenant-root"
        self.root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    # -- Legitimate paths must still work --

    def test_allows_nested_export(self) -> None:
        actual = resolve_export_path(self.root, "reports/partner-billing/2026-04/export.csv")
        expected = (self.root / "reports" / "partner-billing" / "2026-04" / "export.csv").resolve(strict=False)
        self.assertEqual(actual, expected)

    # -- Single-encoded traversal must be rejected --

    def test_rejects_percent_encoded_dotdot(self) -> None:
        with self.assertRaises(ExportPathViolation):
            resolve_export_path(self.root, "..%2fsecrets.csv")

    # -- Double-encoded traversal must be rejected (the hotfix) --

    def test_rejects_double_encoded_traversal(self) -> None:
        with self.assertRaises(ExportPathViolation):
            resolve_export_path(self.root, "%252e%252e%252fsecrets.csv")

    # -- Mixed-slash traversal must be rejected --

    def test_rejects_mixed_slash_traversal(self) -> None:
        with self.assertRaises(ExportPathViolation):
            resolve_export_path(self.root, "reports\\\\..\\\\..\\\\secrets.csv")

    # -- Absolute path must be rejected --

    def test_rejects_absolute_path(self) -> None:
        with self.assertRaises(ExportPathViolation):
            resolve_export_path(self.root, "/etc/passwd")

    # -- Drive-qualified path must be rejected --

    def test_rejects_drive_qualified_path(self) -> None:
        with self.assertRaises(ExportPathViolation):
            resolve_export_path(self.root, "C:\\\\temp\\\\secrets.csv")


if __name__ == "__main__":
    unittest.main()
