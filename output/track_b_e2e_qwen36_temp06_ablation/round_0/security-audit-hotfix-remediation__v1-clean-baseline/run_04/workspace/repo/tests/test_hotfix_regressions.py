from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from export_api.storage import ExportPathViolation, resolve_export_path


class HotfixRegressionTests(unittest.TestCase):
    """Regression tests for the path traversal hotfix (APPSEC-EXP-17)."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "tenant-root"
        self.root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    # -- Legitimate paths must still work --

    def test_allows_nested_export(self) -> None:
        actual = resolve_export_path(
            self.root, "reports/partner-billing/2026-04/export.csv"
        )
        expected = (
            self.root / "reports" / "partner-billing" / "2026-04" / "export.csv"
        ).resolve(strict=False)
        self.assertEqual(actual, expected)

    # -- Encoded traversal must be blocked --

    def test_blocks_percent_encoded_dotdot(self) -> None:
        with self.assertRaises(ExportPathViolation):
            resolve_export_path(self.root, "..%2fsecrets.csv")

    def test_blocks_double_encoded_traversal(self) -> None:
        with self.assertRaises(ExportPathViolation):
            resolve_export_path(self.root, "%252e%252e%252fsecrets.csv")

    def test_blocks_mixed_slash_traversal(self) -> None:
        with self.assertRaises(ExportPathViolation):
            resolve_export_path(self.root, "reports\\..\\..\\secrets.csv")

    def test_blocks_absolute_path(self) -> None:
        with self.assertRaises(ExportPathViolation):
            resolve_export_path(self.root, "/etc/passwd")

    def test_blocks_drive_qualified_path(self) -> None:
        with self.assertRaises(ExportPathViolation):
            resolve_export_path(self.root, "C:\\temp\\secrets.csv")


if __name__ == "__main__":
    unittest.main()
