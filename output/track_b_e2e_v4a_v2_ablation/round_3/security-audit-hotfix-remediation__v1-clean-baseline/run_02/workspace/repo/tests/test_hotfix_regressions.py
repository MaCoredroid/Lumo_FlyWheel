from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from export_api.storage import ExportPathViolation, resolve_export_path


class HotfixRegressionTests(unittest.TestCase):
    """Regression tests for path traversal hotfix cnb55.path-traversal.hotfix.2026.05.15"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "tenant-root"
        self.root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_allows_legit_nested_export(self) -> None:
        """REQ-legit-nested: partner billing export must remain valid under the tenant root"""
        actual = resolve_export_path(self.root, "reports/partner-billing/2026-04/export.csv")
        expected = (self.root / "reports" / "partner-billing" / "2026-04" / "export.csv").resolve(strict=False)
        self.assertEqual(actual, expected)

    def test_blocks_encoded_dotdot(self) -> None:
        """REQ-encoded-dotdot: percent-encoded parent traversal"""
        with self.assertRaises(ExportPathViolation):
            resolve_export_path(self.root, "..%2fsecrets.csv")

    def test_blocks_double_encoded(self) -> None:
        """REQ-double-encoded: double-encoded separators still represent traversal when decoded twice"""
        with self.assertRaises(ExportPathViolation):
            resolve_export_path(self.root, "%252e%252e%252fsecrets.csv")

    def test_blocks_mixed_slash_traversal(self) -> None:
        """REQ-mixed-slash: mixed slash traversal"""
        with self.assertRaises(ExportPathViolation):
            resolve_export_path(self.root, "reports\\..\\..\\secrets.csv")

    def test_blocks_absolute_path(self) -> None:
        """REQ-absolute: absolute path escape"""
        with self.assertRaises(ExportPathViolation):
            resolve_export_path(self.root, "/etc/passwd")

    def test_blocks_drive_qualified_path(self) -> None:
        """REQ-drive-qualified: drive-qualified path escape"""
        with self.assertRaises(ExportPathViolation):
            resolve_export_path(self.root, "C:\\temp\\secrets.csv")


if __name__ == "__main__":
    unittest.main()
