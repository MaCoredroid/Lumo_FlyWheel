from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from export_api.storage import ExportPathViolation, resolve_export_path


class HotfixRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "tenant-root"
        self.root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_rejects_percent_encoded_parent_segments(self) -> None:
        with self.assertRaises(ExportPathViolation):
            resolve_export_path(self.root, "..%2fsecret.csv")

    def test_rejects_double_encoded_parent_segments(self) -> None:
        with self.assertRaises(ExportPathViolation):
            resolve_export_path(self.root, "%252e%252e%252fsecret.csv")

    def test_preserves_partner_nested_exports(self) -> None:
        actual = resolve_export_path(self.root, "reports/partner-billing/2026-04/export.csv")
        expected = (self.root / "reports" / "partner-billing" / "2026-04" / "export.csv").resolve(strict=False)
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
