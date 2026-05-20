from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from export_api.routes import build_export_path
from export_api.storage import ExportPathViolation


class RouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "tenant-root"
        self.root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_rejects_literal_parent_segments(self) -> None:
        with self.assertRaises(ExportPathViolation):
            build_export_path(self.root, "../secret.csv")

    def test_allows_nested_export(self) -> None:
        actual = build_export_path(self.root, "reports/2026/q2.csv")
        expected = (self.root / "reports" / "2026" / "q2.csv").resolve(strict=False)
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
