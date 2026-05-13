from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from export_api.storage import ExportPathViolation, resolve_export_path


class StorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "tenant-root"
        self.root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_rejects_obvious_absolute_path(self) -> None:
        with self.assertRaises(ExportPathViolation):
            resolve_export_path(self.root, "/etc/passwd")

    def test_returns_candidate_under_root(self) -> None:
        actual = resolve_export_path(self.root, "exports/statement.csv")
        expected = (self.root / "exports" / "statement.csv").resolve(strict=False)
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
