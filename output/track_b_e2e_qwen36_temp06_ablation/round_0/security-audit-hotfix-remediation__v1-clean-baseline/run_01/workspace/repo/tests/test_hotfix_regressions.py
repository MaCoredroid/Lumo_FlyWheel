from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from export_api.storage import ExportPathViolation, resolve_export_path


class HotfixRegressionTests(unittest.TestCase):
    """Regression tests for the export-path-traversal hotfix (SARIF-PY-TRAVERSAL-001)."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "tenant-root"
        self.root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    # --- Samples from artifacts/request_samples.json ---

    def test_REQ_legit_nested_allowed(self) -> None:
        """Partner billing export must remain valid under the tenant root."""
        actual = resolve_export_path(self.root, "reports/partner-billing/2026-04/export.csv")
        expected = (self.root / "reports" / "partner-billing" / "2026-04" / "export.csv").resolve(strict=False)
        self.assertEqual(actual, expected)

    def test_REQ_encoded_dotdot_rejected(self) -> None:
        """Percent-encoded parent traversal must be blocked."""
        with self.assertRaises(ExportPathViolation):
            resolve_export_path(self.root, "..%2fsecrets.csv")

    def test_REQ_double_encoded_rejected(self) -> None:
        """Double-encoded separators must be blocked after iterative decoding."""
        with self.assertRaises(ExportPathViolation):
            resolve_export_path(self.root, "%252e%252e%252fsecrets.csv")

    def test_REQ_mixed_slash_rejected(self) -> None:
        """Mixed backslash traversal must be normalized and blocked."""
        with self.assertRaises(ExportPathViolation):
            resolve_export_path(self.root, "reports\\..\\..\\secrets.csv")

    def test_REQ_absolute_rejected(self) -> None:
        """Absolute path escape must be blocked."""
        with self.assertRaises(ExportPathViolation):
            resolve_export_path(self.root, "/etc/passwd")

    def test_REQ_drive_qualified_rejected(self) -> None:
        """Drive-qualified path escape must be blocked."""
        with self.assertRaises(ExportPathViolation):
            resolve_export_path(self.root, "C:\\temp\\secrets.csv")


if __name__ == "__main__":
    unittest.main()
