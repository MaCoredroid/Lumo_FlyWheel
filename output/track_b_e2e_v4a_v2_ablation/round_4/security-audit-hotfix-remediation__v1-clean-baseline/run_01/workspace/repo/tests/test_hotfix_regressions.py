from __future__ import annotations

import json
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

    def _load_samples(self):
        return json.load(open("/workspace/artifacts/request_samples.json"))

    def test_legit_nested_export_allowed(self) -> None:
        """REQ-legit-nested: partner billing export must remain valid under tenant root."""
        result = resolve_export_path(self.root, "reports/partner-billing/2026-04/export.csv")
        expected = (self.root / "reports" / "partner-billing" / "2026-04" / "export.csv").resolve(strict=False)
        self.assertEqual(result, expected)

    def test_encoded_dotdot_blocked(self) -> None:
        """REQ-encoded-dotdot: percent-encoded parent traversal must be rejected."""
        with self.assertRaises(ExportPathViolation):
            resolve_export_path(self.root, "..%2fsecrets.csv")

    def test_double_encoded_blocked(self) -> None:
        """REQ-double-encoded: double-encoded traversal must be rejected after iterative decoding."""
        with self.assertRaises(ExportPathViolation):
            resolve_export_path(self.root, "%252e%252e%252fsecrets.csv")

    def test_mixed_slash_blocked(self) -> None:
        """REQ-mixed-slash: mixed backslash/forward-slash traversal must be rejected."""
        with self.assertRaises(ExportPathViolation):
            resolve_export_path(self.root, "reports\\\\..\\\\..\\\\secrets.csv")

    def test_absolute_path_blocked(self) -> None:
        """REQ-absolute: absolute path escapes must be rejected."""
        with self.assertRaises(ExportPathViolation):
            resolve_export_path(self.root, "/etc/passwd")

    def test_drive_qualified_blocked(self) -> None:
        """REQ-drive-qualified: Windows drive-qualified paths must be rejected."""
        with self.assertRaises(ExportPathViolation):
            resolve_export_path(self.root, "C:\\\\temp\\\\secrets.csv")

    def test_triple_encoded_blocked(self) -> None:
        """Triple-encoded traversal must also be rejected."""
        with self.assertRaises(ExportPathViolation):
            resolve_export_path(self.root, "%25252e%25252e%25252fsecrets.csv")

    def test_symlink_escape_blocked(self) -> None:
        """Path traversal via symlink escapes must be rejected."""
        # Create a symlink outside tenant root
        outside = Path(self._tmp.name) / "outside"
        outside.mkdir()
        (outside / "secret.csv").write_text("secret")
        (self.root / "link").symlink_to(outside)
        
        # Attempting to traverse via symlink should fail
        with self.assertRaises(ExportPathViolation):
            resolve_export_path(self.root, "link/secret.csv")


if __name__ == "__main__":
    unittest.main()
