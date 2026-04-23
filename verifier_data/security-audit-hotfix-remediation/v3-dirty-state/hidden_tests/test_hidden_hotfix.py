from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path.cwd()
if str(ROOT / "repo") not in sys.path:
    sys.path.insert(0, str(ROOT / "repo"))

from export_api.routes import build_export_path  # noqa: E402
from export_api.storage import ExportPathViolation  # noqa: E402


class HiddenHotfixTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "tenant-root"
        self.root.mkdir(parents=True, exist_ok=True)
        inside = self.root / "reports"
        inside.mkdir(parents=True, exist_ok=True)
        outside = Path(self._tmp.name) / "outside"
        outside.mkdir(parents=True, exist_ok=True)
        sneaky = self.root / "sneaky"
        if not sneaky.exists():
            sneaky.symlink_to(outside, target_is_directory=True)
        self.deploy = json.loads((ROOT / "repo" / "ops" / "deploy_hotfix.json").read_text(encoding="utf-8"))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def assertRejected(self, requested_path: str) -> None:
        with self.assertRaises(ExportPathViolation):
            build_export_path(self.root, requested_path)

    def test_rejects_encoded_variants(self) -> None:
        for requested in ("..%2fsecret.csv", "%252e%252e%252fsecret.csv", "reports\\..\\..\\secret.csv"):
            self.assertRejected(requested)

    def test_rejects_absolute_and_drive_paths(self) -> None:
        self.assertRejected("/etc/passwd")
        self.assertRejected("C:\\temp\\secret.csv")

    def test_rejects_symlink_escape(self) -> None:
        self.assertRejected("sneaky/leak.csv")

    def test_preserves_legitimate_nested_export(self) -> None:
        actual = build_export_path(self.root, "reports/partner-billing/2026-04/export.csv")
        expected = (self.root / "reports" / "partner-billing" / "2026-04" / "export.csv").resolve(strict=False)
        self.assertEqual(actual, expected)

    def test_deploy_note_is_grounded(self) -> None:
        claims = self.deploy.get("behavior_preservation_claims", [])
        self.assertTrue(any(row.get("sample_id") == "REQ-legit-nested" for row in claims))
        self.assertTrue(self.deploy.get("rollback_trigger"))
        self.assertTrue(self.deploy.get("impact_scope"))
        self.assertTrue(self.deploy.get("operator_verification_steps"))


if __name__ == "__main__":
    unittest.main()
