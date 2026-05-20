from __future__ import annotations

from pathlib import Path

from export_api.storage import ExportPathViolation, resolve_export_path


def build_export_path(tenant_root: Path, requested_path: str) -> Path:
    return resolve_export_path(Path(tenant_root), requested_path)
