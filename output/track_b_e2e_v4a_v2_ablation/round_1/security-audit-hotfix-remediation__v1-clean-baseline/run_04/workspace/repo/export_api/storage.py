from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote


class ExportPathViolation(ValueError):
    pass


def _normalize_requested_path(requested_path: str) -> str:
    if not isinstance(requested_path, str) or not requested_path:
        raise ExportPathViolation("requested_path must be a non-empty string")
    normalized = unquote(requested_path).replace("\\", "/")
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    # Iteratively decode until stable to catch double/triple-encoded traversal
    prev = None
    while prev != normalized:
        prev = normalized
        normalized = unquote(normalized)
    return normalized


def resolve_export_path(tenant_root: Path, requested_path: str) -> Path:
    normalized = _normalize_requested_path(requested_path)
    # Block absolute paths, parent traversal, and drive-qualified paths
    if normalized.startswith("/") or ".." in normalized:
        raise ExportPathViolation("blocked suspicious export path")
    # Block Windows drive letters (e.g., C:/ or C:\)
    if re.match(r"^[a-zA-Z]:/", normalized):
        raise ExportPathViolation("blocked suspicious export path")
    # Strip leading slashes for path joining
    normalized = normalized.lstrip("/")
    candidate = (tenant_root / normalized).resolve(strict=False)
    if not str(candidate).startswith(str(tenant_root.resolve())):
        raise ExportPathViolation("candidate escaped the tenant root")
    return candidate
