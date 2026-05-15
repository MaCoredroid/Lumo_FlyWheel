from __future__ import annotations

import re

from pathlib import Path
from urllib.parse import unquote


class ExportPathViolation(ValueError):
    pass


def _normalize_requested_path(requested_path: str) -> str:
    if not isinstance(requested_path, str) or not requested_path:
        raise ExportPathViolation("requested_path must be a non-empty string")
    # Iteratively decode to handle double/triple encoding
    normalized = unquote(requested_path)
    while "%" in normalized:
        decoded = unquote(normalized)
        if decoded == normalized:
            break
        normalized = decoded
    # Normalize path separators
    normalized = normalized.replace("\\", "/")
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    # Check for absolute path before stripping
    if normalized.startswith("/"):
        raise ExportPathViolation("absolute paths not allowed")
    # Reject drive-qualified paths (e.g., C:/ or C:\)
    if re.match(r"^[A-Za-z]:", normalized):
        raise ExportPathViolation("drive-qualified paths not allowed")
    return normalized.lstrip("/")


def resolve_export_path(tenant_root: Path, requested_path: str) -> Path:
    normalized = _normalize_requested_path(requested_path)
    # Reject paths containing .. components
    if ".." in normalized.split("/"):
        raise ExportPathViolation("blocked suspicious export path")
    candidate = (tenant_root / normalized).resolve(strict=False)
    # Ensure resolved path is under tenant root
    resolved_root = tenant_root.resolve()
    root_check = str(resolved_root) + "/"
    if not str(candidate).startswith(root_check):
        raise ExportPathViolation("candidate escaped the tenant root")
    return candidate
