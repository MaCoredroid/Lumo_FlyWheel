from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote


class ExportPathViolation(ValueError):
    pass


def _normalize_requested_path(requested_path: str) -> str:
    if not isinstance(requested_path, str) or not requested_path:
        raise ExportPathViolation("requested_path must be a non-empty string")
    # Handle double-encoding by repeatedly decoding until stable
    normalized = requested_path
    prev = ""
    while normalized != prev:
        prev = normalized
        normalized = unquote(normalized).replace("\\", "/")
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    return normalized


def resolve_export_path(tenant_root: Path, requested_path: str) -> Path:
    normalized = _normalize_requested_path(requested_path)
    # Reject absolute paths before any stripping
    if normalized.startswith("/"):
        raise ExportPathViolation("blocked suspicious export path")
    # Reject Windows drive-qualified paths (e.g., C:/, D:\)
    if re.match(r"^[a-zA-Z]:[/\\]", normalized):
        raise ExportPathViolation("blocked suspicious export path")
    # Strip leading slashes for relative path
    normalized = normalized.lstrip("/")
    # Check for path traversal components after full normalization
    if any(part == ".." for part in normalized.split("/")):
        raise ExportPathViolation("blocked suspicious export path")
    candidate = (tenant_root / normalized).resolve(strict=False)
    tenant_root_resolved = tenant_root.resolve()
    # Ensure candidate is under tenant_root with proper separator
    if not (str(candidate) == str(tenant_root_resolved) or str(candidate).startswith(str(tenant_root_resolved) + "/")):
        raise ExportPathViolation("candidate escaped the tenant root")
    return candidate
