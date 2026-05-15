from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote


class ExportPathViolation(ValueError):
    pass


def _normalize_requested_path(requested_path: str) -> str:
    if not isinstance(requested_path, str) or not requested_path:
        raise ExportPathViolation("requested_path must be a non-empty string")
    # Fully decode URL encoding iteratively to handle double/triple encoding
    normalized = requested_path
    while True:
        decoded = unquote(normalized)
        if decoded == normalized:
            break
        normalized = decoded
    # Normalize path separators
    normalized = normalized.replace("\\", "/")
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    return normalized


def resolve_export_path(tenant_root: Path, requested_path: str) -> Path:
    normalized = _normalize_requested_path(requested_path)
    # Reject absolute paths before any stripping
    if normalized.startswith("/"):
        raise ExportPathViolation("blocked suspicious export path")
    # Reject Windows drive-qualified paths (e.g., C:/ or C:\)
    if len(normalized) >= 2 and normalized[1] == ":":
        raise ExportPathViolation("blocked suspicious export path")
    # Strip leading slashes for relative path handling
    normalized = normalized.lstrip("/")
    # Check for path traversal segments that are ".."
    if ".." in normalized.split("/"):
        raise ExportPathViolation("blocked suspicious export path")
    candidate = (tenant_root / normalized).resolve(strict=False)
    if not str(candidate).startswith(str(tenant_root.resolve())):
        raise ExportPathViolation("candidate escaped the tenant root")
    return candidate
