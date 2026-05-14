from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote


class ExportPathViolation(ValueError):
    pass


def _normalize_requested_path(requested_path: str) -> str:
    if not isinstance(requested_path, str) or not requested_path:
        raise ExportPathViolation("requested_path must be a non-empty string")
    # Normalize separators
    normalized = requested_path.replace("\\", "/")
    # Fully decode URL encoding (handle double/triple encoding)
    prev = None
    while prev != normalized:
        prev = normalized
        normalized = unquote(normalized)
    # Collapse multiple slashes
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    return normalized


def resolve_export_path(tenant_root: Path, requested_path: str) -> Path:
    normalized = _normalize_requested_path(requested_path)
    # Reject absolute paths (before stripping leading slash)
    if normalized.startswith("/"):
        raise ExportPathViolation("blocked suspicious export path")
    # Reject Windows drive-qualified paths (e.g., C:/ or C:\)
    if len(normalized) >= 2 and normalized[1] == ":":
        raise ExportPathViolation("blocked suspicious export path")
    # Reject path traversal after full decoding
    if ".." in normalized:
        raise ExportPathViolation("blocked suspicious export path")
    # Strip leading slashes for path joining
    normalized = normalized.lstrip("/")
    candidate = (tenant_root / normalized).resolve(strict=False)
    # Ensure resolved path is strictly under tenant_root
    resolved_root = tenant_root.resolve(strict=False)
    if not str(candidate).startswith(str(resolved_root) + "/"):
        raise ExportPathViolation("candidate escaped the tenant root")
    return candidate
