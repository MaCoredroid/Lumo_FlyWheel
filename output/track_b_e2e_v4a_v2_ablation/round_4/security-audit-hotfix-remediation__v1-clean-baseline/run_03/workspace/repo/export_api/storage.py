from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote


class ExportPathViolation(ValueError):
    pass


def _normalize_requested_path(requested_path: str) -> str:
    if not isinstance(requested_path, str) or not requested_path:
        raise ExportPathViolation("requested_path must be a non-empty string")
    
    # Iteratively decode until stable to catch double/multi-encoded traversal
    normalized = requested_path
    while True:
        decoded = unquote(normalized)
        if decoded == normalized:
            break
        normalized = decoded
    
    # Check for absolute path escapes BEFORE stripping
    # Unix absolute paths
    if normalized.startswith("/"):
        raise ExportPathViolation("blocked suspicious export path")
    # Windows drive-qualified paths (e.g., C:\ or C:/)
    if len(normalized) >= 2 and normalized[1] == ":":
        raise ExportPathViolation("blocked suspicious export path")
    
    # Normalize separators (after full decoding)
    normalized = normalized.replace("\\", "/")
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    
    return normalized.lstrip("/")


def resolve_export_path(tenant_root: Path, requested_path: str) -> Path:
    normalized = _normalize_requested_path(requested_path)
    if normalized.startswith("/") or ".." in normalized:
        raise ExportPathViolation("blocked suspicious export path")
    candidate = (tenant_root / normalized).resolve(strict=False)
    if not str(candidate).startswith(str(tenant_root.resolve())):
        raise ExportPathViolation("candidate escaped the tenant root")
    return candidate
