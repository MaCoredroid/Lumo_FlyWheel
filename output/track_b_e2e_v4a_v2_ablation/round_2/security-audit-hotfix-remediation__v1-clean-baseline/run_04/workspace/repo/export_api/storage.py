from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from urllib.parse import unquote


class ExportPathViolation(ValueError):
    pass


# Pattern to detect Windows drive letters (e.g., C:, D:)
_DRIVE_LETTER_PATTERN = re.compile(r'^[A-Za-z]:', re.ASCII)


def _normalize_requested_path(requested_path: str) -> str:
    if not isinstance(requested_path, str) or not requested_path:
        raise ExportPathViolation("requested_path must be a non-empty string")
    # Repeatedly decode until stable to handle double/triple encoding
    prev = None
    normalized = requested_path
    while prev != normalized:
        prev = normalized
        normalized = unquote(normalized)
    # Reject absolute paths (after decoding)
    if normalized.startswith("/"):
        raise ExportPathViolation("absolute paths not allowed")
    # Reject Windows drive letters (e.g., C:, D:)
    if _DRIVE_LETTER_PATTERN.match(normalized):
        raise ExportPathViolation("drive-qualified paths not allowed")
    # Normalize separators to forward slash
    normalized = normalized.replace("\\", "/")
    # Collapse multiple slashes
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    # Strip leading slashes (defensive)
    normalized = normalized.lstrip("/")
    return normalized


def resolve_export_path(tenant_root: Path, requested_path: str) -> Path:
    normalized = _normalize_requested_path(requested_path)
    
    # Reject empty paths
    if not normalized:
        raise ExportPathViolation("empty path after normalization")
    
    # Reject paths containing .. segments (after full normalization)
    path_parts = PurePosixPath(normalized).parts
    if ".." in path_parts:
        raise ExportPathViolation("path traversal detected")
    
    # Build candidate path
    candidate = (tenant_root / normalized).resolve(strict=False)
    
    # Ensure the resolved path is under tenant_root
    resolved_root = tenant_root.resolve(strict=False)
    
    # Check if candidate is under the resolved root
    try:
        candidate.relative_to(resolved_root)
    except ValueError:
        raise ExportPathViolation("candidate escaped the tenant root")
    
    return candidate
