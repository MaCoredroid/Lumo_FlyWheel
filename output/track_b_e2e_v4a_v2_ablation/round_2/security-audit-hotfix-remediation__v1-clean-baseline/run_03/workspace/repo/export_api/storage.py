from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote


class ExportPathViolation(ValueError):
    pass


def _normalize_requested_path(requested_path: str) -> str:
    if not isinstance(requested_path, str) or not requested_path:
        raise ExportPathViolation("requested_path must be a non-empty string")
    # Decode fully to handle double-encoding attacks
    decoded = requested_path
    while decoded != unquote(decoded):
        decoded = unquote(decoded)
    # Normalize all separators to forward slashes
    normalized = decoded.replace(chr(92), "/")
    # Collapse multiple slashes
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    # Check for absolute path before stripping
    if normalized.startswith("/"):
        raise ExportPathViolation("absolute path not allowed")
    # Check for Windows drive letter (e.g., C:/ or C:temp)
    if len(normalized) >= 2 and normalized[1] == ":":
        raise ExportPathViolation("drive-qualified path not allowed")
    # Strip leading slashes (should be none remain after check)
    normalized = normalized.lstrip("/")
    return normalized


def resolve_export_path(tenant_root: Path, requested_path: str) -> Path:
    normalized = _normalize_requested_path(requested_path)
    # Block any remaining traversal attempts
    if normalized.startswith(".."):
        raise ExportPathViolation("blocked suspicious export path")
    # Build candidate path and resolve it
    candidate = (tenant_root / normalized).resolve(strict=False)
    tenant_resolved = tenant_root.resolve()
    # Ensure candidate is within tenant root using path components
    try:
        candidate.relative_to(tenant_resolved)
    except ValueError:
        raise ExportPathViolation("candidate escaped the tenant root")
    return candidate
