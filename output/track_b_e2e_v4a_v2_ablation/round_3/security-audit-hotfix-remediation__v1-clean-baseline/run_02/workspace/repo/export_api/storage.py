from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote


class ExportPathViolation(ValueError):
    pass


def _normalize_requested_path(requested_path: str) -> str:
    if not isinstance(requested_path, str) or not requested_path:
        raise ExportPathViolation("requested_path must be a non-empty string")
    # Normalize backslashes to forward slashes before decoding
    normalized = requested_path.replace("\\", "/")
    # Iteratively decode until stable to handle double/multi-encoding
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
    # Block absolute paths (before stripping leading slashes)
    if normalized.startswith("/"):
        raise ExportPathViolation("blocked suspicious export path")
    # Block Windows drive-qualified paths (e.g., C:/ or C:)
    if len(normalized) >= 2 and normalized[1] == ":":
        raise ExportPathViolation("blocked suspicious export path")
    # Strip leading slashes for path joining
    normalized = normalized.lstrip("/")
    # Block parent traversal components
    if ".." in normalized.split("/"):
        raise ExportPathViolation("blocked suspicious export path")
    # Build candidate path and resolve to canonical form
    candidate = (tenant_root / normalized).resolve(strict=False)
    root_resolved = tenant_root.resolve()
    # Use path containment check: candidate must be under root
    try:
        candidate.relative_to(root_resolved)
    except ValueError:
        raise ExportPathViolation("candidate escaped the tenant root")
    return candidate
