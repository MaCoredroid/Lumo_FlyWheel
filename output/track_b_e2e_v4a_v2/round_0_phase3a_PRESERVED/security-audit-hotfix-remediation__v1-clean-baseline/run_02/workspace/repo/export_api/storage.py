from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote


class ExportPathViolation(ValueError):
    pass


def _normalize_requested_path(requested_path: str) -> str:
    if not isinstance(requested_path, str) or not requested_path:
        raise ExportPathViolation("requested_path must be a non-empty string")
    # Decode iteratively until stable to catch double/triple encoding
    normalized = requested_path
    while True:
        next_norm = unquote(normalized).replace("\\", "/")
        while "//" in next_norm:
            next_norm = next_norm.replace("//", "/")
        next_norm = next_norm.lstrip("/")
        if next_norm == normalized:
            break
        normalized = next_norm
    return normalized


def resolve_export_path(tenant_root: Path, requested_path: str) -> Path:
    # Check for absolute paths before normalization
    if requested_path.startswith("/") or requested_path.startswith("\\"):
        raise ExportPathViolation("blocked suspicious export path")
    normalized = _normalize_requested_path(requested_path)
    # Check for path traversal components after full decoding
    if normalized.startswith("..") or "/.." in normalized or normalized.endswith("/.."):
        raise ExportPathViolation("blocked suspicious export path")
    # Block drive-qualified paths (e.g., C:/ or C:\)
    if len(normalized) >= 2 and normalized[1] == ":":
        raise ExportPathViolation("blocked suspicious export path")
    candidate = (tenant_root / normalized).resolve(strict=False)
    if not str(candidate).startswith(str(tenant_root.resolve())):
        raise ExportPathViolation("candidate escaped the tenant root")
    return candidate
