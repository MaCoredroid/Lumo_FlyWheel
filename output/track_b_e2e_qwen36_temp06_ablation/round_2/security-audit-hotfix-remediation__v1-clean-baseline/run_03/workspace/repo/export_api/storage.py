from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote


class ExportPathViolation(ValueError):
    pass


def _normalize_requested_path(requested_path: str) -> str:
    if not isinstance(requested_path, str) or not requested_path:
        raise ExportPathViolation("requested_path must be a non-empty string")
    # Repeatedly decode until stable to defeat double/triple encoding
    decoded = unquote(requested_path)
    while decoded != requested_path:
        requested_path = decoded
        decoded = unquote(requested_path)
    normalized = requested_path.replace("\\", "/")
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    return normalized


def resolve_export_path(tenant_root: Path, requested_path: str) -> Path:
    normalized = _normalize_requested_path(requested_path)
    # Block absolute paths and drive-qualified paths before stripping
    if normalized.startswith("/") or (len(normalized) >= 2 and normalized[1] == ":"):
        raise ExportPathViolation("blocked suspicious export path")
    # Strip leading slashes for relative joining, then check for ..
    normalized = normalized.lstrip("/")
    if ".." in normalized.split("/"):
        raise ExportPathViolation("blocked suspicious export path")
    candidate = (tenant_root / normalized).resolve(strict=False)
    resolved_root = tenant_root.resolve()
    if candidate != resolved_root and not str(candidate).startswith(str(resolved_root) + "/"):
        raise ExportPathViolation("candidate escaped the tenant root")
    return candidate
