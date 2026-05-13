from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote


class ExportPathViolation(ValueError):
    pass


def _normalize_requested_path(requested_path: str) -> str:
    if not isinstance(requested_path, str) or not requested_path:
        raise ExportPathViolation("requested_path must be a non-empty string")

    # Reject absolute paths and drive-qualified paths before any normalization
    if requested_path.startswith("/") or (len(requested_path) >= 2 and requested_path[1] == ":"):
        raise ExportPathViolation("absolute paths are not allowed")

    # Repeatedly decode until stable to handle double/multi-encoded payloads
    normalized = requested_path
    while True:
        decoded = unquote(normalized).replace("\\", "/")
        while "//" in decoded:
            decoded = decoded.replace("//", "/")
        if decoded == normalized:
            break
        normalized = decoded

    # Reject parent traversal segments after full normalization
    parts = normalized.split("/")
    for part in parts:
        if part == "..":
            raise ExportPathViolation("parent traversal is not allowed")

    return normalized


def resolve_export_path(tenant_root: Path, requested_path: str) -> Path:
    normalized = _normalize_requested_path(requested_path)
    candidate = (tenant_root / normalized).resolve(strict=False)
    if not str(candidate).startswith(str(tenant_root.resolve())):
        raise ExportPathViolation("candidate escaped the tenant root")
    return candidate
