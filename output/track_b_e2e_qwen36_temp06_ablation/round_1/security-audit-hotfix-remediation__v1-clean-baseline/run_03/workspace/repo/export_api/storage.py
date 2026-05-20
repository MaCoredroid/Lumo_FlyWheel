from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote


class ExportPathViolation(ValueError):
    pass


def _normalize_requested_path(requested_path: str) -> str:
    if not isinstance(requested_path, str) or not requested_path:
        raise ExportPathViolation("requested_path must be a non-empty string")
    # Repeatedly decode until stable to catch double/multi-encoding
    current = requested_path
    while True:
        decoded = unquote(current)
        if decoded == current:
            break
        current = decoded
    # Block absolute paths and drive-letter paths before any stripping
    if current.startswith("/") or (len(current) >= 2 and current[1] == ":"):
        raise ExportPathViolation("blocked suspicious export path")
    normalized = current.replace("\\", "/")
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    return normalized.lstrip("/")


def resolve_export_path(tenant_root: Path, requested_path: str) -> Path:
    normalized = _normalize_requested_path(requested_path)
    # Block parent-directory traversal
    if normalized.startswith(".."):
        raise ExportPathViolation("blocked suspicious export path")
    parts = normalized.split("/")
    if ".." in parts:
        raise ExportPathViolation("blocked suspicious export path")
    candidate = (tenant_root / normalized).resolve(strict=False)
    try:
        candidate.relative_to(tenant_root.resolve())
    except ValueError:
        raise ExportPathViolation("candidate escaped the tenant root")
    return candidate
