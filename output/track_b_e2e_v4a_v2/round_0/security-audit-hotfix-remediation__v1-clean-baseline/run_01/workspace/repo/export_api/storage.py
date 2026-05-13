from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote


class ExportPathViolation(ValueError):
    pass


def _normalize_requested_path(requested_path: str) -> str:
    if not isinstance(requested_path, str) or not requested_path:
        raise ExportPathViolation("requested_path must be a non-empty string")
    decoded = unquote(requested_path)
    while "%" in decoded:
        decoded = unquote(decoded)
    normalized = decoded.replace("\\", "/")
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    if normalized.startswith("/"):
        raise ExportPathViolation("blocked absolute path")
    if len(normalized) >= 2 and normalized[1] == ":":
        raise ExportPathViolation("blocked drive-qualified path")
    return normalized.lstrip("/")


def resolve_export_path(tenant_root: Path, requested_path: str) -> Path:
    normalized = _normalize_requested_path(requested_path)
    if any(p == ".." for p in normalized.split("/")):
        raise ExportPathViolation("blocked parent traversal")
    candidate = (tenant_root / normalized).resolve(strict=False)
    resolved_root = tenant_root.resolve(strict=False)
    if not str(candidate).startswith(str(resolved_root)):
        raise ExportPathViolation("candidate escaped the tenant root")
    return candidate
