from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote


class ExportPathViolation(ValueError):
    pass


def _normalize_requested_path(requested_path: str) -> str:
    if not isinstance(requested_path, str) or not requested_path:
        raise ExportPathViolation("requested_path must be a non-empty string")
    normalized = unquote(requested_path).replace("\\", "/")
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    return normalized.lstrip("/")


def _decode_until_stable(value: str) -> str:
    """Decode percent-encoding until no further changes occur."""
    prev: str | None = None
    current = value
    while prev is None or prev != current:
        prev = current
        current = unquote(current)
    return current


def _has_parent_escape(normalized: str) -> bool:
    """Check if normalized path contains parent directory traversal."""
    parts = normalized.split("/")
    return ".." in parts


def _is_absolute_path(path: str) -> bool:
    """Check if path is absolute (Unix or Windows style)."""
    if path.startswith("/"):
        return True
    if len(path) >= 2 and path[1] == ":":
        return True
    return False


def resolve_export_path(tenant_root: Path, requested_path: str) -> Path:
    if _is_absolute_path(requested_path):
        raise ExportPathViolation("blocked suspicious export path")
    normalized = _normalize_requested_path(requested_path)
    decoded = _decode_until_stable(normalized)
    if _has_parent_escape(decoded):
        raise ExportPathViolation("blocked suspicious export path")
    candidate = (tenant_root / decoded).resolve(strict=False)
    if not str(candidate).startswith(str(tenant_root.resolve())):
        raise ExportPathViolation("candidate escaped the tenant root")
    return candidate
