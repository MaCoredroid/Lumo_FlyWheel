from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote


class ExportPathViolation(ValueError):
    pass


def _normalize_requested_path(requested_path: str) -> str:
    if not isinstance(requested_path, str) or not requested_path:
        raise ExportPathViolation("requested_path must be a non-empty string")
    # Check for absolute paths before any normalization
    decoded_once = unquote(requested_path)
    if decoded_once.startswith("/") or (len(decoded_once) >= 2 and decoded_once[1] == ":"):
        raise ExportPathViolation("absolute path not allowed")
    # Fully decode: iterate until stable to handle double/multi-encoded paths
    normalized = requested_path
    while True:
        decoded = unquote(normalized).replace("\\", "/")
        while "//" in decoded:
            decoded = decoded.replace("//", "/")
        decoded = decoded.lstrip("/")
        if decoded == normalized:
            break
        normalized = decoded
    return normalized


def _has_traversal_components(path_str: str) -> bool:
    parts = Path(path_str).parts
    for part in parts:
        if part == ".." or part.startswith(".."):
            return True
    return False


def resolve_export_path(tenant_root: Path, requested_path: str) -> Path:
    normalized = _normalize_requested_path(requested_path)
    if _has_traversal_components(normalized):
        raise ExportPathViolation("blocked suspicious export path")
    candidate = (tenant_root / normalized).resolve(strict=False)
    if not str(candidate).startswith(str(tenant_root.resolve())):
        raise ExportPathViolation("candidate escaped the tenant root")
    return candidate
