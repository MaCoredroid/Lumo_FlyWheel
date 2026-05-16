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


def _decode_until_stable(s: str) -> str:
    prev = None
    current = s
    while prev != current:
        prev = current
        current = unquote(current)
    return current


def resolve_export_path(tenant_root: Path, requested_path: str) -> Path:
    if not isinstance(requested_path, str) or not requested_path:
        raise ExportPathViolation("requested_path must be a non-empty string")
    
    raw = requested_path
    if raw.startswith("/") or (len(raw) >= 2 and raw[1] == ":"):
        raise ExportPathViolation("blocked suspicious export path")
    
    normalized = _normalize_requested_path(requested_path)
    fully_decoded = _decode_until_stable(normalized).replace("\\", "/")
    while "//" in fully_decoded:
        fully_decoded = fully_decoded.replace("//", "/")
    fully_decoded = fully_decoded.lstrip("/")
    if ".." in fully_decoded.split("/"):
        raise ExportPathViolation("blocked suspicious export path")
    candidate = (tenant_root / normalized).resolve(strict=False)
    if not str(candidate).startswith(str(tenant_root.resolve())):
        raise ExportPathViolation("candidate escaped the tenant root")
    return candidate
