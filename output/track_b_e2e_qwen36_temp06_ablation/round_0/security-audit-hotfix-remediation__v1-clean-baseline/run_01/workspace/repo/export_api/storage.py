from __future__ import annotations
import re

from pathlib import Path
from urllib.parse import unquote


class ExportPathViolation(ValueError):
    pass


def _normalize_requested_path(requested_path: str) -> str:
    if not isinstance(requested_path, str) or not requested_path:
        raise ExportPathViolation("requested_path must be a non-empty string")
    # Repeatedly decode until stable so double-encoded payloads are caught.
    normalized = requested_path
    while True:
        decoded = unquote(normalized)
        if decoded == normalized:
            break
        normalized = decoded
    # Reject absolute paths and Windows drive-qualified paths before normalizing.
    if normalized.startswith("/") or re.match(r"^[a-zA-Z]:", normalized):
        raise ExportPathViolation("blocked suspicious export path")
    normalized = normalized.replace("\\", "/")
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    normalized = normalized.lstrip("/")
    return normalized


def resolve_export_path(tenant_root: Path, requested_path: str) -> Path:
    normalized = _normalize_requested_path(requested_path)
    segments = normalized.split("/")
    if ".." in segments:
        raise ExportPathViolation("blocked suspicious export path")
    candidate = (tenant_root / normalized).resolve(strict=False)
    if not str(candidate).startswith(str(tenant_root.resolve())):
        raise ExportPathViolation("candidate escaped the tenant root")
    return candidate
