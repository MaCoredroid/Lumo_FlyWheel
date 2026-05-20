from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote


class ExportPathViolation(ValueError):
    pass


def _normalize_requested_path(requested_path: str) -> str:
    if not isinstance(requested_path, str) or not requested_path:
        raise ExportPathViolation("requested_path must be a non-empty string")
    normalized = requested_path.replace("\\", "/")
    while True:
        decoded = unquote(normalized)
        if decoded == normalized:
            break
        normalized = decoded
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
        raise ExportPathViolation("blocked suspicious export path")
    normalized = normalized.lstrip("/")
    return normalized


def resolve_export_path(tenant_root: Path, requested_path: str) -> Path:
    normalized = _normalize_requested_path(requested_path)
    parts = normalized.split("/")
    if ".." in parts:
        raise ExportPathViolation("blocked suspicious export path")
    candidate = (tenant_root / normalized).resolve(strict=False)
    if not str(candidate).startswith(str(tenant_root.resolve())):
        raise ExportPathViolation("candidate escaped the tenant root")
    return candidate
