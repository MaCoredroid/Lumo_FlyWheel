from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote


class ExportPathViolation(ValueError):
    pass


_DRIVE_LETTER_RE = re.compile(r"^[a-zA-Z]:")


def _normalize_requested_path(requested_path: str) -> str:
    if not isinstance(requested_path, str) or not requested_path:
        raise ExportPathViolation("requested_path must be a non-empty string")
    normalized = requested_path
    while True:
        decoded = unquote(normalized)
        if decoded == normalized:
            break
        normalized = decoded
    # Check for absolute / drive-qualified paths BEFORE normalizing separators
    if normalized.startswith("/") or _DRIVE_LETTER_RE.match(normalized):
        raise ExportPathViolation("blocked absolute or drive-qualified path")
    normalized = normalized.replace("\\", "/")
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    return normalized.lstrip("/")


def resolve_export_path(tenant_root: Path, requested_path: str) -> Path:
    normalized = _normalize_requested_path(requested_path)
    if ".." in normalized:
        raise ExportPathViolation("blocked suspicious export path")
    candidate = (tenant_root / normalized).resolve(strict=False)
    if not str(candidate).startswith(str(tenant_root.resolve())):
        raise ExportPathViolation("candidate escaped the tenant root")
    return candidate
