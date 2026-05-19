from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote


class ExportPathViolation(ValueError):
    pass


# Match drive-qualified paths like C:\ or C:/
_DRIVE_LETTER_RE = re.compile(r"^[A-Za-z]:[/\\]")


def _normalize_requested_path(requested_path: str) -> str:
    if not isinstance(requested_path, str) or not requested_path:
        raise ExportPathViolation("requested_path must be a non-empty string")
    normalized = unquote(requested_path).replace("\\", "/")
    # Repeatedly decode until stable to catch double/multi-encoding
    prev = ""
    while prev != normalized:
        prev = normalized
        normalized = unquote(normalized)
    normalized = normalized.replace("\\", "/")
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    return normalized


def resolve_export_path(tenant_root: Path, requested_path: str) -> Path:
    normalized = _normalize_requested_path(requested_path)
    # Block absolute paths (leading /) before stripping
    if normalized.startswith("/"):
        raise ExportPathViolation("blocked suspicious export path")
    # Block drive-qualified paths (e.g. C:/)
    if _DRIVE_LETTER_RE.match(normalized):
        raise ExportPathViolation("blocked suspicious export path")
    normalized = normalized.lstrip("/")
    # Block ".." as a path component
    parts = normalized.split("/")
    if any(part == ".." for part in parts):
        raise ExportPathViolation("blocked suspicious export path")
    candidate = (tenant_root / normalized).resolve(strict=False)
    if not str(candidate).startswith(str(tenant_root.resolve())):
        raise ExportPathViolation("candidate escaped the tenant root")
    return candidate
