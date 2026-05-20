from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote


class ExportPathViolation(ValueError):
    pass


_DRIVE_LETTER_RE = re.compile(r"^[A-Za-z]:")


def _normalize_requested_path(requested_path: str) -> str:
    if not isinstance(requested_path, str) or not requested_path:
        raise ExportPathViolation("requested_path must be a non-empty string")
    if requested_path.startswith("/") or _DRIVE_LETTER_RE.match(requested_path):
        raise ExportPathViolation("blocked suspicious export path")
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


def resolve_export_path(tenant_root: Path, requested_path: str) -> Path:
    normalized = _normalize_requested_path(requested_path)
    for part in normalized.split("/"):
        if part == "..":
            raise ExportPathViolation("blocked suspicious export path")
    candidate = (tenant_root / normalized).resolve(strict=False)
    root_resolved = tenant_root.resolve()
    if not (str(candidate).startswith(str(root_resolved) + "/") or candidate == root_resolved):
        raise ExportPathViolation("candidate escaped the tenant root")
    return candidate
