from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote


class ExportPathViolation(ValueError):
    pass


_DRIVE_LETTER_RE = re.compile(r'^[A-Za-z]:')


def _normalize_requested_path(requested_path: str) -> str:
    if not isinstance(requested_path, str) or not requested_path:
        raise ExportPathViolation("requested_path must be a non-empty string")

    # Reject absolute and drive-qualified paths before any normalization
    if requested_path.startswith("/") or requested_path.startswith("\\"):
        raise ExportPathViolation("blocked suspicious export path")
    if _DRIVE_LETTER_RE.match(requested_path):
        raise ExportPathViolation("blocked suspicious export path")

    # Iteratively decode to catch double/triple encoding
    normalized = requested_path
    for _ in range(5):
        decoded = unquote(normalized).replace("\\", "/")
        if decoded == normalized:
            break
        normalized = decoded

    # Re-check after decoding in case decoding revealed absolute/drive paths
    if normalized.startswith("/") or normalized.startswith("\\"):
        raise ExportPathViolation("blocked suspicious export path")
    if _DRIVE_LETTER_RE.match(normalized):
        raise ExportPathViolation("blocked suspicious export path")

    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    return normalized.lstrip("/")


def resolve_export_path(tenant_root: Path, requested_path: str) -> Path:
    normalized = _normalize_requested_path(requested_path)
    if ".." in normalized:
        raise ExportPathViolation("blocked suspicious export path")
    tenant_root_resolved = tenant_root.resolve()
    candidate = (tenant_root / normalized).resolve(strict=False)
    if candidate != tenant_root_resolved and not str(candidate).startswith(str(tenant_root_resolved) + "/"):
        raise ExportPathViolation("candidate escaped the tenant root")
    return candidate
