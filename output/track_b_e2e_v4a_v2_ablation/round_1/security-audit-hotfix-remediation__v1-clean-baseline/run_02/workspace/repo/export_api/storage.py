from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote


class ExportPathViolation(ValueError):
    pass


def _normalize_requested_path(requested_path: str) -> str:
    if not isinstance(requested_path, str) or not requested_path:
        raise ExportPathViolation("requested_path must be a non-empty string")
    
    # Reject absolute paths before any normalization
    if requested_path.startswith("/"):
        raise ExportPathViolation("blocked suspicious export path")
    # Reject drive-qualified paths (e.g., C:\, D:/)
    if re.match(r"^[A-Za-z]:[\\/]", requested_path):
        raise ExportPathViolation("blocked suspicious export path")
    # Repeatedly decode until stable to handle double/triple encoding
    normalized = requested_path
    while True:
        decoded = unquote(normalized)
        if decoded == normalized:
            break
        normalized = decoded
    # After decoding, re-check for absolute and drive-qualified paths
    if normalized.startswith("/"):
        raise ExportPathViolation("blocked suspicious export path")
    if re.match(r"^[A-Za-z]:[\\/]", normalized):
        raise ExportPathViolation("blocked suspicious export path")
    # Normalize all slashes to forward slashes
    normalized = normalized.replace("\\", "/")
    # Collapse multiple slashes
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    # Strip leading slashes to prevent absolute path interpretation
    normalized = normalized.lstrip("/")
    return normalized


def resolve_export_path(tenant_root: Path, requested_path: str) -> Path:
    normalized = _normalize_requested_path(requested_path)
    # Reject paths containing parent directory traversal
    if normalized.startswith("/") or ".." in normalized.split("/"):
        raise ExportPathViolation("blocked suspicious export path")
    # Build the export path and verify it stays within tenant_root
    candidate = (tenant_root / normalized).resolve(strict=False)
    tenant_resolved = tenant_root.resolve()
    if not str(candidate).startswith(str(tenant_resolved)):
        raise ExportPathViolation("candidate escaped the tenant root")
    return candidate
