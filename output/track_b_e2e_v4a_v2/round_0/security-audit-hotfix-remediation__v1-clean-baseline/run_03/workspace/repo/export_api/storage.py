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
    # Reject absolute paths before stripping
    if normalized.startswith("/"):
        raise ExportPathViolation("blocked suspicious export path")
    # Reject Windows drive-qualified paths (e.g., C:/ or C:\)
    if len(normalized) >= 2 and normalized[1] == ":":
        raise ExportPathViolation("blocked suspicious export path")
    # Iteratively decode until stable to catch double/multi-encoded sequences
    while True:
        decoded = unquote(normalized)
        if decoded == normalized:
            break
        # Re-check absolute path after each decode iteration
        if decoded.startswith("/"):
            raise ExportPathViolation("blocked suspicious export path")
        # Re-check drive-qualified paths after decoding
        if len(decoded) >= 2 and decoded[1] == ":":
            raise ExportPathViolation("blocked suspicious export path")
        normalized = decoded
    return normalized.lstrip("/")


def resolve_export_path(tenant_root: Path, requested_path: str) -> Path:
    normalized = _normalize_requested_path(requested_path)
    if ".." in normalized:
        raise ExportPathViolation("blocked suspicious export path")
    candidate = (tenant_root / normalized).resolve(strict=False)
    if not str(candidate).startswith(str(tenant_root.resolve())):
        raise ExportPathViolation("candidate escaped the tenant root")
    return candidate
