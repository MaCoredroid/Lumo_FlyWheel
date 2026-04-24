from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from urllib.parse import unquote


class ExportPathViolation(ValueError):
    pass


_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:($|/)")
_RESIDUAL_ESCAPE = re.compile(r"%(2e|2f|5c)", re.IGNORECASE)


def _normalize_parts(requested_path: str) -> list[str]:
    if not isinstance(requested_path, str) or not requested_path.strip():
        raise ExportPathViolation("requested_path must be a non-empty string")

    decoded_once = unquote(requested_path.strip())
    normalized = decoded_once.replace("\\", "/")
    while "//" in normalized:
        normalized = normalized.replace("//", "/")

    if normalized.startswith("/"):
        raise ExportPathViolation("absolute export paths are not allowed")
    if _WINDOWS_DRIVE.match(normalized):
        raise ExportPathViolation("drive-qualified export paths are not allowed")
    if _RESIDUAL_ESCAPE.search(normalized):
        raise ExportPathViolation("encoded traversal operators are not allowed")

    parts: list[str] = []
    for part in PurePosixPath(normalized).parts:
        if part in ("", "."):
            continue
        if part == "..":
            raise ExportPathViolation("parent traversal is not allowed")
        parts.append(part)

    if not parts:
        raise ExportPathViolation("requested_path resolved to an empty export path")
    return parts


def resolve_export_path(tenant_root: Path, requested_path: str) -> Path:
    root = tenant_root.resolve()
    candidate = root.joinpath(*_normalize_parts(requested_path)).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ExportPathViolation("candidate escaped the tenant root") from exc
    return candidate
