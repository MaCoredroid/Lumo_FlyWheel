from __future__ import annotations

from preview.config import PreviewConfig


def preview_runtime_branch(config: PreviewConfig) -> str:
    if config.shadow_enabled:
        return "preview_runtime_branch:shadow_preview_path"
    return "preview_runtime_branch:legacy_preview_path"
