from __future__ import annotations

from preview.config import PreviewConfig
from preview.runtime import preview_runtime_branch


def build_preview_plan(config: PreviewConfig) -> dict[str, object]:
    branch = preview_runtime_branch(config)
    return {
        "branch": branch,
        "shadow_enabled": config.shadow_enabled,
        "effective_mode": config.effective_mode,
        "force_legacy_seen": config.force_legacy_seen,
    }
