from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass
class PreviewConfig:
    default_mode: str = "legacy"
    effective_mode: str = "legacy"
    shadow_enabled: bool = False
    force_legacy_seen: bool = False
    parser_hits: dict[str, str] = field(default_factory=dict)


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def load_preview_env(env: Mapping[str, str]) -> PreviewConfig:
    config = PreviewConfig()
    if _truthy(env.get("ENABLE_SHADOW_PREVIEW")):
        config.shadow_enabled = True
        config.effective_mode = "shadow"
        config.parser_hits["ENABLE_SHADOW_PREVIEW"] = "load_preview_env:ENABLE_SHADOW_PREVIEW"
    elif _truthy(env.get("ENABLE_PREVIEW_V2")):
        config.shadow_enabled = True
        config.effective_mode = "shadow"
        config.parser_hits["ENABLE_PREVIEW_V2"] = "load_preview_env:ENABLE_PREVIEW_V2->ENABLE_SHADOW_PREVIEW"
    if _truthy(env.get("PREVIEW_FORCE_LEGACY")):
        config.force_legacy_seen = True
        config.parser_hits["PREVIEW_FORCE_LEGACY"] = "load_preview_env:PREVIEW_FORCE_LEGACY"
    return config
