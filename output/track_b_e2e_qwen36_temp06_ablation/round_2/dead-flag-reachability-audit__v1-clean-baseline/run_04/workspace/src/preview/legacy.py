from __future__ import annotations


def legacy_force_label(force_legacy_seen: bool) -> str:
    return "legacy-forced" if force_legacy_seen else "legacy-default"
