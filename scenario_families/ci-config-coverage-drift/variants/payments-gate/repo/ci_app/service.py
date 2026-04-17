from __future__ import annotations

from ci_app.policy import active_labels


def run_checks() -> list[str]:
    return active_labels()


def gate_snapshot() -> dict[str, list[str]]:
    required = active_labels()
    optional = [label for label in active_labels(include_optional=True) if label not in required]
    return {"required": required, "optional": optional}
