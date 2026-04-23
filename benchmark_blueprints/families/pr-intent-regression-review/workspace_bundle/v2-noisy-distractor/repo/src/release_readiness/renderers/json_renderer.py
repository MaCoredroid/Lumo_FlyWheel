from __future__ import annotations


def render_json(report: dict) -> dict:
    return {
        "version": report["version"],
        "ready": report["ready"],
        "services": list(report["services"]),
    }
