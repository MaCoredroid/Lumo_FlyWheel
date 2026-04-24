from __future__ import annotations


def render_markdown(report: dict) -> str:
    lines = [
        "# Release Readiness",
        "",
        f"- ready: {report['ready']}",
        f"- services: {', '.join(report['services'])}",
    ]
    return "\n".join(lines)
