from __future__ import annotations


def render_markdown(report: dict) -> str:
    lines = ["# Watchlist Report", ""]
    lines.append("## Alerts")
    for item in report.get("alerts", []):
        lines.append(f"- {item['symbol']}: {item['action']} ({item['reason']})")
    lines.append("")
    return "\n".join(lines)
