from __future__ import annotations


def render_markdown(report: dict) -> str:
    lines = ["# Watchlist Report", ""]
    lines.append("## Alert Summary")
    lines.append(f"- Symbols covered: {report['summary']['total_symbols']}")
    lines.append(f"- Gainers: {', '.join(report['summary']['gainers'])}")
    lines.append(f"- Laggards: {', '.join(report['summary']['laggards'])}")
    lines.append("")
    lines.append("## Trade Alerts")
    for item in report.get("alerts", []):
        lines.append(f"- {item['symbol']}: {item['action']} ({item['reason']})")
    follow_up = report.get("follow_up")
    if follow_up:
        lines.append("")
        lines.append("## Watchlist Follow-Up")
        lines.append(f"- Watchlist: {', '.join(follow_up['watchlist'])}")
        lines.append(f"- Note: {follow_up['note']}")
    lines.append("")
    return "\n".join(lines)
