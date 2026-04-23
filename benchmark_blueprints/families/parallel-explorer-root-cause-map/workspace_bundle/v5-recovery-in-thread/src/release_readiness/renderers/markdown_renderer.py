from __future__ import annotations


def render_blocked_owner_section(summary: dict[str, object]) -> str:
    lines = ["## Blocked Owners"]
    for owner in summary["blocked_owners"]:
        lines.append(f"### {owner['display_owner']}")
        lines.append(f"- owner_key: {owner['owner_key']}")
        lines.append(f"- blocked_count: {owner['blocked_count']}")
        lines.append(f"- sources: {', '.join(owner['sources'])}")
    return "\n".join(lines)
