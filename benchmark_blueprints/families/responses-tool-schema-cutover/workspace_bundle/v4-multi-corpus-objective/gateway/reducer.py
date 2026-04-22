from __future__ import annotations

from gateway.adapter import normalize_events


def build_replay(path: str) -> list[dict]:
    transcript: list[dict] = []
    pending_by_tool: dict[str, str] = {}
    for item in normalize_events(path):
        kind = item.get("kind")
        if kind == "assistant_text":
            transcript.append({"type": "assistant", "text": item.get("text", "")})
            continue
        if kind == "tool_result":
            pending_by_tool[item.get("tool_name")] = item.get("output", "")
            continue
        if kind == "tool_call":
            tool_name = item.get("tool_name")
            if any(row.get("tool_name") == tool_name for row in transcript):
                continue
            transcript.append(
                {
                    "type": "tool_call",
                    "tool_name": tool_name,
                    "arguments": item.get("arguments", ""),
                    "call_id": item.get("call_id"),
                }
            )
            if tool_name in pending_by_tool:
                transcript.append(
                    {
                        "type": "tool_result",
                        "tool_name": tool_name,
                        "output": pending_by_tool[tool_name],
                        "call_id": item.get("call_id"),
                    }
                )
    return transcript


def render_replay(path: str) -> str:
    lines: list[str] = []
    for item in build_replay(path):
        if item["type"] == "assistant":
            lines.append(f"assistant: {item['text']}")
        elif item["type"] == "tool_call":
            lines.append(f"tool_call[{item.get('call_id')}] {item['tool_name']} {item['arguments']}")
        elif item["type"] == "tool_result":
            lines.append(f"tool_result[{item.get('call_id')}] {item['tool_name']} => {item['output']}")
    return "\n".join(lines)


def render_cli_summary(path: str) -> str:
    return render_replay(path)
