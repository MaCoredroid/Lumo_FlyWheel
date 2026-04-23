from __future__ import annotations

from gateway.adapter import normalize_events


def build_replay(path: str) -> list[dict]:
    normalized = normalize_events(path)
    transcript: list[dict] = []
    pending_by_call_id: dict[str | None, dict] = {
        item.get("call_id"): item
        for item in normalized
        if item.get("kind") == "tool_result"
    }
    for item in normalized:
        kind = item.get("kind")
        if kind == "assistant_text":
            transcript.append({"type": "assistant", "text": item.get("text", "")})
            continue
        if kind == "tool_result":
            continue
        if kind == "tool_call":
            call_id = item.get("call_id")
            transcript.append(
                {
                    "type": "tool_call",
                    "tool_name": item.get("tool_name"),
                    "arguments": item.get("arguments", ""),
                    "call_id": call_id,
                }
            )
            if call_id in pending_by_call_id:
                result = pending_by_call_id[call_id]
                transcript.append(
                    {
                        "type": "tool_result",
                        "tool_name": result.get("tool_name"),
                        "output": result.get("output", ""),
                        "call_id": call_id,
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
