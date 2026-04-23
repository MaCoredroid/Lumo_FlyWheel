from __future__ import annotations

from gateway.adapter import normalize_events


def build_replay(path: str) -> list[dict]:
    transcript: list[dict] = []
    calls_in_order: list[dict] = []
    results_by_call_id: dict[str | None, list[dict]] = {}

    for item in normalize_events(path):
        kind = item.get("kind")
        if kind == "assistant_text":
            transcript.append({"type": "assistant", "text": item.get("text", "")})
        elif kind == "tool_call":
            calls_in_order.append(item)
        elif kind == "tool_result":
            results_by_call_id.setdefault(item.get("call_id"), []).append(item)

    for call in calls_in_order:
        call_id = call.get("call_id")
        transcript.append(
            {
                "type": "tool_call",
                "tool_name": call.get("tool_name"),
                "arguments": call.get("arguments", ""),
                "call_id": call_id,
            }
        )
        for result in results_by_call_id.pop(call_id, []):
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
