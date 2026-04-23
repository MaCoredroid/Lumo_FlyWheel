from __future__ import annotations

from gateway.adapter import normalize_events


def build_replay(path: str) -> list[dict]:
    transcript: list[dict] = []
    tool_calls: list[dict] = []
    results_by_call_id: dict[str | None, dict] = {}
    result_order: list[str | None] = []

    for item in normalize_events(path):
        kind = item.get("kind")
        if kind == "assistant_text":
            transcript.append({"type": "assistant", "text": item.get("text", "")})
            continue
        if kind == "tool_call":
            tool_calls.append(item)
            continue
        if kind == "tool_result":
            call_id = item.get("call_id")
            results_by_call_id[call_id] = item
            result_order.append(call_id)

    matched_call_ids: set[str | None] = set()
    for call in tool_calls:
        call_id = call.get("call_id")
        transcript.append(
            {
                "type": "tool_call",
                "tool_name": call.get("tool_name"),
                "arguments": call.get("arguments", ""),
                "call_id": call_id,
            }
        )
        result = results_by_call_id.get(call_id)
        if result is None:
            continue
        matched_call_ids.add(call_id)
        transcript.append(
            {
                "type": "tool_result",
                "tool_name": result.get("tool_name") or call.get("tool_name"),
                "output": result.get("output", ""),
                "call_id": call_id,
            }
        )

    for call_id in result_order:
        if call_id in matched_call_ids:
            continue
        result = results_by_call_id[call_id]
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
