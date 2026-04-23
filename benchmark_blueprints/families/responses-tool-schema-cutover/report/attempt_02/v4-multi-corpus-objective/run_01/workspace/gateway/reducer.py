from __future__ import annotations

from gateway.adapter import normalize_events


def build_replay(path: str) -> list[dict]:
    transcript: list[dict] = []
    tool_calls: list[dict] = []
    pending_by_call_id: dict[str, list[dict]] = {}
    orphan_results: list[dict] = []

    for item in normalize_events(path):
        kind = item.get("kind")
        if kind == "assistant_text":
            transcript.append({"type": "assistant", "text": item.get("text", "")})
            continue
        if kind == "tool_result":
            call_id = item.get("call_id")
            if call_id:
                pending_by_call_id.setdefault(call_id, []).append(item)
            else:
                orphan_results.append(item)
            continue
        if kind == "tool_call":
            tool_calls.append(item)

    seen_call_ids: set[str] = set()
    for item in tool_calls:
        call_id = item.get("call_id")
        transcript.append(
            {
                "type": "tool_call",
                "tool_name": item.get("tool_name"),
                "arguments": item.get("arguments", ""),
                "call_id": call_id,
            }
        )
        if call_id:
            seen_call_ids.add(call_id)
        for result in pending_by_call_id.get(call_id, []):
            transcript.append(
                {
                    "type": "tool_result",
                    "tool_name": result.get("tool_name"),
                    "output": result.get("output", ""),
                    "call_id": result.get("call_id"),
                }
            )

    for call_id, results in pending_by_call_id.items():
        if call_id in seen_call_ids:
            continue
        for result in results:
            orphan_results.append(result)

    for result in sorted(orphan_results, key=lambda item: item.get("sequence", 0)):
        transcript.append(
            {
                "type": "tool_result",
                "tool_name": result.get("tool_name"),
                "output": result.get("output", ""),
                "call_id": result.get("call_id"),
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
