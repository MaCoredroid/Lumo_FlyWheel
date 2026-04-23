from __future__ import annotations

from gateway.adapter import normalize_events


def build_replay(path: str) -> list[dict]:
    items = normalize_events(path)
    transcript: list[dict] = []
    results_by_call_id: dict[str, dict] = {}
    orphan_results: list[dict] = []

    for item in items:
        if item.get("kind") != "tool_result":
            continue
        call_id = item.get("call_id")
        if call_id:
            results_by_call_id[call_id] = item
        else:
            orphan_results.append(item)

    for item in items:
        kind = item.get("kind")
        if kind == "assistant_text":
            transcript.append({"type": "assistant", "text": item.get("text", "")})
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
            if call_id in results_by_call_id:
                result = results_by_call_id.pop(call_id)
                transcript.append(
                    {
                        "type": "tool_result",
                        "tool_name": result.get("tool_name"),
                        "output": result.get("output", ""),
                        "call_id": call_id,
                    }
                )

    for result in orphan_results:
        transcript.append(
            {
                "type": "tool_result",
                "tool_name": result.get("tool_name"),
                "output": result.get("output", ""),
                "call_id": result.get("call_id"),
            }
        )

    for result in sorted(results_by_call_id.values(), key=lambda item: item.get("sequence", 0)):
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
