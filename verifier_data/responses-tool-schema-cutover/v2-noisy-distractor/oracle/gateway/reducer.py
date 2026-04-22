from __future__ import annotations

from gateway.adapter import normalize_events


def build_replay(path: str) -> list[dict]:
    transcript: list[dict] = []
    call_positions: dict[str, int] = {}
    for item in normalize_events(path):
        kind = item.get("kind")
        if kind == "assistant_text":
            transcript.append({"type": "assistant", "text": item.get("text", "")})
            continue
        if kind == "tool_call":
            transcript.append(
                {
                    "type": "tool_call",
                    "tool_name": item.get("tool_name"),
                    "arguments": item.get("arguments", ""),
                    "call_id": item.get("call_id"),
                }
            )
            call_positions[item.get("call_id")] = len(transcript) - 1
            continue
        if kind != "tool_result":
            continue

        call_id = item.get("call_id")
        tool_result = {
            "type": "tool_result",
            "tool_name": item.get("tool_name"),
            "output": item.get("output", ""),
            "call_id": call_id,
        }
        if call_id not in call_positions:
            transcript.append(tool_result)
            continue

        insert_at = call_positions[call_id] + 1
        while insert_at < len(transcript):
            row = transcript[insert_at]
            if row.get("type") != "tool_result" or row.get("call_id") != call_id:
                break
            insert_at += 1
        transcript.insert(insert_at, tool_result)
        for seen_call_id, position in list(call_positions.items()):
            if position >= insert_at:
                call_positions[seen_call_id] = position + 1
    return transcript


def render_replay(path: str) -> str:
    lines: list[str] = []
    for item in build_replay(path):
        if item["type"] == "assistant":
            lines.append(f"assistant: {item['text']}")
        elif item["type"] == "tool_call":
            lines.append(f"tool_call[{item['call_id']}] {item['tool_name']} {item['arguments']}")
        elif item["type"] == "tool_result":
            lines.append(f"tool_result[{item['call_id']}] {item['tool_name']} => {item['output']}")
    return "\n".join(lines)


def render_cli_summary(path: str) -> str:
    return render_replay(path)
