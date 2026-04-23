from __future__ import annotations

from gateway.adapter import normalize_events


def build_replay(path: str) -> list[dict]:
    assistant_rows: list[dict] = []
    call_rows: list[dict] = []
    results_by_call_id: dict[str, dict] = {}

    for item in normalize_events(path):
        kind = item.get("kind")
        if kind == "assistant_text":
            assistant_rows.append(
                {
                    "sequence": item.get("sequence", 0),
                    "row": {"type": "assistant", "text": item.get("text", "")},
                }
            )
            continue
        if kind == "tool_result":
            call_id = item.get("call_id")
            if call_id is not None:
                results_by_call_id[call_id] = item
            continue
        if kind == "tool_call":
            call_rows.append(
                {
                    "sequence": item.get("sequence", 0),
                    "row": {
                        "type": "tool_call",
                        "tool_name": item.get("tool_name"),
                        "arguments": item.get("arguments", ""),
                        "call_id": item.get("call_id"),
                    },
                }
            )

    transcript: list[dict] = [entry["row"] for entry in sorted(assistant_rows, key=lambda entry: entry["sequence"])]

    for entry in call_rows:
        row = entry["row"]
        transcript.append(row)
        call_id = row.get("call_id")
        matched_result = results_by_call_id.get(call_id)
        if matched_result is not None:
            transcript.append(
                {
                    "type": "tool_result",
                    "tool_name": matched_result.get("tool_name"),
                    "output": matched_result.get("output", ""),
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
