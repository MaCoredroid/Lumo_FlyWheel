def serialize_events(events):
    """Serialize events preserving order and tool-result correlation."""
    lines = []
    for idx, event in enumerate(events):
        kind = event["kind"]
        if kind == "assistant_text":
            lines.append(f"{idx}|{kind}|{event['text']}")
        elif kind == "tool_call":
            lines.append(f"{idx}|{kind}|{event['call_id']}|{event['name']}|{event['arguments']}")
        elif kind == "tool_result":
            lines.append(f"{idx}|{kind}|{event['call_id']}|{event['output']}")
    return "\n".join(lines)


def replay_from_serialized(serialized):
    """Replay events in order, correlating tool results via call_id."""
    events = []
    for line in serialized.splitlines():
        parts = line.split("|")
        idx = int(parts[0])
        kind = parts[1]
        if kind == "assistant_text":
            events.append((idx, {"kind": "assistant_text", "text": parts[2]}))
        elif kind == "tool_call":
            events.append((idx, {"kind": "tool_call", "call_id": parts[2], "name": parts[3], "arguments": parts[4]}))
        elif kind == "tool_result":
            events.append((idx, {"kind": "tool_result", "call_id": parts[2], "output": parts[3]}))
    sorted_events = sorted(events, key=lambda x: x[0])
    return [event for idx, event in sorted_events]
