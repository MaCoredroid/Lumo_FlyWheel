def serialize_events(events):
    lines = []
    for event in events:
        kind = event["kind"]
        if kind == "assistant_text":
            lines.append(f"assistant|{event['text']}")
        elif kind == "tool_call":
            lines.append(f"tool_call|{event['call_id']}|{event['name']}|{event['arguments']}")
        elif kind == "tool_result":
            lines.append(f"tool_result|{event['call_id']}|{event['output']}")
    return "\n".join(lines)


def replay_from_serialized(serialized):
    events = []
    for line in serialized.splitlines():
        parts = line.split("|")
        if parts[0] == "assistant":
            events.append({"kind": "assistant_text", "text": parts[1]})
        elif parts[0] == "tool_call":
            events.append({"kind": "tool_call", "name": parts[2], "arguments": parts[3]})
        elif parts[0] == "tool_result":
            events.append({"kind": "tool_result", "output": parts[2]})
    return events
