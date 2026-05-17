def serialize_events(events):
    """
    Serialize Responses events to transcript format.
 Preserves event ordering and tool-result correlation via call_id.
    """
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
    """
    Deserialize transcript back to Responses events.
 Event-sourced replay preserves:
    - Exact event ordering from original transcript
    - Tool-result correlation via call_id
    """
    events = []
    for line in serialized.splitlines():
        parts = line.split("|")
        if parts[0] == "assistant":
            events.append({"kind": "assistant_text", "text": parts[1]})
        elif parts[0] == "tool_call":
            events.append({
                "kind": "tool_call",
                "call_id": parts[1],
                "name": parts[2],
                "arguments": parts[3]
            })
        elif parts[0] == "tool_result":
            events.append({
                "kind": "tool_result",
                "call_id": parts[1],
                "output": parts[2]
            })
    return events
