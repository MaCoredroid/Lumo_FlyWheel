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
        event_type, *parts = line.split("|")
        if event_type == "assistant":
            events.append({"kind": "assistant_text", "text": "|".join(parts)})
        elif event_type == "tool_call":
            call_id, name, arguments = line.split("|", 3)[1:]
            events.append(
                {
                    "kind": "tool_call",
                    "call_id": call_id,
                    "name": name,
                    "arguments": arguments,
                }
            )
        elif event_type == "tool_result":
            call_id, output = line.split("|", 2)[1:]
            events.append({"kind": "tool_result", "call_id": call_id, "output": output})
    return events
