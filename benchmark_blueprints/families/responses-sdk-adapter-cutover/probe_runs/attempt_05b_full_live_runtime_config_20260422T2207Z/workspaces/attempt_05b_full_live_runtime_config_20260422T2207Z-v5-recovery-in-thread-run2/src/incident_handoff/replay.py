import json


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
        else:
            lines.append(f"raw_event|{kind}|{json.dumps(event, sort_keys=True)}")
    return "\n".join(lines)


def replay_from_serialized(serialized):
    events = []
    for line in serialized.splitlines():
        if line.startswith("assistant|"):
            events.append({"kind": "assistant_text", "text": line.split("|", 1)[1]})
        elif line.startswith("tool_call|"):
            parts = line.split("|", 3)
            events.append(
                {
                    "kind": "tool_call",
                    "call_id": parts[1],
                    "name": parts[2],
                    "arguments": parts[3],
                }
            )
        elif line.startswith("tool_result|"):
            parts = line.split("|", 2)
            events.append({"kind": "tool_result", "call_id": parts[1], "output": parts[2]})
        elif line.startswith("raw_event|"):
            _, _, payload = line.split("|", 2)
            events.append(json.loads(payload))
    return events
