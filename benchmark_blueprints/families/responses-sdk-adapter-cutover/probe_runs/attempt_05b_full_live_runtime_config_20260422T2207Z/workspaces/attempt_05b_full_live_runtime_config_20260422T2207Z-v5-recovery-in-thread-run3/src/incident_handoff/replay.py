import json


def serialize_events(events):
    return "\n".join(json.dumps(event, separators=(",", ":")) for event in events)


def replay_from_serialized(serialized):
    events = []
    for line in serialized.splitlines():
        if not line:
            continue
        if line.lstrip().startswith("{"):
            events.append(json.loads(line))
            continue

        parts = line.split("|")
        if parts[0] == "assistant":
            events.append({"kind": "assistant_text", "text": parts[1]})
        elif parts[0] == "tool_call":
            events.append(
                {
                    "kind": "tool_call",
                    "call_id": parts[1],
                    "name": parts[2],
                    "arguments": parts[3],
                }
            )
        elif parts[0] == "tool_result":
            events.append({"kind": "tool_result", "call_id": parts[1], "output": parts[2]})
    return events
