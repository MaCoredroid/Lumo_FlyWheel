import json


def serialize_events(events):
    return "\n".join(json.dumps(event, separators=(",", ":")) for event in events)


def replay_from_serialized(serialized):
    events = []
    for line in serialized.splitlines():
        if not line:
            continue
        if line.startswith("{"):
            events.append(json.loads(line))
            continue

        if line.startswith("assistant|"):
            _, text = line.split("|", 1)
            events.append({"kind": "assistant_text", "text": text})
        elif line.startswith("tool_call|"):
            _, call_id, name, arguments = line.split("|", 3)
            events.append(
                {
                    "kind": "tool_call",
                    "call_id": call_id,
                    "name": name,
                    "arguments": arguments,
                }
            )
        elif line.startswith("tool_result|"):
            _, call_id, output = line.split("|", 2)
            events.append(
                {
                    "kind": "tool_result",
                    "call_id": call_id,
                    "output": output,
                }
            )
    return events
