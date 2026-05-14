def serialize_events(events):
    lines = []
    for event in events:
        kind = event["kind"]
        if kind == "assistant_text":
            text = event["text"].replace("|", "\\|")
            lines.append(f"assistant|{text}")
        elif kind == "tool_call":
            call_id = event["call_id"].replace("|", "\\|")
            name = event["name"].replace("|", "\\|")
            arguments = event["arguments"].replace("|", "\\|")
            lines.append(f"tool_call|{call_id}|{name}|{arguments}")
        elif kind == "tool_result":
            call_id = event["call_id"].replace("|", "\\|")
            output = event["output"].replace("|", "\\|")
            lines.append(f"tool_result|{call_id}|{output}")
    return "\n".join(lines)


def replay_from_serialized(serialized):
    events = []
    for line in serialized.splitlines():
        parts = line.split("|")
        if parts[0] == "assistant":
            text = "|".join(parts[1:]).replace("\\|", "|")
            events.append({"kind": "assistant_text", "text": text})
        elif parts[0] == "tool_call":
            call_id = parts[1].replace("\\|", "|")
            name = parts[2].replace("\\|", "|")
            arguments = "|".join(parts[3:]).replace("\\|", "|")
            events.append({"kind": "tool_call", "call_id": call_id, "name": name, "arguments": arguments})
        elif parts[0] == "tool_result":
            call_id = parts[1].replace("\\|", "|")
            output = "|".join(parts[2:]).replace("\\|", "|")
            events.append({"kind": "tool_result", "call_id": call_id, "output": output})
    return events
