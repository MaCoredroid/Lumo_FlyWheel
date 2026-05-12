def serialize_events(events):
    lines = []
    for event in events:
        kind = event["kind"]
        if kind == "assistant_text":
            text = event['text'].replace('\n', '\\n').replace('|', '\x00')
            lines.append(f"assistant|{text}")
        elif kind == "tool_call":
            call_id = event['call_id'].replace('\n', '\\n').replace('|', '\x00')
            name = event['name'].replace('\n', '\\n').replace('|', '\x00')
            arguments = event['arguments'].replace('\n', '\\n').replace('|', '\x00')
            lines.append(f"tool_call|{call_id}|{name}|{arguments}")
        elif kind == "tool_result":
            call_id = event['call_id'].replace('\n', '\\n').replace('|', '\x00')
            output = event['output'].replace('\n', '\\n').replace('|', '\x00')
            lines.append(f"tool_result|{call_id}|{output}")
    return "\n".join(lines)


def replay_from_serialized(serialized):
    events = []
    for line in serialized.splitlines():
        parts = line.split("|")
        if parts[0] == "assistant":
            text = parts[1].replace('\x00', '|').replace('\\n', '\n')
            events.append({"kind": "assistant_text", "text": text})
        elif parts[0] == "tool_call":
            events.append({
                "kind": "tool_call",
                "call_id": parts[1].replace('\x00', '|').replace('\\n', '\n'),
                "name": parts[2].replace('\x00', '|').replace('\\n', '\n'),
                "arguments": parts[3].replace('\x00', '|').replace('\\n', '\n')
            })
        elif parts[0] == "tool_result":
            events.append({
                "kind": "tool_result",
                "call_id": parts[1].replace('\x00', '|').replace('\\n', '\n'),
                "output": parts[2].replace('\x00', '|').replace('\\n', '\n')
            })
    return events
