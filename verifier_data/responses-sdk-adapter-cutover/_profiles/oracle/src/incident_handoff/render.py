import json


def render_transcript(events):
    lines = []
    for event in events:
        sequence = event.get("sequence", "?")
        kind = event["kind"]
        if kind == "assistant_text":
            lines.append(f"assistant[{sequence}]: {event['text']}")
        elif kind == "tool_call":
            lines.append(f"tool_call({event['call_id']})[{sequence}]: {event['name']} {event['arguments']}")
        elif kind == "tool_result":
            lines.append(f"tool_result({event['call_id']})[{sequence}]: {event['output']}")
        elif kind == "unknown_event":
            payload = json.dumps(event["payload"], sort_keys=True)
            lines.append(f"unknown_event({event['event_type']})[{sequence}]: {payload}")
    return "\n".join(lines)
