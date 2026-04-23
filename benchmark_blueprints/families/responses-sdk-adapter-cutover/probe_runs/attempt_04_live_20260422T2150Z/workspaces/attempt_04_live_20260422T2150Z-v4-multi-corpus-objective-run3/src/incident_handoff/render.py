def render_transcript(events):
    lines = []
    for event in events:
        kind = event["kind"]
        if kind == "assistant_text":
            lines.append(f"assistant: {event['text']}")
        elif kind == "tool_call":
            lines.append(f"tool_call({event['call_id']}): {event['name']} {event['arguments']}")
        elif kind == "tool_result":
            lines.append(f"tool_result({event['call_id']}): {event['output']}")
    return "\n".join(lines)
