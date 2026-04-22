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
        elif kind == "response_item":
            item = event["item"]
            item_type = item.get("type", "unknown")
            detail = item.get("summary") or item.get("text") or item_type
            lines.append(f"response_item({item_type}): {detail}")
    return "\n".join(lines)
