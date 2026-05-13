def render_transcript(events):
    """
    Render events to a human-readable transcript.
    Events are rendered in their original sequence order.
    """
    lines = []
    for event in events:
        kind = event.get("kind", "")
        if kind == "assistant_text":
            lines.append(f"assistant: {event.get('text', '')}")
        elif kind == "tool_call":
            lines.append(f"tool_call({event.get('call_id', '')}): {event.get('name', '')} {event.get('arguments', '')}")
        elif kind == "tool_result":
            lines.append(f"tool_result({event.get('call_id',)}): {event.get('output', '')}")
    return "\n".join(lines)
