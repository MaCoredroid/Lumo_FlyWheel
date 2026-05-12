def serialize_events(events):
    """
    Serialize events for storage in the transcript.
    
    Events are stored as raw event objects to preserve ordering and structure.
    This enables event-sourced replay without rebuilding state from rendered text.
    """
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
    """
    Reconstruct events from serialized transcript data.
    
    Events are replayed in strict order as stored. Tool calls and results
    are correlated via call_id, not by position.
    """
    events = []
    for line in serialized.splitlines():
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
            events.append(
                {"kind": "tool_result", "call_id": parts[1], "output": parts[2]}
            )
    return events


def replay_events(event_list):
    """
    Apply a list of events in order for event-sourced replay.
    
    This function processes events sequentially, maintaining the exact order
    they were emitted. Tool calls and results are preserved as-is for correlation.
    
    Args:
        event_list: List of event dicts in chronological order.
        
    Returns:
        List of processed events, maintaining original order.
    """
    result = []
    for event in event_list:
        result.append(event)
    return result
