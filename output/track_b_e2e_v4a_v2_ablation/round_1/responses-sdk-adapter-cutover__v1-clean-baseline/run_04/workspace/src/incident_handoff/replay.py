def serialize_events(events):
    """
    Serialize events for storage. Preserves event ordering and tool-result correlation.
    Each event is stored with its sequence index and call_id for correlation.
    """
    lines = []
    for idx, event in enumerate(events):
        kind = event["kind"]
        call_id = event.get("call_id", "")
        if kind == "assistant_text":
            lines.append(f"{idx}|assistant_text||{event['text']}")
        elif kind == "tool_call":
            lines.append(f"{idx}|tool_call|{call_id}|{event['name']}|{event['arguments']}")
        elif kind == "tool_result":
            lines.append(f"{idx}|tool_result|{call_id}|{event['output']}")
    return "\n".join(lines)


def replay_from_serialized(serialized):
    """
    Reconstruct events from serialized storage.
    Events are replayed in strict sequential order by their sequence index.
    Tool calls and results are correlated via call_id.
    """
    events = []
    for line in serialized.splitlines():
        parts = line.split("|", 4)
        seq_idx = int(parts[0])
        kind = parts[1]
        call_id = parts[2] if len(parts) > 2 else None
        
        if kind == "assistant_text":
            events.append({
                "seq": seq_idx,
                "kind": "assistant_text",
                "text": parts[3] if len(parts) > 3 else ""
            })
        elif kind == "tool_call":
            events.append({
                "seq": seq_idx,
                "kind": "tool_call",
                "call_id": call_id,
                "name": parts[3] if len(parts) > 3 else "",
                "arguments": parts[4] if len(parts) > 4 else ""
            })
        elif kind == "tool_result":
            events.append({
                "seq": seq_idx,
                "kind": "tool_result",
                "call_id": call_id,
                "output": parts[3] if len(parts) > 3 else ""
            })
    # Sort by sequence index to preserve event ordering
    events.sort(key=lambda e: e["seq"])
    # Remove seq field to match original event format
    for event in events:
        del event["seq"]
    return events
