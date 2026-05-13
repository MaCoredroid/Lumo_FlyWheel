def normalize_response_items(events):
    """
    Normalize Responses API events into a unified format.
    Preserves event ordering and tool-result correlation via call_id.
    Handles both legacy message format and Responses event format.
    """
    normalized = []
    for event in events:
        event_type = event.get("type", "")
        
        # Handle Responses API format
        if event_type == "response.output_text":
            normalized.append({
                "kind": "assistant_text",
                "text": event.get("text", "")
            })
        elif event_type == "response.function_call":
            normalized.append({
                "kind": "tool_call",
                "call_id": event.get("call_id", ""),
                "name": event.get("name", ""),
                "arguments": event.get("arguments", "")
            })
        elif event_type == "response.function_call_output":
            normalized.append({
                "kind": "tool_result",
                "call_id": event.get("call_id", ""),
                "output": event.get("output", "")
            })
        # Handle legacy message format (for backward compatibility)
        elif event_type == "message":
            content = event.get("content", [])
            for c in content:
                if c.get("type") == "output_text":
                    normalized.append({
                        "kind": "assistant_text",
                        "text": c.get("text", "")
                    })
        elif event_type == "tool_call":
            normalized.append({
                "kind": "tool_call",
                "call_id": event.get("call_id", ""),
                "name": event.get("name", ""),
                "arguments": event.get("arguments", "")
            })
        elif event_type == "tool_result":
            normalized.append({
                "kind": "tool_result",
                "call_id": event.get("call_id", ""),
                "output": event.get("output", "")
            })
        # Handle function_call format (alternative naming)
        elif event_type == "function_call":
            normalized.append({
                "kind": "tool_call",
                "call_id": event.get("id", ""),
                "name": event.get("tool_name", ""),
                "arguments": event.get("arguments", "")
            })
        elif event_type == "function_call_output":
            normalized.append({
                "kind": "tool_result",
                "call_id": event.get("tool_call_id", ""),
                "output": event.get("output", "")
            })
    return normalized
