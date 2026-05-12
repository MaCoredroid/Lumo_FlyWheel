def normalize_response_items(items):
    """Normalize Responses API items to internal event format.
    
    Handles Responses event semantics where content is an array of
    output_text objects with type and text fields.
    """
    events = []
    for item in items:
        item_type = item["type"]
        if item_type == "message":
            # Responses format: content is array of {type: "output_text", text: "..."}
            content = item["content"]
            if isinstance(content, list):
                text = "".join(
                    c["text"] for c in content if c.get("type") == "output_text"
                )
            else:
                text = content
            events.append({"kind": "assistant_text", "text": text})
        elif item_type == "tool_call":
            events.append(
                {
                    "kind": "tool_call",
                    "call_id": item["call_id"],
                    "name": item["name"],
                    "arguments": item["arguments"],
                }
            )
        elif item_type == "tool_result":
            events.append(
                {
                    "kind": "tool_result",
                    "call_id": item["call_id"],
                    "output": item["output"],
                }
            )
    return events
