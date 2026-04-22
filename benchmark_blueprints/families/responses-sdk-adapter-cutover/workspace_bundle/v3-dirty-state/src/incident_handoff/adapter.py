def normalize_response_items(items):
    events = []
    for item in items:
        item_type = item["type"]
        if item_type == "message":
            events.append({"kind": "assistant_text", "text": item["content"]})
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
