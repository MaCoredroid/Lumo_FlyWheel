def normalize_response_items(items):
    events = []
    for item in items:
        item_type = item.get("type") or item.get("kind")
        if item_type == "message" or item_type == "assistant_text":
            content = item.get("content") or item.get("text")
            if isinstance(content, list):
                text = "".join(c.get("text") for c in content if c.get("type") == "output_text")
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
