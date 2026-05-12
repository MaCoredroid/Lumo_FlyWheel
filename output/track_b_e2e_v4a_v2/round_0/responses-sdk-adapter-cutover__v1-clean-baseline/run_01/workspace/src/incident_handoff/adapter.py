def normalize_response_items(items):
    events = []
    for item in items:
        item_type = item["type"]
        if item_type == "message":
            content = item["content"]
            if isinstance(content, list):
                text_parts = [c["text"] for c in content if c.get("type") == "output_text"]
                text = "".join(text_parts)
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
        elif item_type == "function_call":
            events.append(
                {
                    "kind": "tool_call",
                    "call_id": item["id"],
                    "name": item["tool_name"],
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
        elif item_type == "function_call_output":
            events.append(
                {
                    "kind": "tool_result",
                    "call_id": item["tool_call_id"],
                    "output": item["output"],
                }
            )
    return events
