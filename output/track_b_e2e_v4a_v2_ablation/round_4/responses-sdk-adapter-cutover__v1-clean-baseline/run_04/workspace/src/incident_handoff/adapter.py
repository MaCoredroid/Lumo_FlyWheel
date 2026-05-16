def normalize_response_items(items):
    events = []
    for item in items:
        item_type = item["type"]
        if item_type == "message":
            content = item["content"]
            if isinstance(content, list) and len(content) > 0:
                text = content[0].get("text", "")
            else:
                text = content
            events.append({"kind": "assistant_text", "text": text})
        elif item_type in ("tool_call", "function_call"):
            call_id = item.get("call_id") or item.get("id")
            name = item.get("name") or item.get("tool_name")
            arguments = item.get("arguments", "")
            events.append(
                {
                    "kind": "tool_call",
                    "call_id": call_id,
                    "name": name,
                    "arguments": arguments,
                }
            )
        elif item_type in ("tool_result", "function_call_output"):
            call_id = item.get("call_id") or item.get("tool_call_id")
            output = item.get("output", "")
            events.append(
                {
                    "kind": "tool_result",
                    "call_id": call_id,
                    "output": output,
                }
            )
    return events
