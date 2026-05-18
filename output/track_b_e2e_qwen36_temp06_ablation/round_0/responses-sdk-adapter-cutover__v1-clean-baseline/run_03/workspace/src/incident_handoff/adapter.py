def _extract_text(item):
    """Extract text from a Responses message item whose content is an array."""
    content = item.get("content", [])
    if isinstance(content, list):
        texts = [c["text"] for c in content if c.get("type") == "output_text"]
        return " ".join(texts) if texts else ""
    return content if isinstance(content, str) else ""


def normalize_response_items(items):
    events = []
    for item in items:
        item_type = item["type"]
        if item_type == "message":
            events.append({"kind": "assistant_text", "text": _extract_text(item)})
        elif item_type in ("tool_call", "function_call"):
            call_id = item.get("call_id", item.get("id", ""))
            name = item.get("name", item.get("tool_name", ""))
            events.append(
                {
                    "kind": "tool_call",
                    "call_id": call_id,
                    "name": name,
                    "arguments": item.get("arguments", ""),
                }
            )
        elif item_type in ("tool_result", "function_call_output"):
            call_id = item.get("call_id", item.get("tool_call_id", ""))
            events.append(
                {
                    "kind": "tool_result",
                    "call_id": call_id,
                    "output": item.get("output", ""),
                }
            )
    return events
