def _message_blocks(item):
    content = item.get("content", [])
    if isinstance(content, str):
        return [content]

    blocks = []
    for block in content:
        if block.get("type") in {"output_text", "text"}:
            text = block.get("text")
            if text:
                blocks.append(text)
    return blocks


def normalize_response_items(items):
    events = []
    for item in items:
        item_type = item["type"]
        if item_type == "message":
            for text in _message_blocks(item):
                events.append({"kind": "assistant_text", "text": text})
        elif item_type in {"tool_call", "function_call"}:
            events.append(
                {
                    "kind": "tool_call",
                    "call_id": item["call_id"],
                    "name": item["name"],
                    "arguments": item["arguments"],
                }
            )
        elif item_type in {"tool_result", "function_call_output"}:
            events.append(
                {
                    "kind": "tool_result",
                    "call_id": item["call_id"],
                    "output": item["output"],
                }
            )
    return events
