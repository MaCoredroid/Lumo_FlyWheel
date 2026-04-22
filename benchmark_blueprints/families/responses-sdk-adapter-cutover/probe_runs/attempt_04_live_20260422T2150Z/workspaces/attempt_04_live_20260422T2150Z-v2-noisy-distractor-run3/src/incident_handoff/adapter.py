def _message_text_events(content):
    if isinstance(content, str):
        return [{"kind": "assistant_text", "text": content}]

    events = []
    for block in content:
        if block.get("type") == "output_text":
            events.append({"kind": "assistant_text", "text": block["text"]})
    return events


def normalize_response_items(items):
    events = []
    for item in items:
        item_type = item["type"]
        if item_type == "message":
            events.extend(_message_text_events(item.get("content", [])))
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
