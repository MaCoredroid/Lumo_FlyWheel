def _append_message_events(events, content):
    if isinstance(content, str):
        events.append({"kind": "assistant_text", "text": content})
        return

    for block in content or []:
        if block.get("type") == "output_text":
            events.append({"kind": "assistant_text", "text": block["text"]})


def normalize_response_items(items):
    events = []
    for item in items:
        item_type = item["type"]
        if item_type == "message":
            _append_message_events(events, item.get("content"))
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
