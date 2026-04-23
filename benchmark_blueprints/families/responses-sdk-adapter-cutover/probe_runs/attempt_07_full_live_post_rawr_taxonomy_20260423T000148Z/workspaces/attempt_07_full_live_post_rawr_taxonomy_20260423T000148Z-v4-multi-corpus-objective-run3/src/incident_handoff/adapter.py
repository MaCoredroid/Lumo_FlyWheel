def _ordered_items(items):
    indexed_items = list(enumerate(items))
    return [
        item
        for _, item in sorted(
            indexed_items,
            key=lambda pair: (pair[1].get("sequence", pair[0]), pair[0]),
        )
    ]


def _message_events(item):
    content = item.get("content", [])
    if isinstance(content, str):
        return [{"kind": "assistant_text", "text": content}]

    events = []
    for block in content:
        if block.get("type") == "output_text":
            events.append({"kind": "assistant_text", "text": block["text"]})
    return events


def normalize_response_items(items):
    events = []
    for item in _ordered_items(items):
        item_type = item["type"]
        if item_type == "message":
            events.extend(_message_events(item))
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
