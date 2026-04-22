def _ordered_items(items):
    indexed = enumerate(items)
    return [
        item
        for _, item in sorted(
            indexed,
            key=lambda pair: (pair[1].get("sequence", pair[0]), pair[0]),
        )
    ]


def normalize_response_items(items):
    events = []
    for item in _ordered_items(items):
        item_type = item["type"]
        if item_type == "message":
            for block in item.get("content", []):
                if block.get("type") == "output_text":
                    events.append({"kind": "assistant_text", "text": block["text"]})
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
