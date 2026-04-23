def _ordered_items(items):
    indexed_items = []
    for original_index, item in enumerate(items):
        indexed_items.append((item.get("sequence", original_index), original_index, item))
    indexed_items.sort(key=lambda row: (row[0], row[1]))
    return [item for _, _, item in indexed_items]


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
        else:
            events.append({"kind": "response_item", "item": item})
    return events
