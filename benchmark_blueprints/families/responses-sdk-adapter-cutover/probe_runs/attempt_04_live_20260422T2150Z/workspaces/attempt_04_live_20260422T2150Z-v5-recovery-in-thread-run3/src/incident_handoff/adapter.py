def _ordered_items(items):
    indexed_items = list(enumerate(items))
    if any("sequence" in item for _, item in indexed_items):
        return [
            item
            for _, item in sorted(
                indexed_items,
                key=lambda indexed: (indexed[1].get("sequence", float("inf")), indexed[0]),
            )
        ]
    return [item for _, item in indexed_items]


def normalize_response_items(items):
    events = []
    for item in _ordered_items(items):
        item_type = item["type"]
        if item_type == "message":
            for block in item.get("content", []):
                if block["type"] == "output_text":
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
            events.append({"kind": item_type, **{k: v for k, v in item.items() if k != "type"}})
    return events
