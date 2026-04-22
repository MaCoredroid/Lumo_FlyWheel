def _item_sequence(item, original_index):
    sequence = item.get("sequence")
    if sequence is None:
        return (1, original_index)
    return (0, sequence, original_index)


def normalize_response_items(items):
    events = []
    ordered_items = sorted(
        enumerate(items),
        key=lambda indexed_item: _item_sequence(indexed_item[1], indexed_item[0]),
    )
    for _, item in ordered_items:
        item_type = item["type"]
        if item_type == "message":
            for block in item.get("content", []):
                block_type = block.get("type")
                if block_type in {"output_text", "text"}:
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
            events.append(
                {
                    "kind": "response_item",
                    "type": item_type,
                    "item": item,
                }
            )
    return events
