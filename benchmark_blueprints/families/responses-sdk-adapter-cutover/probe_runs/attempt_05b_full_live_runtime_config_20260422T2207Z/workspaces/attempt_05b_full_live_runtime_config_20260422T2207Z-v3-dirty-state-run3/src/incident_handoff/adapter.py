def _item_sequence(item, fallback_index):
    return item.get("sequence", fallback_index)


def normalize_response_items(items):
    events = []
    ordered_items = sorted(
        enumerate(items),
        key=lambda pair: (_item_sequence(pair[1], pair[0]), pair[0]),
    )
    for _, item in ordered_items:
        item_type = item["type"]
        if item_type == "message":
            content = item.get("content", [])
            if isinstance(content, str):
                events.append({"kind": "assistant_text", "text": content})
                continue
            for block in content:
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
        elif item_type == "legacy_message" and item.get("role") == "assistant":
            events.append({"kind": "assistant_text", "text": item["content"]})
    return events
