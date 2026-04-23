def _item_sequence(item, fallback_index):
    sequence = item.get("sequence")
    if sequence is None:
        return fallback_index
    return sequence


def _message_events(item):
    if isinstance(item.get("content"), str):
        return [{"kind": "assistant_text", "text": item["content"]}]
    events = []
    for block in item.get("content", []):
        if block.get("type") == "output_text":
            events.append({"kind": "assistant_text", "text": block["text"]})
    return events


def normalize_response_items(items):
    events = []
    ordered_items = sorted(
        enumerate(items),
        key=lambda indexed_item: _item_sequence(indexed_item[1], indexed_item[0]),
    )
    for _, item in ordered_items:
        item_type = item["type"]
        if item_type == "message":
            events.extend(_message_events(item))
        elif item_type == "legacy_message" and item.get("role") == "assistant":
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
        else:
            passthrough_event = {"kind": item_type}
            for key, value in item.items():
                if key != "type":
                    passthrough_event[key] = value
            events.append(passthrough_event)
    return events
