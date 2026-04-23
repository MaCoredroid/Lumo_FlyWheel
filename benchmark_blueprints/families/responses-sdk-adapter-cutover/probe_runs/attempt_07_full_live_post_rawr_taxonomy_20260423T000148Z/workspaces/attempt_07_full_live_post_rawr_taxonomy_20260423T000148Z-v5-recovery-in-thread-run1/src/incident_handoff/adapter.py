def normalize_response_items(items):
    events = []
    ordered_items = sorted(
        enumerate(items),
        key=lambda entry: (entry[1].get("sequence", entry[0]), entry[0]),
    )
    for _, item in ordered_items:
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
            events.append(
                {
                    "kind": "response_event",
                    "event_type": item_type,
                    "payload": item,
                }
            )
    return events
