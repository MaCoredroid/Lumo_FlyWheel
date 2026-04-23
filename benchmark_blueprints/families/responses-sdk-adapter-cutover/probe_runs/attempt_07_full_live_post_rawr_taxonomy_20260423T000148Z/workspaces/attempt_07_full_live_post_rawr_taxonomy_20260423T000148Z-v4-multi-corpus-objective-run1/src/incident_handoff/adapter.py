def _message_events(item):
    events = []
    for block in item.get("content", []):
        if block.get("type") == "output_text":
            events.append({"kind": "assistant_text", "text": block["text"]})
    return events


def normalize_response_items(items):
    events = []
    indexed_items = list(enumerate(items))
    indexed_items.sort(key=lambda pair: (pair[1].get("sequence", pair[0]), pair[0]))

    for _, item in indexed_items:
        item_type = item["type"]
        if item_type == "message":
            events.extend(_message_events(item))
        elif item_type in {"tool_call", "function_call"}:
            events.append(
                {
                    "kind": "tool_call",
                    "call_id": item["call_id"],
                    "name": item["name"],
                    "arguments": item["arguments"],
                }
            )
        elif item_type in {"tool_result", "function_call_output"}:
            events.append(
                {
                    "kind": "tool_result",
                    "call_id": item["call_id"],
                    "output": item["output"],
                }
            )
    return events
