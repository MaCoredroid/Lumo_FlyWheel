def _sequence_key(index, item):
    return (item.get("sequence", index), index)


def _iter_message_blocks(item):
    for block in item.get("content", []):
        if block.get("type") == "output_text":
            yield {"kind": "assistant_text", "text": block["text"]}


def normalize_response_items(items):
    response_items = items.get("output", items) if isinstance(items, dict) else items
    events = []
    ordered_items = sorted(enumerate(response_items), key=lambda pair: _sequence_key(*pair))
    for _, item in ordered_items:
        item_type = item["type"]
        if item_type == "message":
            events.extend(_iter_message_blocks(item))
        elif item_type in {"tool_call", "function_call"}:
            events.append(
                {
                    "kind": "tool_call",
                    "call_id": item["call_id"],
                    "name": item.get("name", item.get("function", {}).get("name")),
                    "arguments": item.get("arguments", item.get("function", {}).get("arguments")),
                }
            )
        elif item_type in {"tool_result", "function_call_output"}:
            events.append(
                {
                    "kind": "tool_result",
                    "call_id": item["call_id"],
                    "output": item.get("output", item.get("content")),
                }
            )
    return events
