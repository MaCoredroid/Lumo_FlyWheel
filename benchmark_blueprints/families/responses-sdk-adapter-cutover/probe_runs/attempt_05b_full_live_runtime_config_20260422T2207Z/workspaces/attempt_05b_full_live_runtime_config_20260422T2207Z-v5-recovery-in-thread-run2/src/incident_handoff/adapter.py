def _event_sort_key(index, item):
    return (item.get("sequence", index), index)


def _message_text(content):
    if isinstance(content, str):
        return content

    parts = []
    for block in content or []:
        if block.get("type") == "output_text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def normalize_response_items(items):
    events = []
    ordered_items = sorted(enumerate(items), key=lambda pair: _event_sort_key(*pair))
    for _, item in ordered_items:
        item_type = item["type"]
        if item_type == "message":
            events.append({"kind": "assistant_text", "text": _message_text(item.get("content"))})
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
            passthrough_event = {"kind": item_type, "raw": item}
            if "call_id" in item:
                passthrough_event["call_id"] = item["call_id"]
            events.append(passthrough_event)
    return events
