def _sequence_for(item, default_index):
    return int(item.get("sequence", item.get("index", default_index)))


def _message_text(blocks):
    if isinstance(blocks, str):
        return blocks
    parts = []
    for block in blocks or []:
        if block.get("type") in {"output_text", "text"} and "text" in block:
            parts.append(block["text"])
    return "\n".join(part for part in parts if part)


def normalize_response_items(items):
    events = []
    ordered = sorted(
        enumerate(items),
        key=lambda pair: _sequence_for(pair[1], pair[0]),
    )
    for index, item in ordered:
        item_type = item["type"]
        sequence = _sequence_for(item, index)
        if item_type == "message":
            text = _message_text(item.get("content", []))
            if text:
                events.append({"kind": "assistant_text", "text": text, "sequence": sequence})
        elif item_type == "tool_call":
            events.append(
                {
                    "kind": "tool_call",
                    "call_id": item["call_id"],
                    "name": item["name"],
                    "arguments": item["arguments"],
                    "sequence": sequence,
                }
            )
        elif item_type == "tool_result":
            events.append(
                {
                    "kind": "tool_result",
                    "call_id": item["call_id"],
                    "output": item["output"],
                    "sequence": sequence,
                }
            )
        else:
            events.append(
                {
                    "kind": "unknown_event",
                    "event_type": item_type,
                    "payload": item,
                    "sequence": sequence,
                }
            )
    return events
