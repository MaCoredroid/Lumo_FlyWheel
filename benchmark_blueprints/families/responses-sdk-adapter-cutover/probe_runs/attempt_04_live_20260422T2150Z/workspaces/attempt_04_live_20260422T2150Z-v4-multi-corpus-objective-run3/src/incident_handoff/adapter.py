def _iter_content_blocks(item):
    content = item.get("content", [])
    if isinstance(content, str):
        return [{"type": "output_text", "text": content}]
    return content or []


def _sequence_key(index, item):
    sequence = item.get("sequence")
    if not isinstance(sequence, int):
        sequence = index
    return (sequence, index)


def normalize_response_items(items):
    events = []
    ordered_items = sorted(enumerate(items), key=lambda pair: _sequence_key(*pair))
    for _, item in ordered_items:
        item_type = item["type"]
        if item_type == "message":
            for block in _iter_content_blocks(item):
                if block.get("type") in {"output_text", "text"} and block.get("text"):
                    events.append({"kind": "assistant_text", "text": block["text"]})
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
