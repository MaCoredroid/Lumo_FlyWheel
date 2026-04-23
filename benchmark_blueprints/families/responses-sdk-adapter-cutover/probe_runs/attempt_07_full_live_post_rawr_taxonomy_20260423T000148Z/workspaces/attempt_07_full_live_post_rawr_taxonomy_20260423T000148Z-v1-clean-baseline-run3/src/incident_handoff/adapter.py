def _iter_message_text_events(item):
    content = item.get("content", [])

    if isinstance(content, str):
        yield {"kind": "assistant_text", "text": content}
        return

    for block in content:
        block_type = block.get("type")
        if block_type in {"output_text", "text"}:
            yield {"kind": "assistant_text", "text": block["text"]}


def normalize_response_items(items):
    events = []
    for item in items:
        item_type = item["type"]
        if item_type == "message":
            events.extend(_iter_message_text_events(item))
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
    return events
