def _response_items(payload):
    if isinstance(payload, dict):
        if "output" in payload:
            return payload["output"]
        if "items" in payload:
            return payload["items"]
    return payload


def _message_text_events(item):
    events = []
    for block in item.get("content", []):
        block_type = block.get("type")
        if block_type in {"output_text", "text"} and "text" in block:
            events.append({"kind": "assistant_text", "text": block["text"]})
    return events


def normalize_response_items(payload):
    events = []
    for item in _response_items(payload):
        item_type = item["type"]
        if item_type == "message":
            events.extend(_message_text_events(item))
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
