def _iter_response_items(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return payload.get("output", [])
    raise TypeError("Responses payload must be a list of events or a dict with an output list")


def normalize_response_items(items):
    events = []
    for item in _iter_response_items(items):
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
    return events
