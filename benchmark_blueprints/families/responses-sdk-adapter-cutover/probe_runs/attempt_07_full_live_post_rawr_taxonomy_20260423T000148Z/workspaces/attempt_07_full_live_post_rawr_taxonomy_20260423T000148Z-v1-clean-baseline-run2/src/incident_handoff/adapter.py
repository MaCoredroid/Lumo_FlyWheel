import json


def _stringify(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"))


def _extract_message_text(content):
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            block_type = block.get("type")
            if block_type in {"output_text", "text", "input_text"} and "text" in block:
                parts.append(block["text"])
        return "".join(parts)

    return ""


def normalize_response_items(items):
    events = []
    for item in items:
        item_type = item["type"]
        if item_type == "message":
            text = _extract_message_text(item.get("content", ""))
            if text:
                events.append({"kind": "assistant_text", "text": text})
        elif item_type in {"tool_call", "function_call"}:
            events.append(
                {
                    "kind": "tool_call",
                    "call_id": item.get("call_id", item.get("id")),
                    "name": item.get("name", item.get("tool_name")),
                    "arguments": _stringify(item.get("arguments", "")),
                }
            )
        elif item_type in {"tool_result", "function_call_output"}:
            events.append(
                {
                    "kind": "tool_result",
                    "call_id": item.get("call_id", item.get("tool_call_id")),
                    "output": _stringify(item.get("output", "")),
                }
            )
    return events
