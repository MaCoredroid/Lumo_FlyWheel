import json


def _item_sequence(item, fallback_index):
    return item.get("sequence", fallback_index)


def _normalize_message_content(content):
    if isinstance(content, str):
        return [{"kind": "assistant_text", "text": content}]

    events = []
    for block in content or []:
        if isinstance(block, str):
            events.append({"kind": "assistant_text", "text": block})
            continue

        block_type = block.get("type")
        if block_type in {"output_text", "text"}:
            events.append({"kind": "assistant_text", "text": block["text"]})
    return events


def _normalize_tool_call(item):
    arguments = item.get("arguments", "")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, sort_keys=True)

    return {
        "kind": "tool_call",
        "call_id": item.get("call_id", item.get("id")),
        "name": item["name"],
        "arguments": arguments,
    }


def _normalize_tool_result(item):
    output = item.get("output", item.get("result"))
    if not isinstance(output, str):
        output = json.dumps(output, sort_keys=True)

    return {
        "kind": "tool_result",
        "call_id": item.get("call_id", item.get("id")),
        "output": output,
    }


def normalize_response_items(items):
    events = []
    ordered_items = sorted(
        enumerate(items),
        key=lambda pair: (_item_sequence(pair[1], pair[0]), pair[0]),
    )
    for _, item in ordered_items:
        item_type = item["type"]
        if item_type == "message":
            events.extend(_normalize_message_content(item.get("content")))
        elif item_type in {"tool_call", "function_call"}:
            events.append(_normalize_tool_call(item))
        elif item_type in {"tool_result", "function_call_output"}:
            events.append(_normalize_tool_result(item))
        elif item_type == "legacy_message" and item.get("role") == "assistant":
            events.extend(_normalize_message_content(item.get("content")))
    return events
