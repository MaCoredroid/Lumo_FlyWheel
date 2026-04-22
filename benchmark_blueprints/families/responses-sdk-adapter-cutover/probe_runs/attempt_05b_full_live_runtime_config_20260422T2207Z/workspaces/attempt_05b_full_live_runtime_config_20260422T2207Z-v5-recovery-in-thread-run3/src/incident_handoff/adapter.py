def _ordered_response_items(items):
    indexed_items = list(enumerate(items))
    return [
        item
        for _, item in sorted(
            indexed_items,
            key=lambda entry: (
                entry[1].get("sequence", float("inf")),
                entry[0],
            ),
        )
    ]


def _copy_sequence(item, event):
    if "sequence" in item:
        event["sequence"] = item["sequence"]
    return event


def normalize_response_items(items):
    events = []
    for item in _ordered_response_items(items):
        item_type = item["type"]
        if item_type == "message":
            content = item.get("content", [])
            if isinstance(content, str):
                events.append(_copy_sequence(item, {"kind": "assistant_text", "text": content}))
                continue

            for block in content:
                if block.get("type") == "output_text":
                    events.append(
                        _copy_sequence(item, {"kind": "assistant_text", "text": block["text"]})
                    )
        elif item_type == "legacy_message":
            if item.get("role") == "assistant":
                events.append(
                    _copy_sequence(item, {"kind": "assistant_text", "text": item["content"]})
                )
            else:
                events.append({"kind": "response_item", "item": dict(item)})
        elif item_type == "tool_call":
            events.append(
                _copy_sequence(
                    item,
                    {
                        "kind": "tool_call",
                        "call_id": item["call_id"],
                        "name": item["name"],
                        "arguments": item["arguments"],
                    },
                )
            )
        elif item_type == "tool_result":
            events.append(
                _copy_sequence(
                    item,
                    {
                        "kind": "tool_result",
                        "call_id": item["call_id"],
                        "output": item["output"],
                    },
                )
            )
        else:
            events.append({"kind": "response_item", "item": dict(item)})
    return events
