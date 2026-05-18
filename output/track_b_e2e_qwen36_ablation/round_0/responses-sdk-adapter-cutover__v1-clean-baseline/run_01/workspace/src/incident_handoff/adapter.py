def normalize_response_items(items):
    events = []
    for item in items:
        item_type = item["type"]
        if item_type == "message":
            texts = [
                block["text"]
                for block in item["content"]
                if block.get("type") == "output_text"
            ]
            events.append({"kind": "assistant_text", "text": "".join(texts)})
        elif item_type in ("tool_call", "function_call"):
            call_id_key = "call_id" if item_type == "tool_call" else "id"
            name_key = "name" if item_type == "tool_call" else "tool_name"
            events.append(
                {
                    "kind": "tool_call",
                    "call_id": item[call_id_key],
                    "name": item[name_key],
                    "arguments": item["arguments"],
                }
            )
        elif item_type in ("tool_result", "function_call_output"):
            call_id_key = "call_id" if item_type == "tool_result" else "tool_call_id"
            events.append(
                {
                    "kind": "tool_result",
                    "call_id": item[call_id_key],
                    "output": item["output"],
                }
            )
    return events
