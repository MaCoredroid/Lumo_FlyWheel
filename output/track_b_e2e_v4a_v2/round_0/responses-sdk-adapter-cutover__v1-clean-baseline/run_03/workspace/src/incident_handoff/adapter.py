def normalize_response_items(items):
    """
    Normalize chat-completion items to Responses event semantics.
    Preserves event ordering and tool-result correlation via call_id.
    """
    def extract_text(content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "output_text":
                    texts.append(item.get("text", ""))
            return "".join(texts)
        return str(content)

    events = []
    for item in items:
        item_type = item["type"]
        if item_type == "message":
            events.append({"kind": "assistant_text", "text": extract_text(item["content"])})
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
