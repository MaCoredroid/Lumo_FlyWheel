def _extract_text_from_content(content):
    """
    Extract text from Responses API content structure.
    
    Handles both plain string content and structured content arrays.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "output_text":
                texts.append(item.get("text", ""))
        return " ".join(texts)
    return str(content)


def normalize_response_items(items):
    """
    Normalize Responses API items into event stream format.
    
    Preserves event ordering as emitted by the Responses API.
    Tool calls and results are correlated via call_id.
    """
    events = []
    for item in items:
        item_type = item["type"]
        if item_type == "message":
            text = _extract_text_from_content(item["content"])
            events.append({"kind": "assistant_text", "text": text})
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


def correlate_tool_calls_with_results(events):
    """
    Build a mapping of tool calls to their results using call_id correlation.
    
    This preserves the Responses event semantics where tool calls and results
    are matched by explicit identifier, not by position.
    """
    call_map = {}
    for event in events:
        if event["kind"] == "tool_call":
            call_id = event["call_id"]
            call_map[call_id] = {"call": event, "result": None}
        elif event["kind"] == "tool_result":
            call_id = event["call_id"]
            if call_id in call_map:
                call_map[call_id]["result"] = event
    return call_map
