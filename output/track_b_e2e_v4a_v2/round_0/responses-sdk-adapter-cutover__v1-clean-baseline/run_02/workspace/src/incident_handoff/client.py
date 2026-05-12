WIRE_API = "responses"
LEGACY_WRAPPER = False


def request_wire_config():
    return {
        "wire_api": WIRE_API,
        "transcript_mode": "responses_events",
    }


def extract_response_items(response):
    """
    Extract items from a Responses API response.
    
    Returns the items array which contains messages, tool calls, and tool results
    in the order they were emitted by the API.
    """
    return response.get("output", [])
