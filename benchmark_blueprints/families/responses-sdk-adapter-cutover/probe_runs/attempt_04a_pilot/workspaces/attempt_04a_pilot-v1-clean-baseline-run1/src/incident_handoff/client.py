WIRE_API = "responses"
LEGACY_WRAPPER = False


def request_wire_config():
    return {
        "wire_api": WIRE_API,
        "transcript_mode": "response_output_items",
        "preserve_event_order": True,
    }


def extract_response_items(response):
    if "output" in response:
        return response["output"]
    if "items" in response:
        return response["items"]
    return []
