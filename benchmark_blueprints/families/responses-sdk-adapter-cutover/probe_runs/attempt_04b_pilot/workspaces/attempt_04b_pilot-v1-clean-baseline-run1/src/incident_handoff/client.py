WIRE_API = "responses"
TRANSCRIPT_MODE = "responses_events"
LEGACY_WRAPPER = False


def request_wire_config():
    return {
        "wire_api": WIRE_API,
        "transcript_mode": TRANSCRIPT_MODE,
    }


def extract_response_items(response):
    if "output" in response:
        return response["output"]
    if "response" in response and isinstance(response["response"], dict):
        nested_response = response["response"]
        if "output" in nested_response:
            return nested_response["output"]
    raise KeyError("Responses payload missing output items")
