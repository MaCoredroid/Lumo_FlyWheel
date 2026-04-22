WIRE_API = "responses"
LEGACY_WRAPPER = False
TRANSCRIPT_MODE = "responses_events"


def request_wire_config():
    return {
        "wire_api": WIRE_API,
        "transcript_mode": TRANSCRIPT_MODE,
    }


def extract_response_items(response):
    if "output" in response:
        return response["output"]
    if "items" in response:
        return response["items"]
    return response
