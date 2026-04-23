WIRE_API = "responses"
LEGACY_WRAPPER = False


def request_wire_config():
    return {
        "wire_api": WIRE_API,
        "transcript_mode": "response_events",
    }


def extract_response_items(response):
    if "output" in response:
        return response["output"]
    return response["response"]["output"]
