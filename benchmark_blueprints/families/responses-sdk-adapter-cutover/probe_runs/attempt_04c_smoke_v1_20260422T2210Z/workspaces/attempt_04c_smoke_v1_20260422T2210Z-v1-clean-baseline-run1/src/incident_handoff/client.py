WIRE_API = "responses"
LEGACY_WRAPPER = False


def request_wire_config():
    return {
        "wire_api": WIRE_API,
        "transcript_mode": "responses_events",
    }


def extract_response_items(response):
    if isinstance(response, list):
        return response
    if "output" in response:
        return response["output"]
    if "items" in response:
        return response["items"]
    raise KeyError("Responses payload is missing an output item list")
