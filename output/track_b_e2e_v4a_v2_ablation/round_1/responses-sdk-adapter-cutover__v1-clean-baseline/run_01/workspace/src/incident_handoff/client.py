WIRE_API = "responses"
LEGACY_WRAPPER = False


def request_wire_config():
    return {
        "wire_api": WIRE_API,
        "transcript_mode": "event",
    }


def extract_response_items(response):
    return response["events"]
