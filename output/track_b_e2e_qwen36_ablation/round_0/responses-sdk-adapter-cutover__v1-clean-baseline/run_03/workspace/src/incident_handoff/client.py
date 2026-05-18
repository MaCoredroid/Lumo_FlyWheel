WIRE_API = "responses"


def request_wire_config():
    return {
        "wire_api": WIRE_API,
        "transcript_mode": "event_stream",
    }


def extract_response_items(response):
    return response["output"]
