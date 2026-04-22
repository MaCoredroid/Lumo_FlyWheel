WIRE_API = "responses"


def request_wire_config():
    return {
        "wire_api": WIRE_API,
        "transcript_mode": "responses_events",
    }


def extract_response_items(response):
    if isinstance(response, list):
        return response
    if isinstance(response, dict) and isinstance(response.get("output"), list):
        return response["output"]
    raise ValueError("expected Responses output items")
