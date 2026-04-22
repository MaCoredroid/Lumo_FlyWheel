WIRE_API = "responses"
LEGACY_WRAPPER = "chat_completions"


def request_wire_config():
    return {
        "wire_api": WIRE_API,
        "transcript_mode": "responses_events",
    }


def extract_response_items(response):
    if isinstance(response, dict) and "output" in response:
        return response["output"]
    return response.get("choices", [{}])[0].get("message", {}).get("content", [])
