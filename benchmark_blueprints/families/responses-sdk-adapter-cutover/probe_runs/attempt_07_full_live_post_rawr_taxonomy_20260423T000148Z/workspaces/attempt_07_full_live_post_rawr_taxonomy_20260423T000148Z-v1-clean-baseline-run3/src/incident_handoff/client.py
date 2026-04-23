WIRE_API = "responses"
LEGACY_WRAPPER = False


def request_wire_config():
    return {
        "wire_api": WIRE_API,
        "transcript_mode": "responses_events",
    }


def extract_response_items(response):
    if "output" in response:
        return response["output"]
    if "response" in response and "output" in response["response"]:
        return response["response"]["output"]
    return response["choices"][0]["message"]["content"]
