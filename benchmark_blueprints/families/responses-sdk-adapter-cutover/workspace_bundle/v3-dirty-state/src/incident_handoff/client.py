WIRE_API = "chat_completions"
LEGACY_WRAPPER = True


def request_wire_config():
    return {
        "wire_api": WIRE_API,
        "transcript_mode": "legacy_messages",
    }


def extract_response_items(response):
    return response["choices"][0]["message"]["content"]
