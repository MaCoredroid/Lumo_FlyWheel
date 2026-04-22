from services.api.src.review_state import normalize_review_state


def build_release_gate_update(payload: dict) -> dict:
    approval_state = normalize_review_state(payload.get("approval_state"))
    return {
        "release_id": payload["release_id"],
        "approval_state": approval_state,
    }


def serialize_release_gate(record: dict) -> dict:
    return {
        "release_id": record["release_id"],
        "approval_state": normalize_review_state(record.get("approval_state")),
        "operator_hint": "watch human_review_required in the admin echo",
    }
