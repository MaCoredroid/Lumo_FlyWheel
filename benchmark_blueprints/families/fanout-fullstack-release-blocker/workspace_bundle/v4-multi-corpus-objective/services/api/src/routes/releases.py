from services.api.src.review_state import normalize_review_state


def build_release_gate_update(payload: dict) -> dict:
    approval_state = normalize_review_state(payload.get("approval_state", "manual_review"))
    return {
        "release_id": payload["release_id"],
        "approval_state": approval_state,
    }


def serialize_release_gate(record: dict) -> dict:
    return {
        "release_id": record["release_id"],
        "approval_state": record.get("approval_state", "manual_review"),
        "operator_hint": "watch manual_review in the admin echo",
    }
