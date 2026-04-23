def build_preview_payload(
    request_id: str,
    approval_state: str,
    preview_enabled: bool,
    actor: str,
) -> dict:
    payload = {
        "request_id": request_id,
        "actor": actor,
        "approval_state": approval_state,
        "requires_human_review": approval_state == "human_review_required",
    }
    if preview_enabled:
        payload["preview_path"] = f"/preview/{request_id}"
        return payload
    payload["status"] = "preview_unavailable"
    return payload
