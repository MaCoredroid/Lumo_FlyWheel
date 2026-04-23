def build_preview_payload(
    request_id: str,
    approval_state: str,
    preview_enabled: bool,
    actor: str,
) -> dict:
    if preview_enabled:
        return {
            "request_id": request_id,
            "actor": actor,
            "preview_path": f"/preview/{request_id}",
            "approval_state": approval_state,
            "requires_human_review": approval_state == "human_review_required",
        }
    return {
        "request_id": request_id,
        "status": "preview_unavailable",
        "actor": actor,
    }
