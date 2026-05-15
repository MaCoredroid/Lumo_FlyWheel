DEFAULT_APPROVAL_STATE = "human_review_required"
ACTIVE_APPROVAL_STATES = {"human_review_required", "auto_approve", "blocked"}


def normalize_review_state(raw_state: str | None) -> str:
    state = (raw_state or DEFAULT_APPROVAL_STATE).strip()
    # Legacy compatibility: map old token to new
    if state == "manual_review":
        return "human_review_required"
    if state in ACTIVE_APPROVAL_STATES:
        return state
    raise ValueError(f"unsupported approval_state: {raw_state!r}")
