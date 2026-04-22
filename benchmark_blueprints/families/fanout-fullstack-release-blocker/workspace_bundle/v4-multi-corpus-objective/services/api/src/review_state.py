DEFAULT_APPROVAL_STATE = "manual_review"
ACTIVE_APPROVAL_STATES = {"manual_review", "auto_approve", "blocked"}


def normalize_review_state(raw_state: str | None) -> str:
    state = (raw_state or DEFAULT_APPROVAL_STATE).strip()
    if state in ACTIVE_APPROVAL_STATES:
        return state
    raise ValueError(f"unsupported approval_state: {raw_state!r}")
