DEFAULT_APPROVAL_STATE = "human_review_required"
ACTIVE_APPROVAL_STATES = {"human_review_required", "auto_approve", "blocked"}

_COMPAT_MAP = {
    "manual_review": "human_review_required",
}


def normalize_review_state(raw_state: str | None) -> str:
    state = (raw_state or DEFAULT_APPROVAL_STATE).strip()
    if state in ACTIVE_APPROVAL_STATES:
        return state
    if state in _COMPAT_MAP:
        return _COMPAT_MAP[state]
    raise ValueError(f"unsupported approval_state: {raw_state!r}")
