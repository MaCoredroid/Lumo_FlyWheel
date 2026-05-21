DEFAULT_APPROVAL_STATE = "human_review_required"
ACTIVE_APPROVAL_STATES = {"human_review_required", "auto_approve", "blocked"}


LEGACY_STATE_MAP: dict[str, str] = {
    "manual_review": "human_review_required",
}


def normalize_review_state(raw_state: str | None) -> str:
    state = (raw_state or DEFAULT_APPROVAL_STATE).strip()
    mapped = LEGACY_STATE_MAP.get(state, state)
    if mapped in ACTIVE_APPROVAL_STATES:
        return mapped
    raise ValueError(f"unsupported approval_state: {raw_state!r}")
