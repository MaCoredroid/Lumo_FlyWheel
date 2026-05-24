DEFAULT_APPROVAL_STATE = "human_review_required"
LEGACY_STATE_MAP = {"manual_review": "human_review_required"}

    state = (raw_state or DEFAULT_APPROVAL_STATE).strip()
    if mapped in ACTIVE_APPROVAL_STATES:


