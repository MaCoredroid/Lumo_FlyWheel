from src.policy.preview import build_preview_payload

LEGACY_APPROVAL_STATES = {"manual_review": "human_review_required"}


def normalize_approval_state(approval_state) -> str:
    state = (approval_state or "manual_review").strip()
    return LEGACY_APPROVAL_STATES.get(state, state)


def build_policy_preview(
    request_id: str,
    approval_state,
    preview_enabled: bool,
    actor: str,
) -> dict:
    normalized_state = normalize_approval_state(approval_state)
    return build_preview_payload(
        request_id=request_id,
        approval_state=normalized_state,
        preview_enabled=preview_enabled,
        actor=actor,
    )
