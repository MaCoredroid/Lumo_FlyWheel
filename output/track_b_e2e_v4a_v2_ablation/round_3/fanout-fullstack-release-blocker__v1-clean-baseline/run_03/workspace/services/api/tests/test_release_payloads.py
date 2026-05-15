from services.api.src.review_state import normalize_review_state
from services.api.src.routes.releases import build_release_gate_update, serialize_release_gate


def test_legacy_manual_review_normalizes() -> None:
    assert normalize_review_state("manual_review") == "human_review_required"


def test_release_gate_update_emits_new_token() -> None:
    update = build_release_gate_update({"release_id": "rel-ship-0422", "approval_state": "human_review_required"})
    assert update["approval_state"] == "human_review_required"


def test_release_gate_echo_uses_new_token() -> None:
    emitted = serialize_release_gate({"release_id": "rel-ship-0422", "approval_state": "human_review_required"})
    assert emitted["approval_state"] == "human_review_required"
