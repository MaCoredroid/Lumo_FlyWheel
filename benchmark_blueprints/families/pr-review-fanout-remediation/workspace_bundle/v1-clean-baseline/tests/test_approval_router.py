import unittest

from src.policy.approval_router import build_policy_preview, normalize_approval_state


class ApprovalRouterTests(unittest.TestCase):
    def test_manual_review_normalizes(self) -> None:
        self.assertEqual(normalize_approval_state("manual_review"), "human_review_required")

    def test_live_preview_uses_normalized_state(self) -> None:
        payload = build_policy_preview(
            request_id="req-214",
            approval_state="manual_review",
            preview_enabled=True,
            actor="review-bot",
        )
        self.assertEqual(payload["approval_state"], "human_review_required")


if __name__ == "__main__":
    unittest.main()
