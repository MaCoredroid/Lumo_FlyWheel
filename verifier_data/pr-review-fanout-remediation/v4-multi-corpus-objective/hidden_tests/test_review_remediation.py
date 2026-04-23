import unittest

from src.policy.approval_router import build_policy_preview


class HiddenPreviewContractTests(unittest.TestCase):
    def test_fallback_retains_human_review_fields(self) -> None:
        payload = build_policy_preview(
            request_id="req-214",
            approval_state="manual_review",
            preview_enabled=False,
            actor="review-bot",
        )
        self.assertEqual(payload["status"], "preview_unavailable")
        self.assertEqual(payload["approval_state"], "human_review_required")
        self.assertTrue(payload["requires_human_review"])

    def test_live_preview_still_uses_normalized_state(self) -> None:
        payload = build_policy_preview(
            request_id="req-214",
            approval_state="manual_review",
            preview_enabled=True,
            actor="review-bot",
        )
        self.assertEqual(payload["approval_state"], "human_review_required")
        self.assertEqual(payload["preview_path"], "/preview/req-214")

    def test_no_legacy_preview_hint_alias(self) -> None:
        payload = build_policy_preview(
            request_id="req-214",
            approval_state="manual_review",
            preview_enabled=False,
            actor="review-bot",
        )
        self.assertNotIn("legacy_preview_hint", payload)


if __name__ == "__main__":
    unittest.main()
