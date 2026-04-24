import unittest

from src.policy.preview import build_preview_payload


class PreviewPayloadTests(unittest.TestCase):
    def test_live_preview_includes_human_review_fields(self) -> None:
        payload = build_preview_payload(
            request_id="req-214",
            approval_state="human_review_required",
            preview_enabled=True,
            actor="review-bot",
        )
        self.assertEqual(payload["approval_state"], "human_review_required")
        self.assertTrue(payload["requires_human_review"])
        self.assertEqual(payload["preview_path"], "/preview/req-214")

    def test_preview_unavailable_keeps_status(self) -> None:
        payload = build_preview_payload(
            request_id="req-214",
            approval_state="human_review_required",
            preview_enabled=False,
            actor="review-bot",
        )
        self.assertEqual(payload["status"], "preview_unavailable")
        self.assertEqual(payload["actor"], "review-bot")


if __name__ == "__main__":
    unittest.main()
