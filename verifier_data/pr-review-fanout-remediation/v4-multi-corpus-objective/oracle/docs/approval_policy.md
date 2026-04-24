# Approval Policy

Preview routing keeps the live preview payload aligned with policy routing.

## Fallback Response

When preview rendering is unavailable, return:

```json
{
  "request_id": "req-214",
  "status": "preview_unavailable",
  "actor": "review-bot",
  "approval_state": "human_review_required",
  "requires_human_review": true
}
```

## Notes

- `manual_review` is retired in favor of `human_review_required`.
- The fallback response keeps both `approval_state` and
  `requires_human_review` aligned with the live preview path.
- Downstream consumers read `approval_state` on fallback payloads first, so the fallback contract must keep that field populated.
