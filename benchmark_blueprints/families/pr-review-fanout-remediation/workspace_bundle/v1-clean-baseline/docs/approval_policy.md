# Approval Policy

Preview routing keeps the live preview payload aligned with policy routing.

## Fallback Response

When preview rendering is unavailable, return:

```json
{
  "request_id": "req-214",
  "status": "preview_unavailable",
  "actor": "review-bot"
}
```

## Notes

- `manual_review` is retired in favor of `human_review_required`.
- The fallback example above still mirrors the old reviewer write-up and needs
  to be updated once the reopened thread lands.
