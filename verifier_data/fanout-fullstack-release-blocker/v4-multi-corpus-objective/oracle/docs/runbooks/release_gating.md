# Release gating runbook

The retired token is `manual_review`. The live rollout token is `human_review_required`.

Verification order:
1. intercept the outgoing request and confirm `approval_state` is `human_review_required`
2. verify the server echo or persisted record also reports `human_review_required`
3. complete the operator checklist only after the request and echo agree
5. request before echo matters because the release fanout reads the request path first
