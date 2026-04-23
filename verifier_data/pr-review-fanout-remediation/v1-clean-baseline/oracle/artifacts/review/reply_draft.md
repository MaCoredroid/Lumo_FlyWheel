# Review Replies

## T-214-01

Fixed the reopened fallback path in `src/policy/approval_router.py` and
`src/policy/preview.py` so the `preview_unavailable` response now keeps
`approval_state="human_review_required"` and `requires_human_review=true`.

## T-214-02

Closing as duplicate of `T-214-01`. The same fallback payload fix covers this
thread, so I did not apply a second code path.

## T-214-03

Did not apply this suggestion. The comment points at an outdated diff hunk and
reviving `legacy_preview_branch` would move the branch away from the current
fallback contract.

## T-214-04

Added a fallback regression in `tests/test_preview.py` for
`preview_enabled=False` with normalized `manual_review`.

## T-214-05

Updated `docs/approval_policy.md` so the fallback response example now includes
`approval_state` and `requires_human_review`. 