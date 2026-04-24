# Remediation Report

## Scope

- Acted on `T-214-01`, `T-214-04`, and `T-214-05`.
- Closed `T-214-02` as duplicate of `T-214-01`.
- Did not apply `T-214-03` because the stale digest and outdated hunk were not authoritative on the current branch.

## What Changed

- `src/policy/approval_router.py`: normalize `manual_review` before building the preview payload.
- `src/policy/preview.py`: keep `approval_state` and `requires_human_review` on the fallback payload.
- `tests/test_preview.py`: add the fallback regression case.
- `docs/approval_policy.md`: update the fallback example to the final contract.

## Notes

- The stale digest was used only as a hint; the review export remained the source of truth.
- `T-214-03` was intentionally not applied.
- The previous attempt patch added `legacy_preview_hint`; I did not revive that alias.
