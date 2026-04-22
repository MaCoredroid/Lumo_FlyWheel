# Partial Progress Rubric

Award probe-only credit when a submission threads `approval_state` across multiple real surfaces but still fails a hidden contract check.

Suggested heuristic bands:
- `0`: single-surface or cosmetic-only edits
- `4`: backend plus one neighboring surface updated
- `6`: backend, CLI, and frontend all updated but docs/artifacts still stale
- `8`: code and docs mostly aligned, but preview or rollout note still invalid
- `10`: nearly complete submission missing only one hidden constraint

These points must stay in `P_benchmark` only and never enter `M_training`.
