# Evaluator Contract

## Scope
This evaluator scores one solver attempt against `task_spec.md` for the `ui-review-screenshot-remediation` family.

## Required Solver Inputs
- `task_spec.md`
- `codex/config.toml`
- `skills/screenshot-review-fix/SKILL.md`

## 100-Point Breakdown
- `20` Review-thread grounding
  - distinguishes actionable vs stale screenshot feedback
  - uses the current screenshot artifact as the active evidence surface
- `25` Responsive layout remediation quality
  - addresses overlap without changing reviewer-visible copy
  - proposes a product-code fix, not a screenshot or golden hack
- `15` Accessibility repair quality
  - preserves visible keyboard focus on the overflow action menu
- `15` Multi-viewport coverage
  - covers `360px`, `390px`, and `430px`
  - anticipates narrower, wider, or zoomed mobile cases
- `15` Reviewer-reply and note quality
  - concrete reply artifact path
  - correct treatment of stale thread
- `10` Constraint discipline
  - no text-shortening, label renaming, or screenshot-only changes

## Caps
- Cap at `20/100` if the attempt does not ground review triage in concrete thread ids and the current screenshot evidence surface.
- Cap at `20/100` if the attempt does not propose a concrete product-code fix under `web/src/**`.
- Cap at `15/100` if the attempt changes reviewer-visible copy, accessible names, or screenshots to make the layout pass.
- Cap at `25/100` if the attempt covers only one viewport width.
- Cap at `20/100` if the stale screenshot thread is treated as actionable.

## Scoring Procedure
1. Score each category from the solver's actual attempt output.
2. Apply caps after raw scoring.
3. Record the capped result in `benchmark_run.md`.

## Evidence Rules
- Full grounding points require concrete thread ids and current-screenshot evidence, not placeholder ids or generic restatements of the task spec.
- Full layout points require a credible DOM or CSS change path, not generalized “fix responsive layout” prose.
- Full a11y points require keyboard focus reasoning, not mouse-hover styling.
- Full coverage points require all three named widths and at least one hidden-regression guard.
- Full reply points require both `artifacts/review_reply.json` and `web/docs/review_response.md` behavior.

## Hardness Target
- Calibration target for a naive `GPT-5.4/high` solver: about `20/100`
- Upper guardrail: if a naive solver appears to score above `30/100`, harden the task and rerun
- Lower guardrail: if a serious solver falls below `10/100` because the blueprint is incoherent, clarify the task and rerun
