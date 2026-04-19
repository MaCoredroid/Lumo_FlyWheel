# Benchmark Run: Checkout Screenshot Triage

## Run Metadata
- Model: `gpt-5.4`
- Reasoning effort: `high`
- Agent id: `019da331-b980-78a2-a750-137c93cb0f77`
- Run context: family package only under this directory
- Files available to solver:
  - `task_spec.md`
  - `evaluator_contract.md`
  - `codex/config.toml`
  - `skills/mobile-checkout-triage/SKILL.md`

## Actual Solver Attempt
The child agent attempted the task from the family package and produced:
- A concrete runtime-fix hypothesis centered on mobile layout growth, sticky-summary interaction, and bad post-validation focus handoff.
- A repair plan that preserves desktop behavior and calls for safe-area and viewport-sensitive handling.
- A release note draft.
- An explicit limitation notice that no actual code patch, screenshots, trace output, or test results could be produced from the package alone.

Representative solver statement:

> “The regression is most likely caused by mobile checkout layout logic that assumes a stable sticky-summary/footer footprint… Separately, validation failure focus is being sent to a non-actionable element instead of the visible error target or first invalid control.”

## Scoring Breakdown
- Root-cause repair across the affected mobile checkout state: `14/25`
- Robust mobile layout behavior across viewport, safe-area, and content variations: `3/20`
- Correct post-validation focus target and accessibility behavior: `4/15`
- Desktop and sticky-summary non-regression: `2/15`
- Fresh machine-generated evidence from the repaired flow: `0/10`
- Test integrity and anti-shortcut discipline: `4/10`
- Release note quality: `5/5`

Raw score: `32/100`

## Applied Caps
- No fresh machine-generated evidence: cap `25/100`
- No runtime checkout fix actually submitted: cap `20/100`

Final scored run: `20/100`

## Judgment
- Target band (`15-25`): met
- Coherence judgment: coherent
- Rerun required: no

## Why The Score Lands Near 20
The solver extracted the likely failure mode from the family package and produced a plausible plan, but it could not submit the runtime fix or the required fresh evidence. The explicit caps keep the calibration run from over-scoring on text-only reasoning.

