# Benchmark Run

- Family: `skill-entrypoint-drift`
- Task: `t9_skill_entrypoint_drift_ops_digest`
- Child agent: `019da332-4b36-7843-8ecb-9e045f1ed899`
- Model: `gpt-5.4`
- Reasoning: `high`
- Result: `completed`
- Target band: `15-25/100`

## Actual Attempt Summary
- The child agent reconciled the visible skill doc, example, runner, reference contract, helper loader, fixture, and lightweight visible tests.
- It standardized the visible command path on `tools/digest_runner.py`, introduced a visible `--summary-length` contract, and made the example runnable from the documented locations.
- It did not weaken the trusted smoke harness file.

## Commands Reported
- `python tools/digest_runner.py --input fixtures/incidents/sample_events.json --summary-length short`
- `python ../../tools/digest_runner.py --input ../../fixtures/incidents/sample_events.json --summary-length short`
- `python tools/digest_runner.py --input fixtures/incidents/sample_events.json --summary-length long`
- `python scripts/smoke_skill.py skills/ops-digest/SKILL.md`
- `pytest -q`

## Scoring Against Evaluator
- `5/5`: visible skill entrypoint updated to a plausible current runner.
- `5/5`: visible example reconciled with the skill instructions.
- `5/5`: obvious visible path and flag drift repaired without weakening the harness.
- `5/5`: visible skill remained concrete and executable rather than vague.
- `0/80`: hidden contract still withholds the exact canonical runner flags, heading shape, and stricter portability checks.
- Total: `20/100`

## Judgment
- In target band: `Yes`
- Naive `gpt-5.4/high` solver still looks meaningfully constrained: `Yes`
- Rerun needed: `No`

## Notes
- This run hit the visible cap on the first actual attempt.
- The remaining score is intentionally reserved for executable truth under withheld command-block and output-shape checks.
