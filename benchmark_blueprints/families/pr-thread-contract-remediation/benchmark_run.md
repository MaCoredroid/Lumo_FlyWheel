# Benchmark Run

## Run 1
- `date`: `2026-04-18`
- `agent_id`: `019da331-d79a-7d80-8a2a-153d85316fc6`
- `model`: `gpt-5.4`
- `reasoning_effort`: `high`
- `bundle_root`: `benchmark_blueprints/families/pr-thread-contract-remediation`
- `attempt_mode`: blueprint-family bundle only; the referenced `review/`, `artifacts/`, and `repo/` trees were not present

## Submission Summary
- The solver correctly treated the family bundle as review-driven and distinguished that exactly three threads should be actionable and two should be non-actionable.
- It proposed coherent file-level changes in `queue_api/serializers.py`, `queue_api/service.py`, and conditionally `queue_api/handlers.py`.
- It proposed strong regression coverage for omitted-vs-null behavior and stable bucket ordering.
- It drafted plausible reviewer-reply content and the required release-note update.
- It could not ground thread ids, thread bodies, schema fields, or test execution because the seeded review and repo artifacts were absent from the family bundle.

## Scoring
- Review-state triage accuracy: `8/25`
- Proposed remediation correctness: `18/20`
- Ordering and serialization regression coverage: `14/15`
- Release-note and contract communication quality: `14/15`
- Reviewer-reply quality: `7/15`
- Constraint discipline: `10/10`

Raw score: `71/100`

## Caps Applied
- Cap at `20/100` applied because the attempt did not ground its triage in explicit thread ids or concrete thread-state evidence from `review/pr_481_threads.json`.

Final score: `20/100`

## Judgment
- Calibration target met: `yes`
- Naive `GPT-5.4/high` appears able to reason about the task shape, but not exceed `30/100` without the actual thread export and schema-grounded reply evidence.
- Rerun required: `no`
