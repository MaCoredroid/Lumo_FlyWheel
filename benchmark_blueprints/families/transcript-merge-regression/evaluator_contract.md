# Evaluator Contract

## Scope
- Family: `transcript-merge-regression`
- Task: `cnb55-core-transcript-merge-regression-reducer-diagnosis`
- Runtime class: telemetry repo with merge reducer, render layer, incident summary, and JSONL session artifacts

## Evaluation Intent
Measure whether the solver identifies and repairs the true merge invariant instead of suppressing visible duplicate symptoms. The evaluator should reward stable event identity reasoning and preserve incident summary semantics.

## Required Submission Artifacts
- Merge-layer code patch
- Regression tests for merge identity and summary invariants
- Incident note update

## Scoring Model
- `100` total points, awarded by category:
  - `45`: merge-layer repair and event-identity correctness
  - `20`: regression protection across merge, render, and summary tests
  - `15`: incident-summary preservation
  - `10`: incident note quality
  - `10`: shortcut resistance across hidden traces

## Score Caps
- Cap at `10/100` for note-only or analysis-only submissions.
- Cap at `20/100` for render-layer duplicate filtering or post-completion dropping strategies.
- Cap at `20/100` for plausible merge commentary with no code-level invariant repair.
- Cap at `25/100` if stable identity is recognized but deferred-output handling remains weak.
- Cap at `30/100` if incident summary semantics or note alignment remain stale.
- Cap at `50/100` if merge logic is correct but robustness coverage is weak.

## Visible Gates
- `pytest -q tests/test_merge.py tests/test_render.py tests/test_incident_summary.py`

## Hidden Gates
- Same-name tool outputs sharing role but not event identity must both survive
- Interleaved fragments must merge correctly
- Legitimate deferred tool output after completion must survive
- Debug-only post-completion noise must not create a render block
- Incident summary totals must remain stable
- Grep or AST guard rejects post-render duplicate filtering as the primary fix

## Integrity And Shortcut Checks
- Reject dropping all post-completion fragments
- Reject re-keying from visible-fixture sequence indexes
- Reject note-only justifications for a wrong render-layer fix
- Reject deletion or weakening of summary regression tests

## Variant Hardness Notes
- `V1`: visible artifacts expose the issue but not the full invariant
- `V2`: noisy duplicate debug lines are deliberate distractors
- `V3`: dirty telemetry experiments must be preserved
- `V4`: incident note and before-after reasoning are mandatory
- `V5`: follow-up injects evidence that a symptom-only fix regressed legitimate deferred output

## Current Hardness Judgment
- Actual recorded solver run: `20/100`
- Naive `gpt-5.4/high` above `30/100`: `unlikely under current rubric`
