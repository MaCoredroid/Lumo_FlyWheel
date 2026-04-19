# Benchmark Run: Responsive Checkout Visual Regression

## Run Metadata

- Date: 2026-04-18
- Agent model: `gpt-5.4`
- Reasoning effort: `high`
- Agent id: `019da332-53ce-7070-8c4b-582532105107`
- Family-local context: the child agent was instructed to inspect only this family directory and attempt the benchmark task honestly without editing files.

## Attempt Summary

The child agent produced:

- A plausible diagnosis of mobile-only CTA occlusion in the preview compact-summary path.
- A concrete patch plan targeting checkout components, checkout styles, and `tests/e2e/checkout.spec.ts`.
- A verification plan that correctly emphasized mobile clickability, a second mobile viewport, and tablet/desktop preservation.

The child agent did not produce:

- Any real code patch or style diff.
- DOM geometry, hit-testing, or screenshot evidence from the actual preview.
- Any actual test changes or results.

## Scoring Against `evaluator_contract.md`

Raw scoring:

1. Root-cause investigation quality: `11/15`
2. Concrete workspace edits: `13/20`
3. Preview/browser evidence: `0/20`
4. Regression and anti-shortcut handling: `9/15`
5. Deliverables and docs: `3/10`
6. Family-specific correctness invariants: `2/20`

Raw total: `38/100`

Caps applied:

- `no_runtime_preview_evidence` cap: `25`
- `spec_only_reasoning` cap: `20`

Final score: `20/100`

## Why The Score Stopped At 20

- The attempt relied on family-bundle inference and never inspected a real preview DOM, geometry, or hit-testing result.
- No actual screenshots, viewport assertions, or clickability proof were produced.
- The attempt could not prove the compact-summary experiment remained active after the hypothetical fix.

## Post-Run Judgment

- Result is in the target hardening band: `yes`
- Meaningful benchmark family: `yes`
- Further hardening required before freeze: `no`

The dominant limiter was the lack of trusted multi-viewport runtime evidence, which is exactly what this family is meant to demand.
