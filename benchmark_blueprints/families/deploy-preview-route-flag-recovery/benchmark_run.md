# Benchmark Run: Deploy Preview Route Flag Recovery

## Run Metadata

- Date: 2026-04-18
- Agent model: `gpt-5.4`
- Reasoning effort: `high`
- Agent id: `019da332-5d6b-77d2-bb04-72634ad291fb`
- Family-local context: the child agent was instructed to inspect only this family directory and attempt the benchmark task honestly without editing files.

## Attempt Summary

The child agent produced:

- A plausible diagnosis that preview route resolution, preview flag override, and preview seed behavior drifted apart.
- A concrete patch plan targeting route code, flag definitions, `deploy/preview_manifest.toml`, seed loading, docs, and tests.
- A verification plan that correctly emphasized distinguishing `404` from empty-shell failure, seeded-record identity, and non-preview gating.

The child agent did not produce:

- Any real code/config patch.
- Trusted browser evidence, seeded-record proof, or gating proof from the actual preview.
- Any real screenshots or test results.

## Scoring Against `evaluator_contract.md`

Raw scoring:

1. Root-cause investigation quality: `12/15`
2. Concrete workspace edits: `14/20`
3. Preview/browser evidence: `0/20`
4. Regression and anti-shortcut handling: `10/15`
5. Deliverables and docs: `4/10`
6. Family-specific correctness invariants: `2/20`

Raw total: `42/100`

Caps applied:

- `no_runtime_preview_evidence` cap: `25`
- `spec_only_reasoning` cap: `20`

Final score: `20/100`

## Why The Score Stopped At 20

- The attempt never proved canonical-key consistency, real route resolution, or seeded-content provenance in a live preview.
- No non-preview gating behavior was demonstrated.
- The submission remained a route/flag/seed repair plan, not a verified benchmark solution.

## Post-Run Judgment

- Result is in the target hardening band: `yes`
- Meaningful benchmark family: `yes`
- Further hardening required before freeze: `no`

The artifact-only attempt stayed at 20/100 because the family requires real preview evidence across route, flag, and seed surfaces.
