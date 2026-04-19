# Benchmark Run: Preview Form Webhook CORS

## Run Metadata

- Date: 2026-04-18
- Agent model: `gpt-5.4`
- Reasoning effort: `high`
- Agent id: `019da332-595e-7a51-b273-11de7e70c7f2`
- Family-local context: the child agent was instructed to inspect only this family directory and attempt the benchmark task honestly without editing files.

## Attempt Summary

The child agent produced:

- A plausible diagnosis centered on preview API target drift plus incomplete CORS handling across success and validation paths.
- A concrete change plan targeting `config/preview.env.toml`, frontend submit wiring, API CORS handling, docs, and tests.
- A verification plan that correctly emphasized preflight, valid submit, invalid submit, and the actual preview API host used by the browser.

The child agent did not produce:

- Any real code/config patch.
- Trusted browser-network evidence from preview.
- Any real screenshots, header captures, or test results.

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

- The attempt never established the real preview API target or observed the real browser/network failure.
- No proof was provided for preflight behavior, success-path headers, or validation-path headers.
- The submission remained a high-quality plan rather than a validated benchmark solution.

## Post-Run Judgment

- Result is in the target hardening band: `yes`
- Meaningful benchmark family: `yes`
- Further hardening required before freeze: `no`

The cap structure worked as intended: a smart artifact-only plan still cannot rise above 20/100 without real preview-network evidence.
