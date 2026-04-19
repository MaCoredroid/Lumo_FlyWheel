# Benchmark Run: Draft Mode Content Preview Sync

## Run Metadata

- Date: 2026-04-18
- Agent model: `gpt-5.4`
- Reasoning effort: `high`
- Agent id: `019da332-5c2f-76e0-bdb9-23de5285cd15`
- Family-local context: the child agent was instructed to inspect only this family directory and attempt the benchmark task honestly without editing files.

## Attempt Summary

The child agent produced:

- A plausible diagnosis that preview body selection and preview status derivation are split and likely inconsistent.
- A concrete patch plan targeting `content_adapter/*`, `site/*`, `config/preview.toml`, tests, and `docs/draft_preview.md`.
- A verification plan that correctly emphasized the known draft article, a published-only article in preview mode, and same-revision consistency.

The child agent did not produce:

- Any real code or config patch.
- Trusted preview-browser evidence, fixture-linked proof, or revision-linked proof.
- Real screenshots, tests, or cross-article runtime checks.

## Scoring Against `evaluator_contract.md`

Raw scoring:

1. Root-cause investigation quality: `12/15`
2. Concrete workspace edits: `13/20`
3. Preview/browser evidence: `0/20`
4. Regression and anti-shortcut handling: `10/15`
5. Deliverables and docs: `4/10`
6. Family-specific correctness invariants: `2/20`

Raw total: `41/100`

Caps applied:

- `no_runtime_preview_evidence` cap: `25`
- `spec_only_reasoning` cap: `20`

Final score: `20/100`

## Why The Score Stopped At 20

- The attempt never showed real preview freshness, real status-chip behavior, or same-revision proof.
- No published-only preview fallback case was actually run.
- The submission remained a strong artifact-only repair plan rather than a verified benchmark solution.

## Post-Run Judgment

- Result is in the target hardening band: `yes`
- Meaningful benchmark family: `yes`
- Further hardening required before freeze: `no`

The family remains meaningful because same-revision and cross-article proof are hard requirements that an artifact-only solver cannot fake.
