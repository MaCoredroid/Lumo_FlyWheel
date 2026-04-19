# Benchmark Run: Preview Auth Cookie Handoff

## Run Metadata

- Date: 2026-04-18
- Agent model: `gpt-5.4`
- Reasoning effort: `high`
- Agent id: `019da332-5118-7090-a1e0-23f596e04f46`
- Family-local context: the child agent was instructed to inspect only this family directory and attempt the benchmark task honestly without editing files.

## Attempt Summary

The child agent produced:

- A plausible diagnosis centered on preview-host cookie scope with redirect-origin drift as a secondary cause.
- A concrete patch plan targeting `config/preview_host.toml`, `docs/auth_preview.md`, a likely auth guard in `web/`, and the session/callback writer in `auth_proxy/`.
- A verification plan that correctly emphasized fresh-profile login, reload persistence, cookie clearing, and preview-host cookie inspection.

The child agent did not produce:

- Any real code patch or diff.
- Trusted preview-browser runtime evidence.
- Screenshots, network captures, or test results from the benchmark workspace.

## Scoring Against `evaluator_contract.md`

Raw scoring:

1. Root-cause investigation quality: `12/15`
2. Concrete workspace edits: `13/20`
3. Preview/browser evidence: `0/20`
4. Regression and anti-shortcut handling: `9/15`
5. Deliverables and docs: `4/10`
6. Family-specific correctness invariants: `2/20`

Raw total: `40/100`

Caps applied:

- `no_runtime_preview_evidence` cap: `25`
- `spec_only_reasoning` cap: `20`

Final score: `20/100`

## Why The Score Stopped At 20

- The attempt never left the family bundle and therefore supplied no trusted preview runtime evidence.
- The attempt inferred likely fixes from the family artifacts but did not disambiguate the real bug with live browser/network observations.
- The attempt did not demonstrate session persistence, cookie semantics, or origin rejection in a running preview.

## Post-Run Judgment

- Result is in the target hardening band: `yes`
- Meaningful benchmark family: `yes`
- Further hardening required before freeze: `no`

The dominant limiter was the intended artifact-only cap, which keeps a naive GPT-5.4/high family-local attempt near 20/100.
