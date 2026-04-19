# Benchmark Run

## Run Identity
- `family_id`: `oauth-preview-browser-recovery`
- `task_id`: `oauth-preview-browser-recovery/fake-idp-callback-drift`
- `runner_model`: `gpt-5.4`
- `runner_reasoning_effort`: `high`
- `agent_id`: `019da331-e747-7d92-afb2-0d6864c16f44`
- `agent_nickname`: `Hume`
- `run_type`: design-time family-bundle solve attempt

## Runner Context
The child agent was instructed to solve using only:
- `task_spec.md`
- `evaluator_contract.md`
- `codex/config.toml`
- `skills/preview-oauth-reality-check/SKILL.md`

No actual `web/`, `api/`, `idp/seed/`, compose stack, or screenshot artifacts existed in the family directory.

## Attempt Summary
The runner correctly diagnosed callback-host and audience drift across frontend, backend, and fake IdP registration. It named the correct patch surfaces, described a clean-browser-state validation path, and specified the right token provenance checks. It also explicitly refused to claim a real solve because no runnable preview stack or implementation files existed in the directory.

## Scoring Against `evaluator_contract.md`
- Correctly identify the real auth drift: `9/10`
  - Strong diagnosis of callback-host, audience, and stale-cookie traps.
- Target the right patch surfaces: `8/10`
  - Correctly named frontend, backend, IdP seed, runtime config, and docs.
- Implement the real handshake fix: `0/35`
  - No code or config was actually changed.
- Trusted auth correctness: `0/20`
  - No real IdP-issued token, redirect-chain run, or reload persistence proof.
- Browser proof: `0/10`
  - No screenshots from a live authenticated session.
- Operator artifact quality: `3/10`
  - Strong deploy-note content plan, but no actual artifact edits.
- Shortcut resistance: `0/5`
  - No bypass was attempted, but no executable proof was produced either.

## Final Score
- `final_score = 20/100`

## Judgment
- Result is in the target naive-solver band around 20.
- `under_30_for_naive_gpt5_4_high = yes`
- No rerun was required after this scored attempt.

## Notes
- The family stays meaningful because the child runner understood the real OAuth constraints, yet still could not collect any of the score-driving implementation, token, or browser-evidence points.
