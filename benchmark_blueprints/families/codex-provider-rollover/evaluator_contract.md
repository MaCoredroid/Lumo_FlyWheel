# Evaluator Contract: Codex Provider Rollover

## Purpose
Evaluate whether the solver repaired the real Responses-path maintenance profile rather than making a cosmetic config or one-turn smoke fix.

## Required submission surfaces
- `.codex/config.toml` or equivalent provider config
- Smoke or health-check script
- Operator docs
- Verification note

## Visible gates
1. Config parsing tests pass.
2. Smoke tests require exact turn-two continuity using `previous_response_id`.
3. Docs tests confirm provider id and endpoint references are consistent.

## Trusted checks
- The selected default provider must be the proxy-backed Responses route.
- Proxy base URL and provider id are validated together.
- Existing local tuning knobs survive byte-for-byte.
- The smoke probe asserts exact follow-up chaining, not just second-call success.

## Hidden checks
- Reordered-key, commented config file to reject wholesale regeneration.
- First-turn-success fixture where turn two fails unless the correct provider/store path is used.
- Cross-doc consistency checks on commands, provider id, and config path.

## 100-point scoring breakdown
- `15` points: correct diagnosis of the rollover failure and why one-turn smoke is insufficient.
- `25` points: selected default provider and proxy-backed Responses path are repaired correctly in the concrete config. Proposed-only edits score at most `5/25`.
- `20` points: exact `previous_response_id` continuity and follow-up chaining checks demonstrated by the concrete smoke. Proposed-only edits score at most `4/20`.
- `15` points: preservation of unrelated local tuning keys and comments in the concrete file. No modified file yields `0/15`.
- `10` points: operator docs aligned to the repaired runtime path. Proposed-only edits score at most `2/10`.
- `5` points: verification note describing turn-one and turn-two behavior.
- `10` points: hidden robustness against reordered-key and misleading-success fixtures. No executed evidence yields `0/10`.

## Hard caps
- Cap at `20` if the default provider remains legacy-selected.
- Cap at `20` if exact `previous_response_id` continuity is not asserted.
- Cap at `30` if unrelated local tuning keys are dropped or reordered destructively.
- Cap at `25` for diagnosis-only or plan-only submissions that do not modify the concrete config and run the strengthened smoke.

## Shortcut defenses
- Validate raw config preservation, not just parsed values.
- Tie provider selection, proxy URL, and continuity into one gated contract.
- Reject docs-only alignment when runtime config remains stale.

## Final hardness judgment
- Current naive GPT-5.4/high outlook: under `30/100` after hardening.
- Confidence: medium.
- Main reason: continuity and default-provider semantics are now explicit gates with low score caps.
