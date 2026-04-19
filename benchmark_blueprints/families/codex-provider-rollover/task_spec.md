# Codex Provider Rollover

Scenario type: `config_rollover_and_smoke`

## Task prompt
Restore the local Responses-path maintenance profile after a provider rollover. The repo’s `.codex/config.toml`, the maintenance smoke probe, and the operator docs drifted after the serving stack moved from a direct vLLM endpoint to the local proxy-backed Responses path. A one-turn smoke still passes, but follow-up chaining fails because the config points at the wrong provider id and the smoke script does not assert `previous_response_id` continuity. Repair the maintenance profile so the configured Codex path, smoke probe, and docs agree again. Preserve unrelated experimental config knobs already present in the local file.

## Workspace bundle
- `serving_maintenance/.codex/config.toml` with one stale provider stanza, one active provider stanza, and locally edited experimental knobs that must survive.
- `serving_maintenance/scripts/smoke_responses_profile.py` that validates turn one but not follow-up chaining.
- `serving_maintenance/fixtures/http/turn1_ok.json` and `turn2_missing_store.json` showing the current failure mode.
- `serving_maintenance/docs/provider_rollover.md` and `docs/smoke.md` with partially outdated command examples and endpoint references.
- `serving_maintenance/tests/` covering TOML loading, smoke behavior, and docs snippets; hidden tests assert that the repaired smoke checks provider id, proxy base URL, and multi-turn continuity without deleting the user’s local tuning keys.

## Seeded automation or maintenance drift
- The config still targets a legacy provider name and direct backend URL even though the maintenance profile is supposed to use the proxy-backed Responses route.
- The smoke script accepts any successful first response id and never probes turn-two retrieval or `previous_response_id` chaining.
- Docs still describe a valid-looking but wrong direct-endpoint workflow.
- The local config contains extra speed knobs; naïve rewrites that replace the entire file will lose them and fail hidden checks.

## Required surfaces
- `shell`
- `apply_patch`
- `automation`
- TOML diff reasoning
- Terminal smoke execution against fixture-backed responses

## Deliverables
- A repaired `.codex/config.toml` or equivalent provider config artifact that preserves unrelated local tuning keys.
- A code patch for the smoke probe so it asserts multi-turn continuity on the maintained Responses path.
- A docs update for the operator-facing rollover instructions.
- A short verification note showing the repaired one-turn and follow-up expectations.

## Grader contract
- Visible tests must pass for config parsing, smoke probe behavior, docs references, and exact turn-two continuity using `previous_response_id`.
- Trusted checks verify that:
  - The active provider stanza points at the proxy-backed Responses route rather than the deprecated direct endpoint.
  - The selected default provider id and proxy base URL are validated together rather than as separate strings.
  - Experimental local tuning keys present in the seeded file are preserved.
  - The smoke probe explicitly checks follow-up chaining using `previous_response_id`.
  - The legacy provider may remain in the file for history, but it must not remain selected as default.
- Hidden checks introduce:
  - A config file with reordered keys and comments to punish wholesale regeneration.
  - A misleading first-turn success fixture where turn two fails unless the correct provider and store path are used.
  - A docs snippet test ensuring the command examples and config path all point at the same provider id.
  - A raw-file preservation check that unrelated local tuning lines survive byte-for-byte.
- Scoring cap: if the default provider remains legacy-selected, or if turn-two continuity is not asserted exactly, the submission cannot score above `20/100`.

## Red-team traps
- Rewrite the whole TOML file from scratch and drop unrelated local knobs.
- Make the smoke probe assert only that turn one returns an id.
- Patch docs alone while leaving the configured provider id stale.
- Hardcode the visible response ids or fixture filenames.
- Add a second provider stanza and leave the legacy one selected as default.

## Quality gate
- Actual GPT-5.4/high family-bundle run score: `21/100` under the current evaluator.
- Naive GPT-5.4/high under-30 verdict after hardening: `yes`.
- Run method note: actual child-agent solver attempt against the family bundle only; concrete `serving_maintenance/` workspace files were not present in this authoring repo, so the submission could outline the repair but not prove continuity or key preservation by execution.
- Hardening applied after probe:
  - Follow-up chaining is required, so a one-turn smoke fix is intentionally incomplete.
  - Hidden tests preserve comments and reordered keys to punish file replacement instead of targeted edits.
  - The grader requires preservation of local tuning keys, blocking simplistic template overwrite strategies.
  - Exact `previous_response_id` continuity is now part of visible grading, not a hidden nice-to-have.
  - Default-provider selection and proxy base URL are now coupled, closing the “fix one field only” path.
  - Wrong default-provider semantics or missing exact continuity now hard-cap the score at 20.
- Residual risk accepted: a solver might still patch the right fields by luck, but without continuity semantics and byte-preservation it should remain in the low 20s.
