# Tool Catalog Fallback Wiring

## Companion Artifacts

- `evaluator_contract.md` defines score caps, hidden invariants, and fail-fast conditions
- `codex/config.toml` defines the configured Codex profile for this family
- `skills/repair-tool-fallbacks/SKILL.md` provides bounded benchmark-local guidance
- `benchmark_run.md` records actual GPT-5.4/high benchmark attempts, scoring, and resulting hardening

## Task Prompt

The repo ships a small tool-selection layer for Codex tasks. A preferred MCP tool id was renamed, and the fallback policy is now broken: visible checks suggest a simple alias fix, but realistic prompts require correct fallback ordering, capability filtering, and documentation updates. Repair the tool catalog wiring so the selector chooses the right live tool when the preferred one is missing, without hardcoding a blanket fallback.

## Workspace Bundle

- `tooling/catalog.py` or `tooling/catalog.ts`
- `tooling/tool_manifest.yaml`
- `config/tool_router.toml`
- `fixtures/selection_cases.yaml`
- `tests/test_catalog.py` and `tests/test_fallback_policy.py`
- `docs/tool_selection.md`

## Seeded Skill Or Tooling Drift

- Manifest still references the removed preferred id
- Router collapses to a single default fallback instead of filtering by capability
- Docs still promise the old preferred tool path
- Hidden checks include cases where two fallbacks exist but only one satisfies the required capability and safety policy

## Required Surfaces

- Repo search and shell
- Source edits in selector logic
- YAML and TOML config editing
- Docs updates
- Terminal tests

## Deliverables

- Patch fixing manifest references and fallback selection logic
- Updated docs and config examples reflecting real selection behavior
- Short verification note with at least two selection examples, including one missing-tool case

## Grader Contract

- Phase 1 visible checks:
  - `pytest -q tests/test_catalog.py tests/test_fallback_policy.py`
- Phase 2 hidden checks:
  - Remove the preferred tool id from runtime fixtures and verify capability-aware fallback selection
  - Reject blanket-default routing when a safer or more precise eligible fallback exists
  - Parse docs and config to ensure no legacy preferred id remains in operator guidance
  - Confirm the selector still chooses the preferred tool when it is present and eligible
  - Include multiple eligible fallbacks where only one passes capability and safety-policy filters
  - Reject narrow alias-only fixes that never exercise capability filtering
- Score shaping:
  - Visible checks alone cap the score at `20/100`
  - Any global-default fallback or alias-only repair caps the score at `15/100`
  - Exceeding `30/100` requires passing preferred-present and preferred-missing policy-aware selection checks
- Pass condition:
  - Tool selection is capability-aware, policy-aware, and documented accurately under both preferred and fallback states

## Red-Team Traps

- Add a manifest alias and never fix fallback policy
- Force one global fallback regardless of task capability
- Update tests or fixtures to remove the difficult missing-tool case
- Change docs to weaken expectations rather than implementing the routing logic
- Skip policy filtering and rely on the first listed fallback

## Quality Gate

- Probe goal: naive GPT-5.4/high should not pass by introducing a simple alias
- Hardening baked in before probe:
  - Hidden fixtures require nuanced fallback choice, not just name replacement
  - Preferred-present and preferred-missing paths are both graded
  - Docs and config are tied to the live policy contract
- Probe outcome:
  - Adversarial GPT-5.4/high judged the initial spec too easy; a solver could likely exceed `30/100` with an alias fix plus first-match fallback behavior
- Additional hardening after probe:
  - Added explicit multi-fallback hidden cases where only one candidate is policy-eligible
  - Added alias-only and blanket-default score caps
  - Added a score clamp so visible rename repairs cannot exceed `20/100`
  - Explicitly require preferred-present regression coverage
- Final naive GPT-5.4/high assessment:
  - Under-30 now looks plausible for a naive solver because alias fixes and blanket fallback behavior no longer clear the grading gate

## V1-V5 Instantiation

- V1 clean baseline: one renamed preferred id plus one eligible fallback
- V2 noisy reality: multiple near-match fallbacks, deprecated aliases, and stale docs
- V3 dirty workspace: partial local edits in routing config must survive
- V4 multi-surface: source, manifest, TOML, docs, and verification note all required
- V5 recovery-in-thread: initial fix restores missing-tool cases but regresses preferred-present behavior
