# Skill Router Contract Upgrade

## Companion Artifacts

- `evaluator_contract.md` defines score caps, hidden invariants, and fail-fast conditions
- `codex/config.toml` defines the configured Codex profile for this family
- `skills/audit-skill-routing/SKILL.md` provides bounded benchmark-local guidance
- `benchmark_run.md` records actual GPT-5.4/high benchmark attempts, scoring, and resulting hardening

## Task Prompt

The repo contains a small skill router that decides which repo-local skill to recommend for a task request. The old router assumed each skill declared a single `trigger` string, but the newer contract allows `triggers`, `negative_triggers`, and `required_inputs`. Upgrade the router and contract docs so the shipped skill bundle routes correctly under the new schema without breaking fallback behavior. Preserve any unrelated in-progress edits.

## Workspace Bundle

- `router/skill_router.py` or `router/skill_router.ts`
- `skills/*/SKILL.md` for three or four concrete repo-local skills
- `config/skill_router.toml` defining default ranking and fallback behavior
- `fixtures/router_cases.yaml` with realistic routing inputs
- `tests/test_skill_router.py` and `tests/test_router_contract.py`
- `docs/skill_routing.md` with stale schema examples

## Seeded Skill Or Tooling Drift

- Router still reads `trigger` and ignores multi-field contract inputs
- Fallback ranking still assumes missing metadata means “best match”
- Docs describe the new schema, but the implementation only partially adopted it
- Hidden tests include ambiguous prompts where `negative_triggers` must suppress an otherwise plausible skill

## Required Surfaces

- Repo search and shell
- Source edits in router logic plus TOML or fixture alignment
- Markdown docs update
- Terminal tests

## Deliverables

- Patch upgrading router logic to the new skill contract
- Updated docs and config examples matching the implemented schema
- Short routing audit note showing one positive and one suppressed match

## Grader Contract

- Phase 1 visible checks:
  - `pytest -q tests/test_skill_router.py tests/test_router_contract.py`
  - Visible tests should include at least one case where `negative_triggers` suppresses an otherwise strong keyword match
  - Visible tests should include at least one case where unmet `required_inputs` makes a skill ineligible
- Phase 2 hidden checks:
  - Route ambiguous prompts and verify `negative_triggers` suppress false positives
  - Verify missing `required_inputs` prevents over-eager routing
  - Confirm fallback only activates when no eligible skill remains after suppression
  - Parse docs and config examples and schema-validate them against the live multi-field keys
  - Reject solutions that delete ambiguous fixtures or weaken routing assertions
  - Parse docs and config examples to ensure they use the live schema keys
- Score shaping:
  - Visible checks alone cap the score at `20/100`
  - Any solution that ignores `negative_triggers`, ignores `required_inputs`, or weakens ambiguous fixtures caps at `15/100`
  - Exceeding `30/100` requires passing hidden ambiguity and fallback-order checks
- Pass condition:
  - Router behavior, docs, and config all match the upgraded multi-field contract

## Red-Team Traps

- Support `triggers` only and ignore `negative_triggers`
- Treat missing `required_inputs` as truthy and over-route
- Force the preferred fallback skill for all ambiguous prompts
- Update docs to match the old code instead of implementing the new contract
- Remove ambiguous fixtures instead of fixing ranking logic

## Quality Gate

- Probe goal: naive GPT-5.4/high should not pass by editing one parser branch
- Hardening baked in before probe:
  - Hidden cases require both suppression and fallback correctness
  - Docs and config are graded, not just source behavior
  - Ambiguous prompts make shallow keyword matching insufficient
- Probe outcome:
  - Adversarial GPT-5.4/high judged the initial spec too easy; a solver could likely exceed `30/100` by adding list support for `triggers` and touching docs/config without fully enforcing suppressors or input gates
- Additional hardening after probe:
  - Elevated one suppressor case and one missing-input case into visible coverage
  - Added schema validation for docs/config examples
  - Added anti-fixture-deletion language
  - Added a scoring clamp so visible routing fixes cannot exceed `20/100`
- Final naive GPT-5.4/high assessment:
  - Under-30 now looks plausible for a naive solver because parser-only upgrades no longer clear the score clamp or the fallback/suppression gates

## V1-V5 Instantiation

- V1 clean baseline: upgrade one parser path and one docs sample
- V2 noisy reality: duplicate trigger vocab, stale routing notes, and misleading fixtures
- V3 dirty workspace: local edits in one skill’s metadata must be preserved
- V4 multi-surface: router code, TOML, docs, and routing evidence note all required
- V5 recovery-in-thread: first fix over-routes after adding `triggers`, then must handle suppressors narrowly
