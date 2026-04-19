# Skill Pack Contract Drift

## Companion Artifacts

- `evaluator_contract.md` defines score caps, hidden invariants, and fail-fast conditions
- `codex/config.toml` defines the configured Codex profile for this family
- `skills/repair-skill-contract/SKILL.md` provides bounded benchmark-local guidance
- `benchmark_run.md` records actual GPT-5.4/high benchmark attempts, scoring, and resulting hardening

## Task Prompt

You are working in a repo-local Codex skill pack. A helper command was renamed from `skill-smoke` to `skill-verify`, but the shipped skill documentation, prompt examples, and Codex config still advertise the old flow. Fix the skill pack so a new operator can follow the documented workflow end to end without hitting stale commands or mismatched output expectations. Preserve unrelated local edits. Do not rewrite the skill from scratch; repair the contract between docs, config, fixtures, and the validation helper.

## Workspace Bundle

- `skills/release_handoff/SKILL.md` with example invocations and required output shape
- `skills/release_handoff/examples/` with before/after sample prompts and expected report snippets
- `codex/config.toml` mapping named skills to smoke commands
- `scripts/skill_verify.py` as the true validator entrypoint
- `tests/test_skill_contract.py` and `tests/test_examples.py`
- `docs/skill_pack_notes.md` with partially stale operator guidance
- Dirty-workspace variant may include unrelated edits under another skill directory and untracked prompt capture files

## Seeded Skill Or Tooling Drift

- Docs and config still point at `python scripts/skill_smoke.py`
- Example outputs still describe the old one-section report instead of the current two-section report
- One visible test only checks for the renamed command token, while hidden checks validate the full documented flow and output contract

## Required Surfaces

- Repo search and shell
- File editing across Markdown, TOML, and small helper code
- Terminal test execution
- Skill-surface reasoning about how `SKILL.md` examples map to the validator contract

## Deliverables

- Patch aligning `SKILL.md`, prompt examples, and `codex/config.toml`
- Minimal code or fixture changes only if needed to preserve the documented contract
- Short validation note proving the documented command path and expected output shape are consistent

## Grader Contract

- Phase 1 visible checks:
  - `pytest -q tests/test_skill_contract.py tests/test_examples.py`
- Phase 2 hidden checks:
  - Execute every shell command example extracted from `SKILL.md` instead of only grepping for the renamed helper token
  - Parse `SKILL.md` and assert every command example resolves to the live validator entrypoint
  - Re-run the documented example flow and verify the generated report contains both required sections in the documented order
  - Confirm `codex/config.toml` skill mapping matches the docs and does not retain the legacy helper name
  - Reject any retained `skill_smoke` alias or compatibility shim
  - Reject solutions that weaken tests or example assertions instead of repairing the contract
  - Confirm unrelated files outside the family task surface were not modified
- Score shaping:
  - Visible checks alone cap the score at `20/100`
  - Any retained legacy alias, shim, or weakened assertion caps the score at `10/100`
  - Exceeding `30/100` requires passing the hidden executable-workflow contract
- Pass condition:
  - The documented workflow is executable, the config wiring matches it, and the examples reflect the real output contract

## Red-Team Traps

- Replace the old command string in docs only and leave config stale
- Add a compatibility shim for `skill_smoke` without updating the true contract
- Hardcode expected output text into an example fixture without making the validator emit it
- Delete or loosen example assertions instead of repairing the workflow
- Rewrite the skill doc broadly while dropping required operator constraints from the original instructions

## Quality Gate

- Probe goal: naive GPT-5.4/high should not clear the task by doing a single token rename
- Hardening baked in before probe:
  - Hidden checks tie docs, config, and emitted report structure together
  - The validator entrypoint and output contract must both line up
  - Dirty-workspace variants require preservation of unrelated edits
- Probe outcome:
  - Adversarial GPT-5.4/high judged the initial spec too easy; a solver could likely exceed `30/100` via grep-and-rename edits plus light example text changes
- Additional hardening after probe:
  - Added executable extraction of every documented command example
  - Added explicit rejection of legacy helper aliases or shims
  - Added a scoring clamp so visible checks cannot exceed `20/100`
  - Added anti-test-weakening language to the grader contract
- Final naive GPT-5.4/high assessment:
  - Under-30 now looks plausible for a naive solver because shallow token replacement no longer clears the score clamp or hidden executable-flow checks

## V1-V5 Instantiation

- V1 clean baseline: one stale command and one stale example block
- V2 noisy reality: deprecated notes, duplicate examples, and one misleading changelog entry
- V3 dirty workspace: unrelated edits under another skill plus untracked prompt captures
- V4 multi-surface: docs, TOML, examples, and validator-facing behavior must all align
- V5 recovery-in-thread: first pass updates docs but misses the report-shape invariant or config mapping
