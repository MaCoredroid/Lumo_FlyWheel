# Skill Entrypoint Drift

- `task_id`: `t9_skill_entrypoint_drift_ops_digest`
- `target_family_id`: `skill-entrypoint-drift`

## Task Prompt
The `ops-digest` skill was partially migrated during a repo refactor from `scripts/build_digest.py` to `tools/digest_runner.py`, but the skill instructions, examples, and smoke path no longer agree. Repair the skill so a Codex agent following `SKILL.md` can run the intended workflow end to end without guessing undocumented flags or paths. Do not solve this by deleting the smoke test or reducing the skill to a vague prose stub.

## Workspace Bundle
- Internal tools repo with one skill bundle and one small runnable helper.
- Key paths:
  - `skills/ops-digest/SKILL.md`
  - `skills/ops-digest/examples/weekly_digest.md`
  - `skills/ops-digest/references/digest_contract.md`
  - `tools/digest_runner.py`
  - `tools/common/report_loader.py`
  - `scripts/smoke_skill.py`
  - `fixtures/incidents/sample_events.json`
  - `tests/test_skill_smoke.py`
  - `tests/test_digest_runner.py`
- Primary local commands:
  - `python scripts/smoke_skill.py skills/ops-digest/SKILL.md`
  - `pytest -q`

## Seeded Integration Or Plugin Drift
- `SKILL.md` still references a pre-refactor entrypoint and obsolete flag behavior.
- One example command path is now only valid from repo root, while the intended skill workflow must also survive skill-bundle-relative execution where documented.
- The runnable helper and examples drifted just enough that a doc-only refresh is insufficient.
- Trusted grading independently checks that the documented workflow, example commands, and generated markdown structure still match the live runner contract.

## Required Surfaces
- `skills`
- `shell`
- `apply_patch`
- `examples`

## Deliverables
- Updated `SKILL.md` with a runnable, accurate workflow.
- Any bounded helper-script or example fixes needed to make the skill truthful.
- Passing smoke and runner tests.
- Short note describing the refactor drift that was reconciled.

## Grader Contract
- Phase 1: execute the documented smoke path exactly as a user would from the repo root, then execute the primary command blocks copied from `SKILL.md` and `examples/weekly_digest.md` under the working directories they claim to support.
- Phase 2: run `pytest -q` and require visible skill smoke tests plus runner tests to pass.
- Phase 3 trusted checks:
  - `SKILL.md` names the canonical runnable entrypoint and valid flags accepted by the live runner.
  - The copied command blocks execute successfully in the documented working directories; repo-root-only fixes fail if the skill claims broader portability.
  - The markdown output described in the skill and example matches the runner's actual heading and summary-shape contract.
  - The task fails if the fix only edits prose while leaving executable paths, example arguments, or output claims inconsistent.
  - Edits that weaken `scripts/smoke_skill.py`, `tests/test_skill_smoke.py`, or `tests/test_digest_runner.py` to bypass execution, argument validation, or output checks receive no credit.
  - Visible smoke plus unit tests are worth at most 20/100 without independent command-block execution and contract alignment.

## Red-Team Traps
- Update `SKILL.md` to say “use the appropriate script” without naming a valid command.
- Add a compatibility shim for the old script path but leave the example arguments stale.
- Remove strict smoke assertions instead of reconciling the contract.
- Change the runner to accept invalid legacy flags while leaving docs contradictory.
- Make the example work only because tests run from repo root.

## Quality Gate
- Intended naive-solver failure mode: patch the main command in `SKILL.md`, overlook relative example paths and the renamed summary flag, then miss hidden contract checks.
- Hardening built into the task:
  - The spec no longer leaks the exact replacement command or flag values.
  - Grading checks executable truth, not just doc plausibility.
  - Hidden verification exercises both repo-root and documented skill-relative execution assumptions.
  - Output-shape assertions stop agents from swapping in a barely related command.
  - Smoke harness and visible tests are integrity-guarded against assertion weakening.
- GPT-5.4/high probe result: pre-hardening estimate `55-75/100`; easy path was near-mechanical command replacement because the draft leaked the exact new entrypoint, flag rename, and hidden rubric.
- Hardening applied after probe:
  - Removed exact fix values from the drift section.
  - Added direct execution of copied command blocks from `SKILL.md` and the example artifact.
  - Added explicit no-credit language for smoke or test weakening.
  - Capped visible smoke plus test credit below 30 unless docs, examples, and runner output agree.
- Final under-30 judgment for a naive GPT-5.4/high solver: `Yes`, now looks under 30 because the obvious checklist path was removed and most credit depends on executable contract alignment across docs and runtime.
- Observed GPT-5.4/high benchmark run: `20/100` on the visible bundle, which lands in the intended `15-25` target band.
