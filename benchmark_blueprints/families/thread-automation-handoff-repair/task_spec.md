# Thread Automation Handoff Repair

## Companion Artifacts

- `evaluator_contract.md` defines score caps, hidden invariants, and fail-fast conditions
- `codex/config.toml` defines the configured Codex profile for this family
- `skills/repair-thread-heartbeat/SKILL.md` provides bounded benchmark-local guidance
- `benchmark_run.md` records actual GPT-5.4/high benchmark attempts, scoring, and resulting hardening

## Task Prompt

Repair a Codex heartbeat automation that is supposed to wake an existing thread with a release-handoff task. A platform change split `kind`, `destination`, and prompt expectations, and the current automation now either wakes the wrong surface or asks for the wrong deliverable. Fix the automation bundle so the recurring task resumes in-thread, produces the right handoff artifact, and does not create duplicate schedules. Preserve any unrelated automation notes.

## Workspace Bundle

- `automations/release_handoff/automation.toml`
- `automations/release_handoff/prompt.md`
- `skills/release_handoff/SKILL.md`
- `tests/test_automation_contract.py`
- `tests/test_prompt_contract.py`
- `docs/automation_ops.md`
- Optional `memory.md` or operator notes for the dirty-workspace variant

## Seeded Skill Or Tooling Drift

- Automation uses stale `kind = "cron"` semantics instead of heartbeat-in-thread behavior
- Prompt text still asks the agent to create a file even though the new contract requires an inbox item or thread reply
- Docs mix old schedule language with the current thread-automation behavior
- Hidden checks detect duplicate automation creation rather than in-place repair

## Required Surfaces

- Config editing in TOML
- Prompt and skill documentation editing
- Shell-based validation
- Automation-surface reasoning about thread continuation versus separate job execution

## Deliverables

- Repaired automation TOML and prompt contract
- Any matching skill-doc correction needed to keep the workflow self-consistent
- Short operator note showing the intended wake-up behavior and duplicate-prevention rule

## Grader Contract

- Phase 1 visible checks:
  - `pytest -q tests/test_automation_contract.py tests/test_prompt_contract.py`
- Phase 2 hidden checks:
  - Validate `kind`, `destination`, and cadence semantics for a thread heartbeat
  - Ensure the prompt requests an in-thread handoff response rather than a file write or detached cron job
  - Confirm docs and skill text do not instruct creating duplicates
  - Require repair in place of the original automation bundle; reject any added replacement bundle
  - Preserve unrelated notes in `memory.md` or operator docs byte-for-byte outside the intended repair surface
  - Reject solutions that create a second automation file instead of repairing the existing one
- Score shaping:
  - Visible checks alone cap the score at `20/100`
  - Any additive “create a new automation” escape caps the score at `0/100`
  - Exceeding `30/100` requires passing hidden heartbeat-semantics, prompt-artifact, and preservation checks
- Pass condition:
  - The automation resumes work in the target thread with the right deliverable semantics and no duplication

## Red-Team Traps

- Flip `kind` only and leave the prompt asking for the wrong output mode
- Create a new automation instead of repairing the existing bundle
- Keep docs stale while making tests pass through a permissive parser
- Change cadence text only and ignore `destination`
- Remove duplicate-detection assertions

## Quality Gate

- Probe goal: naive GPT-5.4/high should not pass by making one TOML edit
- Hardening baked in before probe:
  - Prompt semantics and duplicate prevention are part of grading
  - The task requires coordinated TOML, prompt, and docs alignment
  - Hidden checks reject additive “create a second automation” escapes
- Probe outcome:
  - Adversarial GPT-5.4/high judged the initial spec too easy; a solver could likely exceed `30/100` by changing `kind`, `destination`, and a few prompt keywords
- Additional hardening after probe:
  - Added in-place-repair enforcement and duplicate-bundle rejection
  - Added preservation checks for unrelated operator notes
  - Added a score clamp so TOML-only or keyword-only edits cannot exceed `20/100`
  - Explicitly tied >`30/100` to hidden semantics and artifact-shape checks
- Final naive GPT-5.4/high assessment:
  - Under-30 now looks plausible for a naive solver because string alignment alone no longer clears the grading clamp or in-place semantic checks

## V1-V5 Instantiation

- V1 clean baseline: one wrong `kind` and one stale prompt sentence
- V2 noisy reality: paused-versus-active confusion and contradictory operator notes
- V3 dirty workspace: partial local edits in `memory.md` or adjacent automation docs
- V4 multi-surface: automation TOML, prompt, skill text, and ops note all required
- V5 recovery-in-thread: first fix gets thread destination right but still creates the wrong deliverable or duplicate behavior
