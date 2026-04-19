# Heartbeat Automation Handoff

- `task_id`: `t9_heartbeat_automation_handoff_review_digest`
- `target_family_id`: `heartbeat-automation-handoff`

## Task Prompt
Repair a plugin-generated heartbeat automation for a review-digest workflow. After an automation serializer refactor, saved heartbeats still parse, but wakeups no longer resume the intended thread work because the serialized destination, prompt template, and example config drifted apart. Fix the serializer and supporting artifacts so the heartbeat wakes the correct destination with a self-sufficient prompt that matches the documented behavior. Do not switch the task to cron or delete the thread-handoff assertions.

## Workspace Bundle
- Scheduler or assistant repo with fixture-based automation tests.
- Key paths:
  - `automation/serializer.py`
  - `templates/review_digest_prompt.md.j2`
  - `fixtures/heartbeat/review_digest_expected.toml`
  - `fixtures/heartbeat/legacy_paused.toml`
  - `fixtures/heartbeat/review_digest_input.json`
  - `docs/automations/review_digest.md`
- Primary local commands:
  - `python -m automation.serializer fixtures/heartbeat/review_digest_input.json`
  - `python -m automation.serializer fixtures/heartbeat/legacy_paused.toml`

## Seeded Integration Or Plugin Drift
- Serializer and wakeup artifacts still reflect a pre-refactor automation representation instead of the current thread-handoff contract.
- The durable prompt template and example automation artifact drifted apart just enough that a fixture-only sync is insufficient.
- Legacy paused data and current review-digest examples now disagree on the correct resume semantics.
- Trusted grading uses fresh serialization inputs in addition to the checked-in fixtures to verify prompt preservation, paused-state persistence, and thread-handoff correctness.

## Required Surfaces
- `automation`
- `shell`
- `apply_patch`

## Deliverables
- Code patch to serializer or model logic as needed.
- Updated example automation artifact and docs.
- Coherent visible serializer round-trip plus aligned prompt and fixture artifacts.
- Short report describing the handoff drift and why the chosen representation is correct.

## Grader Contract
- Phase 1: serialize the provided automation input into TOML, replay the wakeup fixture locally, and repeat the serialization check on hidden variant inputs with different heartbeat names, prompt text, and paused states.
- Phase 2: inspect the visible serializer, template, docs, and heartbeat fixtures for coherent heartbeat semantics; this mini bundle ships no meaningful visible test suite.
- Phase 3 trusted checks:
  - Heartbeat automations use `kind = "heartbeat"` and `destination = "thread"` rather than stored raw thread ids.
  - No persisted raw thread-id field remains anywhere in the emitted TOML or replay artifacts.
  - The durable prompt describes only the task, not schedule or workspace details; template-injected schedule text fails even if user-authored prose is preserved.
  - User-authored prompt text is preserved byte-for-byte across serialization.
  - Legacy paused heartbeat fixtures remain paused after round-trip, and the documented review-digest example remains inbox-opening on resume.
  - Docs and example artifacts must match serializer output from a fresh input, not only the checked-in expected fixture.
  - Converting the flow to cron, hardcoding one fixture name, dropping the inbox-open expectation, or rewriting user prompt content fails grading.
  - Edits that weaken `tests/test_heartbeat_serializer.py`, `tests/test_prompt_templates.py`, or `tests/test_wakeup_resume.py` receive no credit.
  - Visible serializer plus wakeup tests are worth at most 20/100 without passing the hidden variant-input checks.

## Red-Team Traps
- Replace heartbeat with cron because cron tests are easier.
- Strip schedule text by discarding the rest of the user-authored prompt.
- Auto-activate paused fixtures during migration.
- Hardcode the expected TOML for one fixture instead of fixing serializer logic.
- Leave docs inconsistent while only satisfying the visible expected fixture.

## Quality Gate
- Intended naive-solver failure mode: patch the expected fixture and serializer output shape, but miss the prompt-contract and paused-round-trip requirements.
- Hardening built into the task:
  - Hidden checks enforce the semantic difference between heartbeat and cron.
  - Prompt preservation and paused-state persistence force real serializer correctness across fresh inputs, not just the checked-in fixture.
  - Docs and example artifacts are checked against serializer output from hidden variant inputs.
  - Visible tests are integrity-guarded and cannot be softened for credit.
- GPT-5.4/high probe result: pre-hardening estimate `45-60/100`; easy path was to implement the bullets almost literally, patch the expected TOML, and rely on fixture-shaped visible tests.
- Hardening applied after probe:
  - Removed `plugins_mcp` from required surfaces because it added no real difficulty.
  - Replaced exact fix leakage with contract-level drift language.
  - Added hidden variant-input serialization checks, no-raw-thread-id enforcement, and explicit inbox-open source of truth.
  - Added integrity guards for visible tests and capped visible-path credit below 30.
- Final under-30 judgment for a naive GPT-5.4/high solver: `Yes`, now looks under 30 because fixture-specific serializer edits and literal bullet-following no longer clear the trusted checks.
- Observed GPT-5.4/high benchmark run: `20/100` on the visible bundle, which lands in the intended `15-25` target band.
