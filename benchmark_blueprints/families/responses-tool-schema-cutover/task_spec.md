# Responses Tool Schema Cutover

## Task Metadata
- `task_id`: `cnb55-core-responses-tool-schema-cutover-router-replay`
- `family_id`: `responses-tool-schema-cutover`
- `scenario_type`: `migration_refactor`

## Task Prompt
The repo has started consuming Responses-style event logs from the agent gateway, but the local adapter still assumes legacy chat/function-call semantics. Repair the gateway so streamed tool calls and tool results are correlated correctly, replay snapshots render the right final transcript, and the repo-local Codex config matches the new runtime contract. Keep the public CLI output stable for successful runs. Do not delete fixtures or relax assertions to get tests green.

## Workspace Bundle
- Small Python service repo rooted at `/workspace`.
- `gateway/adapter.py` still expects `function_call` plus one synthetic assistant message.
- `gateway/reducer.py` builds replay summaries from JSONL event fixtures.
- `fixtures/responses_stream/*.jsonl` contains three recorded Responses transcripts.
- `tests/test_adapter_contract.py` covers event translation and tool-result joins.
- `tests/test_replay_render.py` covers transcript rendering and stable CLI summaries.
- `codex/config.toml` pins runtime and tool-routing defaults.
- `docs/migrations/responses-cutover.md` describes the rollout contract, but it is stale.

## Seeded Breakage
- Adapter still maps `response.output_item.added` and `response.completed` into one legacy assistant blob, dropping per-call `call_id`.
- Tool results are keyed by tool name instead of `call_id`, so repeated tool invocations collapse incorrectly.
- Replay reducer suppresses duplicated tool names, which hides the visible bug on one fixture but corrupts multi-call transcripts.
- `codex/config.toml` still advertises legacy `chat_completions` semantics.
- Migration note still tells operators to look for `function_call.arguments` instead of Responses event items.

## Required Surfaces
- `shell`
- `apply_patch`
- terminal test execution
- transcript fixture inspection

## Expected Deliverables
- Code patch across adapter and replay reducer.
- Updated or added tests that protect `call_id`-scoped joins and replay rendering.
- `codex/config.toml` patch aligned to Responses runtime semantics.
- Short migration note update covering the operator-visible contract change.

## Grader Contract
- Visible gate:
  - `pytest -q tests/test_adapter_contract.py tests/test_replay_render.py`
  - One visible fixture includes two same-name tool calls with out-of-order tool results; passing requires true `call_id` correlation and transcript-order preservation for both invocations.
- Hidden checks:
  - Two-tool same-name replay where only `call_id` disambiguates the result join.
  - Out-of-order tool result arrival with correct final transcript ordering.
  - Grep check that `codex/config.toml` no longer claims legacy chat runtime.
  - Docs check that the migration note references Responses event items and `call_id`.
  - Differential oracle comparing the fixed transcript against a trusted replay render for a withheld JSONL trace.
- Fail conditions:
  - Deleting or weakening tests.
  - Collapsing repeated tool invocations into one rendered step.
  - Hardcoding fixture-specific branches for the visible transcripts.

## Red-Team Traps
- Patch the reducer to dedupe duplicate tool names instead of preserving distinct calls.
- Reconstruct `call_id` from tool name plus ordinal only for the visible fixtures.
- Force sorted output that hides arrival-order bugs while still failing hidden chronology checks.
- Update docs and config only, leaving replay logic semantically broken.
- Monkeypatch the tests to ignore repeated tool invocations.

## Quality Gate
- Probe target:
  - A naive `gpt-5.4/high` solver should land around `20/100`, not `30/100`, unless it can produce a real patch plus evidence.
- Hardening applied after adversarial solver run:
  - The task requires both code and non-code surfaces.
  - Hidden checks include repeated same-name tool calls and withheld replay traces, which invalidate obvious fixture-specific patches.
  - Replay reducer and adapter each contain part of the defect so a one-file patch should not fully pass.
  - Visible coverage now includes same-name repeated tool calls with out-of-order results and requires both invocations to survive in transcript order, not just summary text.
- Actual run result:
  - Child subagent solver attempt using `gpt-5.4/high` scored `20/100` under the family rubric.
  - The run showed good intent but no executable patch, no test output, and no artifact-backed proof.
  - Final assessment: `target hardness met; naive gpt-5.4/high stays around 20/100`.
